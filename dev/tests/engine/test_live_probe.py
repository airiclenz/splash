import contextlib
import copy
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from dev.benchmarks import http_regression
from dev.benchmarks import live_probe as probe

API_KEY = "probe-key"
MODEL = "incoai/Qwen3.8-27B-Splash"


def snapshot(**counters):
    """Return a `/status` document whose diffed counters are all 0 unless given."""
    document = {
        "transport": {"status_stale": False},
        "metrics": {
            "prefill_input_tokens": 0,
            "prefill_wall_ms": 0,
            "decode_output_tokens": 0,
            "decode_wall_ms": 0,
            "drafted_tokens": 0,
            "accepted_draft_tokens": 0,
        },
        "scheduler": {"prefill_rows": 0, "prefill_batches": 0},
        "cache": {
            "reused_tokens": 0,
            "resource_suspensions": 0,
            "resource_replay_tokens": 0,
        },
        "memory_governor": {"denied_reservations": 0},
    }
    for path, value in counters.items():
        section, key = path.split("__")
        document[section][key] = value
    return document


# Cold: 2048 prompt tokens prefilled in 1000 ms over 4 chunks, 128 decoded in
# 2000 ms with 100 of 160 drafts accepted, one suspension and its replay.
COLD_AFTER = snapshot(
    metrics__prefill_input_tokens=2048,
    metrics__prefill_wall_ms=1000,
    metrics__decode_output_tokens=128,
    metrics__decode_wall_ms=2000,
    metrics__drafted_tokens=160,
    metrics__accepted_draft_tokens=100,
    scheduler__prefill_rows=2048,
    scheduler__prefill_batches=4,
    cache__resource_suspensions=1,
    cache__resource_replay_tokens=512,
    memory_governor__denied_reservations=2,
)
# Repeat: 2040 tokens matched, 8 prefilled in 4 ms in one chunk, 128 decoded.
REPEAT_AFTER = snapshot(
    metrics__prefill_input_tokens=2056,
    metrics__prefill_wall_ms=1004,
    metrics__decode_output_tokens=256,
    metrics__decode_wall_ms=3000,
    metrics__drafted_tokens=320,
    metrics__accepted_draft_tokens=220,
    scheduler__prefill_rows=2056,
    scheduler__prefill_batches=5,
    cache__reused_tokens=2040,
    cache__resource_suspensions=1,
    cache__resource_replay_tokens=512,
    memory_governor__denied_reservations=2,
)

# The governor's host page sample as a server that reports it emits it.
HOST_SAMPLE = {
    "host_available_bytes": 6 * 2**30,
    "host_inactive_bytes": 3 * 2**30,
    "host_wired_bytes": 5 * 2**30,
    "host_compressor_bytes": 2**30,
}
REPEAT_AFTER_WITH_HOST = copy.deepcopy(REPEAT_AFTER)
REPEAT_AFTER_WITH_HOST["memory_governor"].update(HOST_SAMPLE)


class StubServer:
    """Serve canned `/status` snapshots in order and a fixed chat reply."""

    def __init__(self, statuses, refuses_effort=False):
        self.statuses = list(statuses)
        self.refuses_effort = refuses_effort
        self.status_requests = 0
        self.chat_bodies = []
        self.authorizations = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def reply(self, document, status=200):
                raw = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                stub.authorizations.append(self.headers.get("Authorization"))
                if self.path == "/v1/models":
                    self.reply({"object": "list", "data": [{"id": MODEL}]})
                    return
                index = min(stub.status_requests, len(stub.statuses) - 1)
                stub.status_requests += 1
                self.reply(stub.statuses[index])

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                stub.chat_bodies.append(body)
                if stub.refuses_effort and "reasoning_effort" in body:
                    message = (
                        "chat template does not support the requested thinking mode"
                    )
                    self.reply({"error": {"message": message}}, status=400)
                    return
                self.reply(
                    {
                        "choices": [{"message": {"content": "1\n2"}}],
                        "usage": {"prompt_tokens": 2048, "completion_tokens": 128},
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exception):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def run_main(arguments):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), mock.patch.object(probe.time, "sleep"):
        code = probe.main(arguments)
    return code, output.getvalue()


class LiveProbeTests(unittest.TestCase):
    def probe_json(self, statuses, refuses_effort=False):
        with StubServer(statuses, refuses_effort) as stub:
            code, output = run_main(
                [
                    "--base-url",
                    stub.base_url,
                    "--api-key",
                    API_KEY,
                    "--prompt-tokens",
                    "64",
                    "--max-tokens",
                    "128",
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        return stub, json.loads(output)

    def test_main_coldThenRepeat_reportsRatesFromStatusDiff(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        _stub, document = self.probe_json(statuses)

        cold, repeat = document["requests"]
        self.assertEqual(document["model"], MODEL)
        self.assertEqual(cold["prefill_tokens_per_second"], 2048.0)
        self.assertEqual(cold["decode_tokens_per_second"], 64.0)
        self.assertEqual(cold["draft_acceptance"], 0.625)
        self.assertEqual(cold["average_prefill_chunk_rows"], 512.0)
        self.assertEqual(cold["cache_matched_tokens"], 0)
        self.assertEqual(cold["resource_suspensions"], 1)
        self.assertEqual(cold["replay_tokens"], 512)
        self.assertEqual(cold["denied_reservations"], 2)
        self.assertEqual(repeat["prefill_tokens_per_second"], 2000.0)
        self.assertEqual(repeat["decode_tokens_per_second"], 128.0)
        self.assertEqual(repeat["draft_acceptance"], 0.75)
        self.assertEqual(repeat["average_prefill_chunk_rows"], 8.0)
        self.assertEqual(repeat["resource_suspensions"], 0)

    def test_main_repeatRequest_reportsCacheMatch(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        stub, document = self.probe_json(statuses)

        self.assertEqual(document["requests"][1]["cache_matched_tokens"], 2040)
        self.assertEqual(stub.chat_bodies[0], stub.chat_bodies[1])
        self.assertEqual(stub.chat_bodies[0]["model"], MODEL)
        self.assertEqual(set(stub.authorizations), {f"Bearer {API_KEY}"})

    def test_main_templateCannotDisableThinking_probesWithTemplateDefault(self):
        statuses = [snapshot(), snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        with contextlib.redirect_stderr(io.StringIO()):
            stub, document = self.probe_json(statuses, refuses_effort=True)

        self.assertEqual(stub.chat_bodies[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", stub.chat_bodies[1])
        self.assertEqual(stub.chat_bodies[1], stub.chat_bodies[2])
        cold, repeat = document["requests"]
        self.assertEqual(cold["prefill_tokens_per_second"], 2048.0)
        self.assertEqual(repeat["cache_matched_tokens"], 2040)

    def test_main_otherRefusal_failsWithoutRetry(self):
        with StubServer([snapshot()], refuses_effort=True) as stub:
            with (
                mock.patch.object(probe, "THINKING_MODE_REFUSAL", "no such message"),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code, _output = run_main(["--base-url", stub.base_url])

        self.assertEqual(code, 1)
        self.assertEqual(len(stub.chat_bodies), 1)

    def test_main_staleSnapshot_isRetriedForAFreshOne(self):
        stale = copy.deepcopy(COLD_AFTER)
        stale["transport"]["status_stale"] = True
        stale["metrics"]["prefill_wall_ms"] = 999999
        statuses = [snapshot(), stale, COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        _stub, document = self.probe_json(statuses)

        self.assertEqual(document["requests"][0]["prefill_tokens_per_second"], 2048.0)

    def test_main_statusStaysStale_retriesAtMostThreeTimesThenFails(self):
        stale = snapshot()
        stale["transport"]["status_stale"] = True

        with StubServer([stale]) as stub:
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                code, _output = run_main(
                    ["--base-url", stub.base_url, "--model", MODEL]
                )

        self.assertEqual(code, 1)
        self.assertEqual(stub.status_requests, probe.STALE_SNAPSHOT_RETRIES + 1)
        self.assertIn("stale", errors.getvalue())
        self.assertEqual(stub.chat_bodies, [])

    def test_main_zeroWallDelta_reportsRateUnavailable(self):
        statuses = [snapshot(), snapshot(), snapshot(), REPEAT_AFTER]

        with StubServer(statuses) as stub:
            code, output = run_main(["--base-url", stub.base_url, "--model", MODEL])

        self.assertEqual(code, 0)
        prefill_row = next(
            line for line in output.splitlines() if line.startswith("prefill tok/s")
        )
        self.assertEqual(prefill_row.split()[-2], probe.UNAVAILABLE)

    def test_main_hostSample_reportsAfterSnapshotReadings(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER_WITH_HOST]

        _stub, document = self.probe_json(statuses)

        cold, repeat = document["requests"]
        for key, value in HOST_SAMPLE.items():
            self.assertEqual(repeat[key], value)
            self.assertIsNone(cold[key])

    def test_main_hostSample_tableShowsReadings(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER_WITH_HOST]

        with StubServer(statuses) as stub:
            code, output = run_main(["--base-url", stub.base_url, "--model", MODEL])

        self.assertEqual(code, 0)
        rows = {
            label: next(line for line in output.splitlines() if line.startswith(label))
            for label in (
                "host available bytes",
                "host inactive bytes",
                "host wired bytes",
                "host compressor bytes",
            )
        }
        self.assertEqual(rows["host available bytes"].split()[-1], str(6 * 2**30))
        self.assertEqual(rows["host inactive bytes"].split()[-1], str(3 * 2**30))
        self.assertEqual(rows["host wired bytes"].split()[-1], str(5 * 2**30))
        self.assertEqual(rows["host compressor bytes"].split()[-1], str(2**30))

    def test_main_statusWithoutHostSample_tableShowsUnavailable(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        with StubServer(statuses) as stub:
            code, output = run_main(["--base-url", stub.base_url, "--model", MODEL])

        self.assertEqual(code, 0)
        wired_row = next(
            line for line in output.splitlines() if line.startswith("host wired bytes")
        )
        self.assertEqual(wired_row.split()[-2:], [probe.UNAVAILABLE] * 2)

    def test_coldPrompt_nonceLeadsAndChangesThePrompt(self):
        first = probe.cold_prompt(64, 16, "aaaa")
        second = probe.cold_prompt(64, 16, "bbbb")

        self.assertTrue(first.startswith("aaaa\n"))
        self.assertNotEqual(first[:4], second[:4])
        self.assertEqual(first, probe.cold_prompt(64, 16, "aaaa"))


class StatusDeltaTests(unittest.TestCase):
    def test_statusDelta_missingCounters_readAsZero(self):
        before = {"metrics": {"decode_wall_ms": 5}}
        after = {"metrics": {"decode_wall_ms": 12, "drafted_tokens": 3}}
        paths = (
            ("metrics", "decode_wall_ms"),
            ("metrics", "drafted_tokens"),
            ("cache", "resource_suspensions"),
        )

        delta = http_regression.status_delta(before, after, paths)

        self.assertEqual(
            delta,
            {
                ("metrics", "decode_wall_ms"): 7,
                ("metrics", "drafted_tokens"): 3,
                ("cache", "resource_suspensions"): 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
