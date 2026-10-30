import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from dev.benchmarks import http_regression
from dev.benchmarks import live_probe as probe

API_KEY = "probe-key"
MODEL = "incoai/Qwen3.8-27B-Splash"
REPLY_CONTENT = "1\n2"


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
                        "choices": [
                            {
                                "message": {"content": REPLY_CONTENT},
                                "finish_reason": "length",
                            }
                        ],
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


STREAM_ID = "abc123"
# The counting reply up to the default stop sequence, as the server sends it.
STOP_CONTENT = probe.STOP_REFERENCE[
    : probe.STOP_REFERENCE.index(probe.DEFAULT_STOP_SEQUENCE)
]
# Main-thread clock readings, in call order: request start, one per stream
# event (role, two content deltas, finish, usage) and request end.
STREAM_CLOCK = (10.0, 10.5, 11.0, 11.25, 11.375, 11.5, 11.75)
STREAM_STOP_TO_DONE_MS = 125.0
STREAM_WALL_MS = 1750.0


def request_record(draft_cycles=10, decode_steps=20, decode_wall_ms=400.0):
    """Return the stop request's `GET /requests` record."""
    return {
        "id": STREAM_ID,
        "draft_cycles": draft_cycles,
        "engine": {"decode_wall_ms": decode_wall_ms, "decode_steps": decode_steps},
    }


def stream_events(content=STOP_CONTENT, finish_reason="stop", completion_tokens=38):
    """Return the chunks of one streamed chat completion, content in two deltas."""
    middle = len(content) // 2

    def chunk(delta, finish=None):
        return {
            "id": f"chatcmpl-{STREAM_ID}",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    usage = {"prompt_tokens": 20, "completion_tokens": completion_tokens}
    return [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": content[:middle]}),
        chunk({"content": content[middle:]}),
        chunk({}, finish_reason),
        {"id": f"chatcmpl-{STREAM_ID}", "choices": [], "usage": usage},
    ]


class StreamStubServer:
    """Serve a scripted SSE chat completion and its `GET /requests` record.

    The record is listed from the `record_after`-th `/requests` poll on (0:
    the first); None never lists it.
    """

    def __init__(self, events, record=None, record_after=0):
        self.events = events
        self.record = record if record is not None else request_record()
        self.record_after = record_after
        self.requests_polls = 0
        self.chat_bodies = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def do_GET(self):
                listed = []
                if stub.record_after is not None:
                    if stub.requests_polls >= stub.record_after:
                        listed = [{"id": "other"}, stub.record]
                stub.requests_polls += 1
                raw = json.dumps({"requests": listed}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                stub.chat_bodies.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": keepalive\n\n")
                for event in stub.events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

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


def scripted_clock(readings):
    """Return a monotonic clock giving `readings` to the main thread in order.

    Other threads (the stub server's) read the real clock.
    """
    real = probe.time.monotonic
    main = threading.current_thread()
    remaining = iter(readings)

    def clock():
        if threading.current_thread() is not main:
            return real()
        return next(remaining)

    return clock


def run_stop_main(stub, *extra):
    """Run the probe with `--stop-sequence` against `stub` on the scripted clock.

    Returns (exit code, stdout, stderr).
    """
    errors = io.StringIO()
    arguments = ["--base-url", stub.base_url, "--model", MODEL, "--stop-sequence"]
    with (
        mock.patch.object(probe.time, "monotonic", scripted_clock(STREAM_CLOCK)),
        contextlib.redirect_stderr(errors),
    ):
        code, output = run_main([*arguments, *extra])
    return code, output, errors.getvalue()


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

    def test_main_record_carriesFinishReasonAndContent(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        _stub, document = self.probe_json(statuses)

        for record in document["requests"]:
            self.assertEqual(record["finish_reason"], "length")
            self.assertEqual(record["content"], REPLY_CONTENT)

    def test_main_promptFile_sendsFileTextOnceWithoutRepeat(self):
        prompt = "def last():\n    return 1\nIn one sentence, what does it do?"
        statuses = [snapshot(), COLD_AFTER]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file:
            file.write(prompt)
        self.addCleanup(os.remove, file.name)

        with StubServer(statuses) as stub:
            code, output = run_main(
                [
                    "--base-url",
                    stub.base_url,
                    "--model",
                    MODEL,
                    "--prompt-file",
                    file.name,
                    "--prompt-tokens",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(stub.chat_bodies), 1)
        self.assertEqual(
            stub.chat_bodies[0]["messages"], [{"role": "user", "content": prompt}]
        )
        (record,) = json.loads(output)["requests"]
        self.assertEqual(record["request"], "cold")
        self.assertEqual(record["prefill_tokens_per_second"], 2048.0)
        self.assertEqual(record["finish_reason"], "length")

    def test_main_promptFileMissing_failsBeforeAnyRequest(self):
        with StubServer([snapshot()]) as stub:
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                code, _output = run_main(
                    ["--base-url", stub.base_url, "--prompt-file", "/no/such/file"]
                )

        self.assertEqual(code, 1)
        self.assertIn("--prompt-file", errors.getvalue())
        self.assertEqual(stub.chat_bodies, [])

    def test_main_table_rendersWithRepliesBelowIt(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        with StubServer(statuses) as stub:
            code, output = run_main(["--base-url", stub.base_url, "--model", MODEL])

        self.assertEqual(code, 0)
        lines = output.splitlines()
        table_rows = lines[2 : 2 + len(probe.REPORT_ROWS)]
        self.assertEqual(
            [row.split("  ")[0].strip() for row in table_rows],
            [label for _key, label in probe.REPORT_ROWS],
        )
        self.assertEqual(
            lines[2 + len(probe.REPORT_ROWS) :],
            [
                f"cold: finish_reason length, content {REPLY_CONTENT!r}",
                f"repeat: finish_reason length, content {REPLY_CONTENT!r}",
            ],
        )

    def test_coldPrompt_nonceLeadsAndChangesThePrompt(self):
        first = probe.cold_prompt(64, 16, "aaaa")
        second = probe.cold_prompt(64, 16, "bbbb")

        self.assertTrue(first.startswith("aaaa\n"))
        self.assertNotEqual(first[:4], second[:4])
        self.assertEqual(first, probe.cold_prompt(64, 16, "aaaa"))


class StopSequenceTests(unittest.TestCase):
    def stop_json(self, stub):
        code, output, errors = run_stop_main(stub, "--json")
        (record,) = json.loads(output)["requests"]
        return code, record, errors

    def test_main_stopSequence_reportsStopToDoneAndCycle(self):
        with StreamStubServer(stream_events()) as stub:
            code, record, _errors = self.stop_json(stub)

        self.assertEqual(code, 0)
        self.assertEqual(record["request"], "stop")
        self.assertEqual(record["stop_to_done_ms"], STREAM_STOP_TO_DONE_MS)
        self.assertEqual(record["cycle_ms"], 40.0)
        self.assertEqual(record["http_wall_ms"], STREAM_WALL_MS)
        self.assertEqual(record["content"], STOP_CONTENT)

    def test_main_stopSequence_streamsTheCountingPromptWithTheStop(self):
        with StreamStubServer(stream_events()) as stub:
            self.stop_json(stub)

        (body,) = stub.chat_bodies
        self.assertEqual(body["stop"], [probe.DEFAULT_STOP_SEQUENCE])
        self.assertEqual(body["messages"][0]["content"], probe.STOP_PROMPT)
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["reasoning_effort"], "none")

    def test_main_stopRecord_carriesEveryReportRowKey(self):
        with StreamStubServer(stream_events()) as stub:
            _code, record, _errors = self.stop_json(stub)

        for key, _label in probe.REPORT_ROWS + probe.STOP_ROWS:
            self.assertIn(key, record)
        self.assertEqual(record["prompt_tokens"], 20)
        self.assertEqual(record["completion_tokens"], 38)
        self.assertIsNone(record["decode_tokens_per_second"])
        self.assertIsNone(record["host_wired_bytes"])

    def test_main_noDraftCycles_dividesDecodeWallBySteps(self):
        record = request_record(draft_cycles=0, decode_steps=20)

        with StreamStubServer(stream_events(), record=record) as stub:
            _code, reported, _errors = self.stop_json(stub)

        self.assertEqual(reported["cycle_ms"], 20.0)

    def test_main_requestRecordLate_isStillFound(self):
        with StreamStubServer(stream_events(), record_after=3) as stub:
            code, record, _errors = self.stop_json(stub)

        self.assertEqual(code, 0)
        self.assertEqual(record["cycle_ms"], 40.0)
        self.assertEqual(stub.requests_polls, 4)

    def test_main_requestRecordNeverArrives_reportsCycleUnavailable(self):
        with StreamStubServer(stream_events(), record_after=None) as stub:
            code, record, errors = self.stop_json(stub)

        self.assertEqual(code, 0)
        self.assertIsNone(record["cycle_ms"])
        self.assertIn("cycle ms unavailable", errors)
        self.assertEqual(stub.requests_polls, probe.REQUEST_RECORD_POLLS + 1)

    def test_main_finishReasonLength_exitsNonZero(self):
        events = stream_events(finish_reason="length", completion_tokens=128)

        with StreamStubServer(events) as stub:
            code, record, errors = self.stop_json(stub)

        self.assertEqual(code, 1)
        self.assertEqual(record["finish_reason"], "length")
        self.assertIn("not a stop-sequence finish", errors)

    def test_main_stopWithoutStopMatchEvidence_exitsNonZero(self):
        cases = {
            "EOS before the stop text": stream_events(content="Sure, 1, 2, 3."),
            "completion at max_tokens": stream_events(completion_tokens=128),
        }
        for name, events in cases.items():
            with self.subTest(name), StreamStubServer(events) as stub:
                code, _record, errors = self.stop_json(stub)

                self.assertEqual(code, 1)
                self.assertIn("not a stop-sequence finish", errors)

    def test_main_stopSequenceTable_addsStopRowsBelowReportRows(self):
        with StreamStubServer(stream_events()) as stub:
            code, output, _errors = run_stop_main(stub)

        self.assertEqual(code, 0)
        rows = probe.REPORT_ROWS + probe.STOP_ROWS
        lines = output.splitlines()
        self.assertEqual(lines[1].split(), ["stop"])
        self.assertEqual(
            [row.split("  ")[0].strip() for row in lines[2 : 2 + len(rows)]],
            [label for _key, label in rows],
        )
        self.assertEqual(lines[2 + len(rows) - 2].split()[-1], "125")

    def test_main_withoutStopSequence_recordsCarryNoStopKeys(self):
        statuses = [snapshot(), COLD_AFTER, COLD_AFTER, REPEAT_AFTER]

        with StubServer(statuses) as stub:
            code, output = run_main(
                ["--base-url", stub.base_url, "--model", MODEL, "--json"]
            )

        self.assertEqual(code, 0)
        for record in json.loads(output)["requests"]:
            for key, _label in probe.STOP_ROWS:
                self.assertNotIn(key, record)

    def test_parseArgs_stopSequence_takesValueOrDefault(self):
        base = ["--base-url", "http://127.0.0.1:1"]

        self.assertIsNone(probe.parse_args(base).stop_sequence)
        self.assertEqual(
            probe.parse_args([*base, "--stop-sequence"]).stop_sequence,
            probe.DEFAULT_STOP_SEQUENCE,
        )
        self.assertEqual(
            probe.parse_args([*base, "--stop-sequence", ", 30,"]).stop_sequence,
            ", 30,",
        )

    def test_parseArgs_unusableStopSequence_fails(self):
        base = ["--base-url", "http://127.0.0.1:1", "--stop-sequence"]
        cases = {
            "absent from the counting reply": [*base, "seventy"],
            "at its very start": [*base, "1, 2"],
            "with --prompt-file": [*base, ", 20,", "--prompt-file", "prompt.txt"],
        }
        for name, arguments in cases.items():
            with (
                self.subTest(name),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                probe.parse_args(arguments)


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
