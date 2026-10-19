import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from dev.benchmarks import pressure_probe as probe

MODEL = "incoai/Qwen3.8-27B-Splash"
PAGE_BYTES = 16384
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Swapins:                                      7.
Swapouts:                                  {swapouts}.
"""


def snapshot(pressure="normal", **counters):
    """Return a `/status` document; counters are given as section__key=value."""
    document = {
        "transport": {"status_stale": False},
        "memory_pressure": pressure,
        "metrics": {},
        "cache": {},
        "state": {},
        "memory_governor": {},
    }
    for path, value in counters.items():
        section, key = path.split("__")
        document.setdefault(section, {})[key] = value
    return document


class StubServer:
    """Serve one `/status` document and a chat reply that reports cached tokens."""

    def __init__(self, status, cached_tokens=0, reply_delay=0.0):
        self.status = status
        self.released = threading.Event()
        self.chat_bodies = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def reply(self, document):
                raw = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply({"data": [{"id": MODEL}]})
                    return
                self.reply(stub.status)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                stub.chat_bodies.append(json.loads(self.rfile.read(length)))
                stub.released.wait(reply_delay)
                self.reply(
                    {
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 8,
                            "prompt_tokens_details": {"cached_tokens": cached_tokens},
                        }
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exception):
        self.released.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class CounterDeltaTests(unittest.TestCase):
    def test_counterDelta_diffsExportedCounters(self):
        before = snapshot(cache__resource_suspensions=1, state__evictions=4)
        after = snapshot(cache__resource_suspensions=3, state__evictions=8)

        delta = probe.counter_delta(before, after)

        self.assertEqual(delta["suspensions"], 2)
        self.assertEqual(delta["state_evictions"], 4)

    def test_counterDelta_missingCounter_readsAsNoneNotZero(self):
        delta = probe.counter_delta(snapshot(), snapshot(state__evictions=2))

        self.assertIsNone(delta["state_evictions"])
        self.assertIsNone(delta["kv_evictions"])

    def test_counterDelta_diffsEachCheckpointFailureCause(self):
        before = snapshot()
        before["cache"]["checkpoint_publication_failure_causes"] = {
            cause: 1 for cause in probe.CHECKPOINT_FAILURE_CAUSES
        }
        after = snapshot()
        after["cache"]["checkpoint_publication_failure_causes"] = {
            cause: 1 + index
            for index, cause in enumerate(probe.CHECKPOINT_FAILURE_CAUSES)
        }
        del after["cache"]["checkpoint_publication_failure_causes"]["error"]

        delta = probe.counter_delta(before, after)

        self.assertEqual(
            [
                delta[f"checkpoint_failures_{cause}"]
                for cause in probe.CHECKPOINT_FAILURE_CAUSES
            ],
            [0, 1, 2, 3, None],
        )


class LaneRecordTests(unittest.TestCase):
    def test_laneRecord_readsCachedTokensFromUsage(self):
        reply = {
            "usage": {
                "prompt_tokens": 57860,
                "completion_tokens": 1024,
                "prompt_tokens_details": {"cached_tokens": 45056},
            }
        }

        record = probe.lane_record(1, "burst", reply, 1247.34, None)

        self.assertEqual(record["cached_tokens"], 45056)
        self.assertEqual(record["prompt_tokens"], 57860)
        self.assertEqual(record["wall_s"], 1247.3)

    def test_laneRecord_failedRequest_keepsErrorAndNoCounts(self):
        record = probe.lane_record(0, "warm0", None, 2.0, "HTTP 503")

        self.assertEqual(record["error"], "HTTP 503")
        self.assertIsNone(record["cached_tokens"])


class PressureSummaryTests(unittest.TestCase):
    def sample(self, at, server, os_level, **readings):
        return {"t": at, "server_pressure": server, "os_level": os_level, **readings}

    def test_pressureSummary_timeInStateCountsIntervalsStartingInIt(self):
        samples = [
            self.sample(0, "normal", 1),
            self.sample(3, "normal", 2),
            self.sample(6, "warning", 2),
            self.sample(10, "warning", 1),
            self.sample(13, "normal", 1),
        ]

        summary = probe.pressure_summary(samples)

        self.assertEqual(summary["server_warning_s"], 7)
        self.assertEqual(summary["os_warn_s"], 7)
        self.assertEqual(summary["os_warn_while_server_normal_s"], 3)
        self.assertEqual(summary["server_flips"], 2)

    def test_pressureSummary_minimaSkipMissingReadingsAndErrorRows(self):
        samples = [
            self.sample(0, "normal", 1, host_available_bytes=5 * 2**30),
            {"t": 1, "error": "timed out"},
            self.sample(2, "warning", 2, host_available_bytes=2**30),
            self.sample(3, "warning", 2, host_available_bytes=None),
        ]

        summary = probe.pressure_summary(samples)

        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["min_host_available_bytes"], 2**30)
        self.assertIsNone(summary["min_governor_headroom_bytes"])

    def test_statusSample_keepsServerAndOsPressure(self):
        status = snapshot("warning", memory_governor__host_headroom_bytes=7)

        sample = probe.status_sample(status, 2, "burst", 5.0)

        self.assertEqual(sample["server_pressure"], "warning")
        self.assertEqual(sample["os_level"], 2)
        self.assertEqual(sample["phase"], "burst")
        self.assertEqual(sample["host_headroom_bytes"], 7)
        self.assertIsNone(sample["suspended"])


class SwapGuardTests(unittest.TestCase):
    def readings(self, *swapouts):
        values = iter(swapouts)
        return lambda: (PAGE_BYTES, next(values))

    def test_parseVmStat_readsPageSizeAndSwapouts(self):
        self.assertEqual(
            probe.parse_vm_stat(VM_STAT.format(swapouts=4242)), (PAGE_BYTES, 4242)
        )

    def test_parseVmStat_unexpectedOutput_raises(self):
        with self.assertRaises(probe.ProbeError):
            probe.parse_vm_stat("nothing here")

    def test_check_growthAtCap_doesNotTrip(self):
        cap = 512 * 2**20
        guard = probe.SwapGuard(cap, self.readings(100, 100 + cap // PAGE_BYTES))

        self.assertFalse(guard.check())
        self.assertEqual(guard.grown_bytes, cap)

    def test_check_growthPastCap_tripsAndStaysTripped(self):
        cap = 512 * 2**20
        pages = cap // PAGE_BYTES
        guard = probe.SwapGuard(cap, self.readings(100, 101 + pages, 100))

        self.assertTrue(guard.check())
        self.assertTrue(guard.check())
        self.assertTrue(guard.tripped.is_set())

    def test_watch_returnsOnceTripped(self):
        guard = probe.SwapGuard(PAGE_BYTES, self.readings(0, 1, 2))
        stop = threading.Event()

        with mock.patch.object(probe, "SWAP_POLL_SECONDS", 0):
            guard.watch(stop)

        self.assertTrue(guard.tripped.is_set())


class PromptTests(unittest.TestCase):
    def test_burstPrompt_extendsTheWarmWords(self):
        warm = probe.lane_words("aaaa", 50)
        burst = probe.lane_words("aaaa", 80)

        self.assertEqual(burst[:50], warm)
        self.assertNotEqual(probe.lane_words("bbbb", 50), warm)


def run_main(arguments, swapouts=(0,)):
    readings = iter(list(swapouts) + [swapouts[-1]] * 1000)
    output = io.StringIO()
    with (
        contextlib.redirect_stdout(output),
        contextlib.redirect_stderr(io.StringIO()),
        mock.patch.object(probe, "read_vm_stat", lambda: (PAGE_BYTES, next(readings))),
        mock.patch.object(probe, "read_os_pressure_level", lambda: 2),
        mock.patch.object(probe, "SWAP_POLL_SECONDS", 0.01),
    ):
        code = probe.main(arguments)
    return code, output.getvalue()


class MainTests(unittest.TestCase):
    def test_main_warmsEachLaneThenBurstsAll(self):
        with StubServer(snapshot("warning"), cached_tokens=40) as stub:
            code, output = run_main(
                [
                    "--base-url",
                    stub.base_url,
                    "--lanes",
                    "2",
                    "--warm-tokens",
                    "80",
                    "--extend-tokens",
                    "20",
                    "--max-tokens",
                    "16",
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        document = json.loads(output)
        self.assertEqual(
            [p["phase"] for p in document["phases"]], ["warm0", "warm1", "burst"]
        )
        burst = document["phases"][2]
        self.assertEqual([lane["cached_tokens"] for lane in burst["lanes"]], [40, 40])
        self.assertEqual(len(stub.chat_bodies), 4)
        warm_prompt = stub.chat_bodies[0]["messages"][0]["content"]
        burst_prompts = [b["messages"][0]["content"] for b in stub.chat_bodies[2:]]
        self.assertTrue(
            any(p.startswith(warm_prompt.rsplit("\n", 1)[0]) for p in burst_prompts)
        )
        self.assertEqual(
            stub.chat_bodies[0]["max_completion_tokens"], probe.WARM_MAX_TOKENS
        )
        self.assertEqual(stub.chat_bodies[2]["max_completion_tokens"], 16)

    def test_main_givenNonces_skipsWarmUp(self):
        with StubServer(snapshot()) as stub:
            code, output = run_main(
                ["--base-url", stub.base_url, "--nonces", "aaaa", "bbbb", "--json"]
            )

        self.assertEqual(code, 0)
        document = json.loads(output)
        self.assertEqual(document["nonces"], ["aaaa", "bbbb"])
        self.assertEqual([p["phase"] for p in document["phases"]], ["burst"])
        self.assertEqual(len(stub.chat_bodies), 2)

    def test_main_swapGrowthPastCap_stopsMidWarmUpWithExitCode3(self):
        grown = 2 * 2**20 // PAGE_BYTES
        with StubServer(snapshot(), reply_delay=30.0) as stub:
            code, output = run_main(
                [
                    "--base-url",
                    stub.base_url,
                    "--lanes",
                    "2",
                    "--warm-tokens",
                    "80",
                    "--swap-guard-mb",
                    "1",
                ],
                swapouts=(0, grown),
            )

        self.assertEqual(code, probe.SWAP_ABORT_EXIT_CODE)
        self.assertIn("swap guard tripped", output)
        self.assertNotIn("burst", output)
        self.assertLessEqual(len(stub.chat_bodies), 1)


if __name__ == "__main__":
    unittest.main()
