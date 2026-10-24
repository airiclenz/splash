import argparse
import contextlib
import io
import itertools
import json
import multiprocessing
import threading
import unittest
import zlib
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
REPLY_CONTENT = "1\n2\n3"
TTFT_MS = 412.5
GIB = 2**30
MIB = 2**20


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
    """Serve one `/status` document and a chat reply that reports cached tokens.

    `status` is a document, or a callable taking the count of chat requests
    received so far and returning one. `cached_tokens` is one count for every
    reply, or a list read one entry per chat request in arrival order.
    """

    def __init__(
        self, status, cached_tokens=0, reply_delay=0.0, reply_content=REPLY_CONTENT
    ):
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
                status = stub.status
                if callable(status):
                    status = status(len(stub.chat_bodies))
                self.reply(status)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                stub.chat_bodies.append(json.loads(self.rfile.read(length)))
                index = len(stub.chat_bodies) - 1
                stub.released.wait(reply_delay)
                cached = (
                    cached_tokens[index]
                    if isinstance(cached_tokens, list)
                    else cached_tokens
                )
                self.reply(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": reply_content,
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 8,
                            "prompt_tokens_details": {"cached_tokens": cached},
                        },
                        "metrics": {
                            "request_latency": {"start_to_first_token_ms": TTFT_MS}
                        },
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

    def test_counterDelta_diffsResumeRestoreDepthAndEvictedResumeStates(self):
        before = snapshot(
            cache__resource_resume_restored_tokens=4096,
            cache__resource_resume_state_evictions=1,
        )
        after = snapshot(
            cache__resource_resume_restored_tokens=12288,
            cache__resource_resume_state_evictions=3,
        )

        delta = probe.counter_delta(before, after)

        self.assertEqual(delta["resume_restored_tokens"], 8192)
        self.assertEqual(delta["resume_state_evictions"], 2)

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

    def test_counterDelta_diffsEachDenialCause(self):
        before = snapshot(
            memory_governor__denied_reservations=2,
            memory_governor__denied_host_reservations=1,
            memory_governor__denied_engine_reservations=1,
        )
        after = snapshot(
            memory_governor__denied_reservations=7,
            memory_governor__denied_host_reservations=5,
            memory_governor__denied_engine_reservations=2,
        )

        delta = probe.counter_delta(before, after)

        self.assertEqual(
            [
                delta["denied_reservations"],
                delta["denied_host_reservations"],
                delta["denied_engine_reservations"],
            ],
            [5, 4, 1],
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

    def test_laneRecord_readsServerTtftFromMetrics(self):
        reply = {"metrics": {"request_latency": {"start_to_first_token_ms": 1830.5}}}

        record = probe.lane_record(0, "turn1", reply, 2.0, None)

        self.assertEqual(record["ttft_ms"], 1830.5)

    def test_laneRecord_readsQueueTimeFromMetrics(self):
        reply = {
            "metrics": {
                "request_latency": {
                    "start_to_first_token_ms": 1356.1,
                    "queue_to_start_ms": 956.7,
                }
            }
        }

        record = probe.lane_record(0, "burst", reply, 2.0, None)

        self.assertEqual(record["queue_ms"], 956.7)
        self.assertEqual(record["ttft_ms"], 1356.1)

    def test_statusSample_readsSchedulingWait(self):
        snapshot = {"admission": {"waiting_schedule": 2, "waiting_memory": 0}}

        sample = probe.status_sample(snapshot, "normal", "burst", 1.0)

        self.assertEqual(sample["waiting_schedule"], 2)

    def test_laneRecord_failedRequest_keepsErrorAndNoCounts(self):
        record = probe.lane_record(0, "warm0", None, 2.0, "HTTP 503")

        self.assertEqual(record["error"], "HTTP 503")
        self.assertIsNone(record["cached_tokens"])

    def test_sendLane_failedRequest_printsTheErrorAtOnce(self):
        client = mock.Mock()
        client.request.side_effect = probe.ProbeError("HTTP 503: engine stopped")
        records = []
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            probe.send_lane(client, 2, "burst", {}, 1.0, records)

        self.assertEqual(records[0]["error"], "HTTP 503: engine stopped")
        self.assertIn("lane 2 burst: HTTP 503: engine stopped", stderr.getvalue())


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
    """Run main against stubbed vm_stat; `swapouts` is a sequence or a callable."""
    if callable(swapouts):
        read_swapouts = swapouts
    else:
        readings = iter(list(swapouts) + [swapouts[-1]] * 1000)
        read_swapouts = readings.__next__
    output = io.StringIO()
    with (
        contextlib.redirect_stdout(output),
        contextlib.redirect_stderr(io.StringIO()),
        mock.patch.object(probe, "read_vm_stat", lambda: (PAGE_BYTES, read_swapouts())),
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


AGENT_STATUS = snapshot(
    cache__resource_suspensions=0,
    cache__resource_replay_tokens=0,
    memory_governor__denied_reservations=0,
    memory_governor__denied_host_reservations=0,
    memory_governor__denied_engine_reservations=0,
)


def agent_arguments(stub, turns, *extra):
    return [
        "--base-url",
        stub.base_url,
        "--agent-turns",
        str(turns),
        "--tool-tokens",
        "20",
        "--warm-tokens",
        "80",
        "--max-tokens",
        "16",
        *extra,
    ]


class AgentModeTests(unittest.TestCase):
    def test_main_agentTurns_resendsPromptReplyAndToolWordsEachTurn(self):
        with StubServer(AGENT_STATUS, cached_tokens=[0, 90, 150]) as stub:
            code, _output = run_main(agent_arguments(stub, 3, "--json"))

        self.assertEqual(code, 0)
        conversations = [body["messages"] for body in stub.chat_bodies]
        self.assertEqual([len(messages) for messages in conversations], [1, 3, 5])
        for previous, following in zip(conversations, conversations[1:]):
            self.assertEqual(following[: len(previous)], previous)
            self.assertEqual(
                following[len(previous)],
                {"role": "assistant", "content": REPLY_CONTENT},
            )
            tool_words, _instruction = following[-1]["content"].split("\n")
            self.assertEqual(len(tool_words.split()), 20)

    def test_main_agentTurnAfterFirstReusesNothing_exitsColdTurnCode(self):
        with StubServer(AGENT_STATUS, cached_tokens=[0, 90, 0]) as stub:
            code, _output = run_main(agent_arguments(stub, 3, "--json"))

        self.assertEqual(code, probe.AGENT_COLD_TURN_EXIT_CODE)

    def test_main_agentTurns_reportsPerTurnFields(self):
        with StubServer(AGENT_STATUS, cached_tokens=[0, 90], reply_delay=0.1) as stub:
            code, output = run_main(
                agent_arguments(stub, 2, "--sample-seconds", "0.01", "--json")
            )

        self.assertEqual(code, 0)
        phases = json.loads(output)["phases"]
        self.assertEqual([phase["phase"] for phase in phases], ["turn0", "turn1"])
        turn = phases[1]
        self.assertEqual(turn["lanes"][0]["cached_tokens"], 90)
        self.assertEqual(turn["lanes"][0]["ttft_ms"], TTFT_MS)
        for key in (
            "suspensions",
            "replay_tokens",
            "denied_host_reservations",
            "denied_engine_reservations",
        ):
            self.assertEqual(turn[key], 0, key)
        self.assertGreater(turn["os_warn_s"], 0)

    def test_formatReport_agentTurns_rendersTtftAndOsWarnRows(self):
        with StubServer(AGENT_STATUS, cached_tokens=[0, 90]) as stub:
            _code, output = run_main(agent_arguments(stub, 2))

        self.assertIn("ttft ms", output)
        self.assertIn("os_warn_s", output)
        self.assertIn("turn1", output)

    def test_main_agentReplyWithoutContent_stopsAfterThatTurn(self):
        with StubServer(AGENT_STATUS, reply_content=None) as stub:
            code, output = run_main(agent_arguments(stub, 3, "--json"))

        self.assertEqual(code, 0)
        self.assertEqual(len(stub.chat_bodies), 1)
        self.assertIn(
            "no message content", json.loads(output)["phases"][0]["lanes"][0]["error"]
        )

    def test_main_agentSwapGrowthPastCap_stopsWithExitCode3(self):
        grown = 2 * 2**20 // PAGE_BYTES
        with StubServer(AGENT_STATUS, reply_delay=30.0) as stub:
            code, output = run_main(
                agent_arguments(stub, 3, "--swap-guard-mb", "1"),
                swapouts=(0, grown),
            )

        self.assertEqual(code, probe.SWAP_ABORT_EXIT_CODE)
        self.assertIn("swap guard tripped", output)
        self.assertLessEqual(len(stub.chat_bodies), 1)

    def test_parseArgs_lanesNotMatchingNonces_isRefused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(
                ["--base-url", "http://x", "--lanes", "3", "--nonces", "a"]
            )

    def test_parseArgs_noncesWithoutLanes_runsOneLanePerNonce(self):
        args = probe.parse_args(["--base-url", "http://x", "--nonces", "a", "b", "c"])

        self.assertEqual(args.lanes, 3)
        self.assertFalse(args.warm)

    def test_parseArgs_agentTurnsWithNonces_isRefused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(
                ["--base-url", "http://x", "--agent-turns", "2", "--nonces", "aaaa"]
            )


class BallastSizingTests(unittest.TestCase):
    def governor_sample(self, headroom_bytes, reserve_bytes):
        return snapshot(
            memory_governor__host_headroom_bytes=headroom_bytes,
            memory_governor__host_reserve_bytes=reserve_bytes,
        )

    def test_hostWarningMarginBytes_isAQuarterOfTheReserveAboveTheFloor(self):
        margins = [
            probe.host_warning_margin_bytes(reserve)
            for reserve in (2 * GIB, 512 * MIB, 0)
        ]

        self.assertEqual(margins, [512 * MIB, 256 * MIB, 256 * MIB])

    def test_autoBallast_isHeadroomLessWarningMarginPlusOvershoot(self):
        sample = self.governor_sample(10 * GIB, 2 * GIB)

        record = probe.auto_ballast(sample)

        self.assertEqual(
            record,
            {
                "mb": 10 * 1024 - 512 + 256,
                "host_headroom_bytes": 10 * GIB,
                "host_reserve_bytes": 2 * GIB,
                "warning_margin_bytes": 512 * MIB,
            },
        )

    def test_autoBallast_headroomBelowTheMargin_floorsAtZero(self):
        sample = self.governor_sample(100 * MIB, 2 * GIB)

        self.assertEqual(probe.auto_ballast(sample)["mb"], 0)

    def test_autoBallast_serverWithoutHostReadings_raises(self):
        with self.assertRaises(probe.ProbeError):
            probe.auto_ballast(snapshot(memory_governor__host_headroom_bytes=GIB))

    def test_parseArgs_ballastMb_acceptsAutoOrAPositiveSize(self):
        sizes = [
            probe.parse_args(
                ["--base-url", "http://x", "--ballast-mb", text]
            ).ballast_mb
            for text in ("auto", "512")
        ]

        self.assertEqual(sizes, ["auto", 512])

    def test_parseArgs_ballastMb_refusesZeroNegativeAndText(self):
        for text in ("0", "-3", "lots"):
            with (
                self.subTest(text=text),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                probe.parse_args(["--base-url", "http://x", "--ballast-mb", text])


class BallastChildTests(unittest.TestCase):
    def test_fillIncompressible_writesBytesTheCompressorCannotShrink(self):
        region = bytearray(64 * 1024)

        with mock.patch.object(probe, "BALLAST_CHUNK_BYTES", 4096):
            probe.fill_incompressible(region)

        self.assertGreater(len(zlib.compress(bytes(region))), len(region))

    def test_holdBallast_reportsReadyThenEndsWhenThePipeCloses(self):
        control, child_end = multiprocessing.Pipe()
        child = threading.Thread(target=probe.hold_ballast, args=(4096, child_end))
        child.start()

        ready = control.recv()
        control.close()
        child.join(5.0)

        self.assertEqual(ready, probe.BALLAST_READY)
        self.assertFalse(child.is_alive())


def fake_ballast_child(stub):
    """Return a BallastChild stand-in that logs starts and closes, and its log.

    Each event records how many chat requests the stub had received, which
    places it at a phase edge. Nothing is allocated.
    """
    events = []

    class FakeBallastChild:
        def __init__(self, byte_count):
            events.append(("start", byte_count, len(stub.chat_bodies)))

        def wait_ready(self, _stop):
            return True

        def close(self):
            events.append(("close", len(stub.chat_bodies)))

    return FakeBallastChild, events


def headroom_falling_per_request(chat_count):
    """Status whose host headroom drops 1 GiB per chat request received."""
    return snapshot(
        memory_governor__host_headroom_bytes=(10 - chat_count) * GIB,
        memory_governor__host_reserve_bytes=GIB,
    )


BURST_ARGUMENTS = ("--lanes", "2", "--warm-tokens", "80", "--extend-tokens", "20")


class BallastMainTests(unittest.TestCase):
    def run_with_fake_child(self, stub, arguments, swapouts=(0,)):
        child, events = fake_ballast_child(stub)
        with mock.patch.object(probe, "BallastChild", child):
            code, output = run_main(arguments, swapouts)
        return code, output, events

    def test_main_ballast_heldFromBurstStartUntilBurstEnds(self):
        with StubServer(snapshot()) as stub:
            code, _output, events = self.run_with_fake_child(
                stub,
                ["--base-url", stub.base_url, *BURST_ARGUMENTS, "--ballast-mb", "64"],
            )

        self.assertEqual(code, 0)
        self.assertEqual(events, [("start", 64 * MIB, 2), ("close", 4)])

    def test_main_agentTurns_ballastHeldFromSecondTurnUntilLastTurnEnds(self):
        with StubServer(AGENT_STATUS, cached_tokens=[0, 90, 150]) as stub:
            code, _output, events = self.run_with_fake_child(
                stub, agent_arguments(stub, 3, "--ballast-mb", "64")
            )

        self.assertEqual(code, 0)
        self.assertEqual(events, [("start", 64 * MIB, 1), ("close", 3)])

    def test_main_swapGuardTripsDuringBurst_closesTheBallastChild(self):
        grown = 2 * MIB // PAGE_BYTES
        with StubServer(snapshot(), reply_delay=30.0) as stub:
            code, _output, events = self.run_with_fake_child(
                stub,
                [
                    "--base-url",
                    stub.base_url,
                    "--nonces",
                    "aaaa",
                    "--ballast-mb",
                    "64",
                    "--swap-guard-mb",
                    "1",
                ],
                swapouts=lambda: grown if stub.chat_bodies else 0,
            )

        self.assertEqual(code, probe.SWAP_ABORT_EXIT_CODE)
        self.assertEqual([event[0] for event in events], ["start", "close"])

    def test_main_ballastAuto_sizedFromTheStatusAtBurstStart(self):
        with StubServer(headroom_falling_per_request) as stub:
            _code, output, events = self.run_with_fake_child(
                stub,
                [
                    "--base-url",
                    stub.base_url,
                    *BURST_ARGUMENTS,
                    "--ballast-mb",
                    "auto",
                    "--json",
                ],
            )

        self.assertEqual(events[0], ("start", 8 * GIB, 2))
        self.assertEqual(
            json.loads(output)["ballast"],
            {
                "mb": 8 * 1024,
                "host_headroom_bytes": 8 * GIB,
                "host_reserve_bytes": GIB,
                "warning_margin_bytes": 256 * MIB,
                "requested": "auto",
                "phase": "burst",
                "held": True,
            },
        )

    def test_main_agentTurnsBallastAuto_sizedAtTheSecondTurnsStart(self):
        with StubServer(headroom_falling_per_request) as stub:
            _code, output, events = self.run_with_fake_child(
                stub, agent_arguments(stub, 3, "--ballast-mb", "auto", "--json")
            )

        self.assertEqual(events[0], ("start", 9 * GIB, 1))
        self.assertEqual(json.loads(output)["ballast"]["phase"], "turn1")

    def test_formatReport_ballast_rendersSizeAndHeadroom(self):
        with StubServer(headroom_falling_per_request) as stub:
            _code, output, _events = self.run_with_fake_child(
                stub,
                [
                    "--base-url",
                    stub.base_url,
                    "--nonces",
                    "aaaa",
                    "--ballast-mb",
                    "auto",
                ],
            )

        self.assertRegex(output, r"ballast_mb\s+10240")
        self.assertRegex(output, rf"ballast_host_headroom_bytes\s+{10 * GIB}")

    def test_main_withoutBallast_reportsNoneAndStartsNoChild(self):
        with StubServer(snapshot()) as stub:
            _code, output, events = self.run_with_fake_child(
                stub, ["--base-url", stub.base_url, "--nonces", "aaaa", "--json"]
            )

        self.assertIsNone(json.loads(output)["ballast"])
        self.assertEqual(events, [])


class RequireFlagTests(unittest.TestCase):
    def sample(self, server, suspensions=None, waiting_memory=None):
        return {
            "server_pressure": server,
            "resource_suspensions": suspensions,
            "waiting_memory": waiting_memory,
        }

    def test_hasWarningSuspension_riseBetweenTwoWarningSamples_passes(self):
        samples = [
            self.sample("warning", 3),
            {"t": 1, "error": "timed out"},
            self.sample("warning", 4),
        ]

        self.assertTrue(probe.has_warning_suspension(samples))

    def test_hasWarningSuspension_riseLeavingNormal_fails(self):
        samples = [
            self.sample("normal", 3),
            self.sample("warning", 4),
            self.sample("warning", 4),
            self.sample("normal", 5),
        ]

        self.assertFalse(probe.has_warning_suspension(samples))

    def test_hasWaitingMemory_aSampleWaiting_passes(self):
        samples = [self.sample("normal", waiting_memory=0), self.sample("warning")]
        samples.append(self.sample("warning", waiting_memory=1))

        self.assertTrue(probe.has_waiting_memory(samples))

    def test_hasWaitingMemory_noSampleWaiting_fails(self):
        samples = [self.sample("normal", waiting_memory=0), self.sample("warning")]

        self.assertFalse(probe.has_waiting_memory(samples))

    def test_exitCode_swapGuardTrip_winsOverTheRequireFlags(self):
        args = argparse.Namespace(
            agent_turns=None,
            require_warning_suspension=True,
            require_waiting_memory=True,
        )
        document = {"swap_aborted": True, "phases": []}

        self.assertEqual(
            probe.exit_code(args, document, []), probe.SWAP_ABORT_EXIT_CODE
        )

    def run_required(self, status, flag):
        with StubServer(status, reply_delay=0.1) as stub:
            code, _output = run_main(
                [
                    "--base-url",
                    stub.base_url,
                    "--nonces",
                    "aaaa",
                    "--sample-seconds",
                    "0.01",
                    flag,
                ]
            )
        return code

    def test_main_requireWarningSuspension_suspensionUnderWarning_exitsZero(self):
        calls = itertools.count()

        def rising(_chat_count):
            return snapshot("warning", cache__resource_suspensions=next(calls))

        code = self.run_required(rising, "--require-warning-suspension")

        self.assertEqual(code, 0)

    def test_main_requireWarningSuspension_noSuspension_exitsFive(self):
        status = snapshot("warning", cache__resource_suspensions=2)

        code = self.run_required(status, "--require-warning-suspension")

        self.assertEqual(code, probe.WARNING_SUSPENSION_MISSING_EXIT_CODE)

    def test_main_requireWaitingMemory_requestWaiting_exitsZero(self):
        status = snapshot(admission__waiting_memory=1)

        code = self.run_required(status, "--require-waiting-memory")

        self.assertEqual(code, 0)

    def test_main_requireWaitingMemory_nothingWaiting_exitsSix(self):
        status = snapshot(admission__waiting_memory=0)

        code = self.run_required(status, "--require-waiting-memory")

        self.assertEqual(code, probe.WAITING_MEMORY_MISSING_EXIT_CODE)


class ReportTests(unittest.TestCase):
    def test_formatReport_rendersResumeDeltasPerPhase(self):
        status = snapshot(
            cache__resource_resume_restored_tokens=4096,
            cache__resource_resume_state_evictions=0,
        )
        with StubServer(status) as stub:
            _code, output = run_main(["--base-url", stub.base_url, "--nonces", "a"])

        self.assertRegex(output, r"resume_restored_tokens\s+0")
        self.assertRegex(output, r"resume_state_evictions\s+0")


if __name__ == "__main__":
    unittest.main()
