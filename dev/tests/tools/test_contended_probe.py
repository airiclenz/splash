import contextlib
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dev.benchmarks import contended_probe as probe

MODEL = "incoai/Qwen3.8-27B-Splash"
# The decode stub streams this many token chunks at most, one per interval.
DECODE_CHUNKS = 2000
DECODE_INTERVAL_SECONDS = 0.002
COLD_DELAY_SECONDS = 0.05


def snapshot(prefill_tokens=0, prefill_wall_ms=0, prefill_rows=0, prefill_batches=0):
    """Return a fresh `/status` document carrying the window's counters."""
    return {
        "transport": {"status_stale": False},
        "metrics": {
            "prefill_input_tokens": prefill_tokens,
            "prefill_wall_ms": prefill_wall_ms,
        },
        "scheduler": {
            "prefill_rows": prefill_rows,
            "prefill_batches": prefill_batches,
        },
    }


# The window: 16384 tokens prefilled in 8000 ms over 128 chunks of 128 rows.
WINDOW_BEFORE = snapshot(64, 20, 64, 1)
WINDOW_AFTER = snapshot(64 + 16384, 20 + 8000, 64 + 16384, 1 + 128)


def token_event(field="content", text="1"):
    return {"choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}]}


def sse_line(payload) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return b"data: " + data.encode() + b"\n"


class GapStatisticsTests(unittest.TestCase):
    def test_gapStatistics_chunksAcrossWindow_countsOnlyGapsEndingInside(self):
        chunk_times = [0.0, 1.0, 1.1, 1.3, 1.6, 2.0, 5.0]

        statistics = probe.gap_statistics(chunk_times, 1.05, 2.0)

        self.assertEqual(statistics["decode_gap_count"], 4)
        self.assertAlmostEqual(statistics["decode_gap_p50_ms"], 250.0)
        self.assertAlmostEqual(statistics["decode_gap_p95_ms"], 385.0)
        self.assertAlmostEqual(statistics["decode_gap_max_ms"], 400.0)

    def test_gapStatistics_gapInterruptedByWindowStart_isMeasuredWhole(self):
        statistics = probe.gap_statistics([0.0, 1.0], 0.5, 2.0)

        self.assertEqual(statistics["decode_gap_count"], 1)
        self.assertAlmostEqual(statistics["decode_gap_max_ms"], 1000.0)

    def test_gapStatistics_noChunkInsideWindow_reportsUnavailable(self):
        statistics = probe.gap_statistics([0.0, 1.0], 1.0, 2.0)

        self.assertEqual(
            statistics,
            {
                "decode_gap_count": 0,
                "decode_gap_p50_ms": None,
                "decode_gap_p95_ms": None,
                "decode_gap_max_ms": None,
            },
        )

    def test_percentile_singleValue_returnsIt(self):
        self.assertEqual(probe.percentile([7.0], probe.PERCENTILE_95), 7.0)


class SseEventsTests(unittest.TestCase):
    def test_sseEvents_keepalivesAndBlanks_yieldsEventsUntilDone(self):
        lines = [
            b": splash-keepalive\n",
            b"\n",
            sse_line(token_event()),
            b"\n",
            sse_line("[DONE]"),
            sse_line(token_event(text="after done")),
        ]

        events = list(probe.sse_events(lines))

        self.assertEqual(events, [token_event()])

    def test_sseEvents_errorEvent_raises(self):
        lines = [sse_line({"error": {"message": "native runtime failed"}})]

        with self.assertRaisesRegex(probe.ProbeError, "native runtime failed"):
            list(probe.sse_events(lines))

    def test_sseEvents_invalidJson_raises(self):
        with self.assertRaisesRegex(probe.ProbeError, "not JSON"):
            list(probe.sse_events([b"data: {truncated\n"]))

    def test_carriesToken_roleAndProgressChunks_doNotCount(self):
        role = token_event("role", "assistant")
        role["choices"][0]["delta"]["content"] = ""
        progress = {"choices": [{"delta": {}}], "prompt_progress": {"processed": 1}}
        usage = {"choices": [], "usage": {"completion_tokens": 8}}

        self.assertFalse(any(map(probe.carries_token, (role, progress, usage))))

    def test_carriesToken_contentOrReasoning_counts(self):
        self.assertTrue(probe.carries_token(token_event("content")))
        self.assertTrue(probe.carries_token(token_event("reasoning_content")))


class ContendedRecordTests(unittest.TestCase):
    def test_contendedRecord_cannedWindow_reportsGapsTtftAndStatusRates(self):
        chunk_times = [9.9, 10.0, 10.1, 10.4, 10.5, 18.0]

        record = probe.contended_record(
            chunk_times, None, 10.05, 18.05, WINDOW_BEFORE, WINDOW_AFTER
        )

        self.assertEqual(record["decode_gap_count"], 4)
        self.assertAlmostEqual(record["decode_gap_max_ms"], 7500.0)
        self.assertTrue(record["decode_covered_overlap"])
        self.assertAlmostEqual(record["cold_ttft_ms"], 8000.0)
        self.assertEqual(record["cold_prefill_tokens"], 16384)
        self.assertAlmostEqual(record["cold_prefill_tokens_per_second"], 2048.0)
        self.assertAlmostEqual(record["average_prefill_chunk_rows"], 128.0)

    def test_contendedRecord_decodeEndedInsideWindow_flagsUncoveredOverlap(self):
        record = probe.contended_record(
            [10.0, 10.1], 10.2, 10.05, 18.05, WINDOW_BEFORE, WINDOW_AFTER
        )

        self.assertFalse(record["decode_covered_overlap"])

    def test_contendedRecord_noPrefillInWindow_reportsRatesUnavailable(self):
        record = probe.contended_record(
            [10.0, 10.1], None, 10.05, 18.05, WINDOW_BEFORE, WINDOW_BEFORE
        )

        self.assertIsNone(record["cold_prefill_tokens_per_second"])
        self.assertIsNone(record["average_prefill_chunk_rows"])

    def test_formatTable_record_rendersEveryRowWithUnavailableAndYesNo(self):
        record = probe.contended_record(
            [10.0], None, 10.05, 18.05, WINDOW_BEFORE, WINDOW_BEFORE
        )

        table = probe.format_table(record)

        self.assertEqual(len(table.splitlines()), len(probe.REPORT_ROWS))
        self.assertRegex(table, r"decode gap p50 ms\s+unavailable")
        self.assertRegex(table, r"decode covered overlap\s+yes")


class StreamingStubServer:
    """Stream a steady decode and a cold reply after a delay; serve `/status`."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.status_requests = 0
        self.chat_bodies = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    document = {"object": "list", "data": [{"id": MODEL}]}
                else:
                    index = min(stub.status_requests, len(stub.statuses) - 1)
                    stub.status_requests += 1
                    document = stub.statuses[index]
                raw = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                stub.chat_bodies.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.stream(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def stream(self, body):
                self.write(token_event("role", "assistant"))
                if body["max_completion_tokens"] == probe.COLD_MAX_TOKENS:
                    self.wfile.write(b": splash-keepalive\n\n")
                    time.sleep(COLD_DELAY_SECONDS)
                    self.write(token_event())
                else:
                    for _ in range(DECODE_CHUNKS):
                        self.write(token_event())
                        time.sleep(DECODE_INTERVAL_SECONDS)
                self.write("[DONE]")

            def write(self, payload):
                self.wfile.write(sse_line(payload) + b"\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
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


class ContendedProbeMainTests(unittest.TestCase):
    def test_main_streamingStub_reportsWindowFromStatusAndDecodeGaps(self):
        output = io.StringIO()
        with StreamingStubServer([WINDOW_BEFORE, WINDOW_AFTER]) as stub:
            with contextlib.redirect_stdout(output):
                code = probe.main(["--base-url", stub.base_url, "--json"])

        report = json.loads(output.getvalue())
        decode_body, cold_body = stub.chat_bodies
        self.assertEqual(code, 0)
        self.assertEqual(report["model"], MODEL)
        self.assertEqual(report["cold_prefill_tokens"], 16384)
        self.assertAlmostEqual(report["average_prefill_chunk_rows"], 128.0)
        self.assertGreater(report["decode_gap_count"], 0)
        self.assertTrue(report["decode_covered_overlap"])
        self.assertGreaterEqual(report["cold_ttft_ms"], COLD_DELAY_SECONDS * 1000)
        self.assertTrue(decode_body["stream"] and cold_body["stream"])
        self.assertEqual(decode_body["max_completion_tokens"], 4096)
        self.assertEqual(cold_body["max_completion_tokens"], probe.COLD_MAX_TOKENS)

    def test_parseArgs_tooFewDecodeTokens_exits(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(["--base-url", "http://x", "--decode-tokens", "1"])


if __name__ == "__main__":
    unittest.main()
