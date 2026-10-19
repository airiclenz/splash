"""Probe a running Splash server's decode latency beside a contended prefill.

Run as ``python -m dev.benchmarks.contended_probe --base-url http://127.0.0.1:8000``.
Streams one long decode; once its first token arrives, streams a cold prompt of
about ``--prompt-tokens`` tokens beside it. The overlap window runs from sending
the cold request to its first token, that is its contended prefill. Reports the
decode's client-side inter-chunk gap p50/p95/max inside the window, the cold
request's TTFT, and its native prefill tok/s and average prefill chunk rows
diffed from ``GET /status`` taken at the window's edges. Never starts a server;
the counters are server-wide, so run it against an otherwise idle server, once
per ``--contended-prefill-rows`` floor being compared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from dev.benchmarks.http_regression import status_delta
from dev.benchmarks.live_probe import (
    API_KEY_ENVIRONMENT,
    DEFAULT_TIMEOUT_SECONDS,
    MINIMUM_PROMPT_TOKENS,
    PREFILL_BATCHES,
    PREFILL_ROWS,
    PREFILL_TOKENS,
    PREFILL_WALL,
    ProbeClient,
    ProbeError,
    cold_prompt,
    format_value,
    fresh_status,
    rate_per_second,
    ratio,
    refuses_thinking_mode,
    resolve_model,
)

DEFAULT_PROMPT_TOKENS = 16384
DEFAULT_DECODE_TOKENS = 4096
# A short decode prompt keeps the decode lane's own prefill out of the window.
DECODE_PROMPT_TOKENS = 64
# The cold lane is read only up to its first token, then closed.
COLD_MAX_TOKENS = 8
CHAT_PATH = "/v1/chat/completions"
SSE_DATA_PREFIX = b"data:"
SSE_DONE = "[DONE]"
# Delta fields that carry generated output; role and progress chunks do not.
TOKEN_DELTA_FIELDS = ("content", "reasoning_content", "tool_calls")
MILLISECONDS_PER_SECOND = 1000.0
MEDIAN = 0.5
PERCENTILE_95 = 0.95
WINDOW_PATHS = (PREFILL_TOKENS, PREFILL_WALL, PREFILL_ROWS, PREFILL_BATCHES)

# (record key, table label) in display order.
REPORT_ROWS = (
    ("decode_gap_count", "decode gaps in overlap"),
    ("decode_gap_p50_ms", "decode gap p50 ms"),
    ("decode_gap_p95_ms", "decode gap p95 ms"),
    ("decode_gap_max_ms", "decode gap max ms"),
    ("decode_covered_overlap", "decode covered overlap"),
    ("cold_ttft_ms", "cold TTFT ms"),
    ("cold_prefill_tokens", "cold prefill tokens"),
    ("cold_prefill_tokens_per_second", "cold prefill tok/s"),
    ("average_prefill_chunk_rows", "avg prefill chunk rows"),
)


# ------------------------------------------------------------------------------
def sse_events(lines):
    """Yield each JSON event of a chat-completions SSE stream.

    `lines` iterates raw byte lines. Comments (keepalives) and blank lines are
    skipped and iteration ends at ``[DONE]``. Raises ProbeError on an event that
    is not JSON and on the server's in-stream error event.
    """
    for line in lines:
        if not line.startswith(SSE_DATA_PREFIX):
            continue
        payload = line[len(SSE_DATA_PREFIX) :].strip().decode(errors="replace")
        if payload == SSE_DONE:
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProbeError(f"stream event is not JSON: {payload[:200]}") from error
        if isinstance(event, dict) and "error" in event:
            raise ProbeError(f"stream failed: {json.dumps(event['error'])[:500]}")
        yield event


# ------------------------------------------------------------------------------
def carries_token(event: dict) -> bool:
    """True when a stream chunk delivers generated output in any choice."""
    for choice in event.get("choices") or ():
        delta = choice.get("delta") or {}
        if any(delta.get(field) for field in TOKEN_DELTA_FIELDS):
            return True
    return False


# ------------------------------------------------------------------------------
def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return the `fraction` quantile of non-empty `sorted_values`.

    Interpolates linearly between the two nearest ranks.
    """
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


# ------------------------------------------------------------------------------
def gap_statistics(chunk_times: list[float], start: float, end: float) -> dict:
    """Return the inter-chunk gaps, in ms, that end inside (start, end].

    `chunk_times` are ascending monotonic seconds of the decode's token chunks.
    A gap counts when its later chunk arrives inside the window, so a gap the
    window's start interrupts is measured whole. Without gaps every statistic
    is None.
    """
    gaps = sorted(
        (later - earlier) * MILLISECONDS_PER_SECOND
        for earlier, later in zip(chunk_times, chunk_times[1:])
        if start < later <= end
    )
    if not gaps:
        return {
            "decode_gap_count": 0,
            "decode_gap_p50_ms": None,
            "decode_gap_p95_ms": None,
            "decode_gap_max_ms": None,
        }
    return {
        "decode_gap_count": len(gaps),
        "decode_gap_p50_ms": percentile(gaps, MEDIAN),
        "decode_gap_p95_ms": percentile(gaps, PERCENTILE_95),
        "decode_gap_max_ms": gaps[-1],
    }


# ------------------------------------------------------------------------------
def contended_record(decode_times, decode_ended_at, start, end, before, after):
    """Build the report from the decode's chunk times and the window's snapshots.

    The window runs from sending the cold request (`start`) to its first token
    (`end`), in monotonic seconds, so its length is the cold TTFT; `before` and `after` are the `/status`
    snapshots taken at its edges. `decode_ended_at` is when the decode stream
    finished on its own, or None when the probe closed it after the window.
    """
    delta = status_delta(before, after, WINDOW_PATHS)
    return {
        **gap_statistics(decode_times, start, end),
        "decode_covered_overlap": decode_ended_at is None or decode_ended_at >= end,
        "cold_ttft_ms": (end - start) * MILLISECONDS_PER_SECOND,
        "cold_prefill_tokens": delta[PREFILL_TOKENS],
        "cold_prefill_tokens_per_second": rate_per_second(
            delta[PREFILL_TOKENS], delta[PREFILL_WALL]
        ),
        "average_prefill_chunk_rows": ratio(
            delta[PREFILL_ROWS], delta[PREFILL_BATCHES]
        ),
    }


# ------------------------------------------------------------------------------
def open_stream(client: ProbeClient, body: dict, timeout: float):
    """POST a streaming chat request and return the open HTTP response.

    Raises ProbeError, with the HTTP status when the server refused it.
    """
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    prepared = urllib.request.Request(
        client.base_url + CHAT_PATH,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        return urllib.request.urlopen(prepared, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise ProbeError(
            f"POST {CHAT_PATH}: HTTP {error.code}: {detail}", error.code
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise ProbeError(f"POST {CHAT_PATH}: {error}") from error


# ------------------------------------------------------------------------------
def stream_body(model: str, prompt: str, max_tokens: int) -> dict:
    """Return a greedy streaming chat body that asks for no thinking."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": True,
    }


# ------------------------------------------------------------------------------
def open_decode(client: ProbeClient, body: dict, timeout: float):
    """Open the decode stream, dropping `reasoning_effort` if the template refuses it.

    Mutates `body` on that fallback so the cold request inherits the same mode.
    """
    try:
        return open_stream(client, body, timeout)
    except ProbeError as error:
        # A template that cannot turn thinking off refuses before any native
        # work; measure with its default thinking mode instead.
        if not refuses_thinking_mode(error):
            raise
    del body["reasoning_effort"]
    print(
        "contended_probe: template cannot disable thinking; "
        "probing with its default reasoning effort",
        file=sys.stderr,
    )
    return open_stream(client, body, timeout)


class DecodeLane:
    """Read the decode stream on a thread, timing each token chunk."""

    # --------------------------------------------------------------------------
    def __init__(self, response):
        """Take ownership of the open decode `response`; it closes on exit."""
        self.response = response
        self.chunk_times: list[float] = []
        self.ended_at: float | None = None
        self.error: ProbeError | None = None
        self.streaming = threading.Event()
        self.stop = threading.Event()

    # --------------------------------------------------------------------------
    def run(self) -> None:
        """Record token chunk times until the stream ends or `stop` is set.

        `ended_at` is set when the stream ends or fails on its own, never when
        `stop` closes it; a failure is kept in `error`. `streaming` is set at the first token chunk, and also when the stream
        ends without one, so a waiter never blocks on a dead lane.
        """
        try:
            with self.response:
                for event in sse_events(self.response):
                    if carries_token(event):
                        self.chunk_times.append(time.monotonic())
                        self.streaming.set()
                    if self.stop.is_set():
                        return
                self.ended_at = time.monotonic()
        except ProbeError as error:
            self.error = error
            self.ended_at = time.monotonic()
        except OSError as error:
            self.error = ProbeError(f"decode stream: {error}")
            self.ended_at = time.monotonic()
        finally:
            self.streaming.set()


# ------------------------------------------------------------------------------
def first_token_time(response) -> float:
    """Return when the stream's first token chunk arrived, then close it."""
    with response:
        try:
            for event in sse_events(response):
                if carries_token(event):
                    return time.monotonic()
        except OSError as error:
            raise ProbeError(f"cold stream: {error}") from error
    raise ProbeError("cold request ended without a token")


# ------------------------------------------------------------------------------
def cold_body(decode_body: dict, prompt_tokens: int) -> dict:
    """Return the cold request: the decode's body and mode with an uncached prompt."""
    prompt = cold_prompt(prompt_tokens, COLD_MAX_TOKENS, uuid.uuid4().hex)
    return dict(
        decode_body,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=COLD_MAX_TOKENS,
    )


# ------------------------------------------------------------------------------
def run_probe(client: ProbeClient, model: str, args) -> dict:
    """Run the decode and the contended cold prefill; return the report record."""
    decode_body = stream_body(
        model,
        cold_prompt(DECODE_PROMPT_TOKENS, args.decode_tokens, uuid.uuid4().hex),
        args.decode_tokens,
    )
    lane = DecodeLane(open_decode(client, decode_body, args.request_timeout))
    reader = threading.Thread(target=lane.run, daemon=True)
    reader.start()
    try:
        lane.streaming.wait(args.request_timeout)
        if not lane.chunk_times:
            raise lane.error or ProbeError("decode produced no token")
        cold = cold_body(decode_body, args.prompt_tokens)
        before = fresh_status(client)
        start = time.monotonic()
        end = first_token_time(open_stream(client, cold, args.request_timeout))
        after = fresh_status(client)
    finally:
        lane.stop.set()
        reader.join()
    if lane.error and lane.ended_at < end:
        raise lane.error
    return contended_record(lane.chunk_times, lane.ended_at, start, end, before, after)


# ------------------------------------------------------------------------------
def format_cell(value) -> str:
    """Render one table cell: booleans as yes/no, the rest as live_probe does."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format_value(value)


# ------------------------------------------------------------------------------
def format_table(record: dict) -> str:
    """Render the record as a two-column label/value table."""
    width = max(len(label) for _, label in REPORT_ROWS)
    return "\n".join(
        f"{label:<{width}}  {format_cell(record[key]):>12}"
        for key, label in REPORT_ROWS
    )


# ------------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="running server, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(API_KEY_ENVIRONMENT),
        help=f"Bearer key (default: ${API_KEY_ENVIRONMENT})",
    )
    parser.add_argument(
        "--model", help="model id (default: the first one /v1/models lists)"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_PROMPT_TOKENS,
        help="approximate cold prompt tokens",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        help="decode budget; must outlast the cold prefill",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    args = parser.parse_args(argv)
    if args.prompt_tokens < MINIMUM_PROMPT_TOKENS:
        parser.error(f"--prompt-tokens must be at least {MINIMUM_PROMPT_TOKENS}")
    if args.decode_tokens < 2 or args.request_timeout <= 0:
        parser.error(
            "--decode-tokens must be at least 2 and --request-timeout positive"
        )
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must start with http:// or https://")
    return args


# ------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Run the probe and print the report; return the process exit code."""
    args = parse_args(argv)
    client = ProbeClient(args.base_url, args.api_key)
    try:
        model = resolve_model(client, args.model)
        record = run_probe(client, model, args)
    except ProbeError as error:
        print(f"contended_probe: {error}", file=sys.stderr)
        return 1
    if not record["decode_covered_overlap"]:
        print(
            "contended_probe: the decode ended before the cold prefill did; "
            "raise --decode-tokens",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps({"model": model, **record}, indent=2))
    else:
        print(f"model: {model}")
        print(format_table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
