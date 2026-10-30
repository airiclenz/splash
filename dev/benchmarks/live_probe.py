"""Probe a running Splash server's prefill and decode rates.

Run as ``python -m dev.benchmarks.live_probe --base-url http://127.0.0.1:8000``.
Sends one cold prompt and then its exact repeat (or, with ``--prompt-file``,
that file's text alone as one request), and reports per request the native
rates and cache/pressure counters diffed from ``GET /status`` taken just before
and just after it, plus the reply's finish reason and content. Never starts a
server; the counters are server-wide, so run it against an otherwise idle
server.

With ``--stop-sequence`` it instead streams one counting reply that the stop
sequence cuts short and reports the gap from the last content delta to the
finish chunk, and the request's decode wall per cycle from ``GET /requests``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

from dev.benchmarks.http_regression import status_delta

DEFAULT_PROMPT_TOKENS = 2048
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT_SECONDS = 1800.0
STATUS_TIMEOUT_SECONDS = 30.0
MINIMUM_PROMPT_TOKENS = 16
# Tokens left for the nonce, the closing instruction and the chat template.
PROMPT_FRAME_TOKENS = 16
# Substring of the server's 400 for a thinking mode the chat template lacks.
THINKING_MODE_REFUSAL = "does not support the requested thinking mode"
STALE_SNAPSHOT_RETRIES = 3
STALE_RETRY_DELAY_SECONDS = 0.15
API_KEY_ENVIRONMENT = "SPLASH_API_KEY"
UNAVAILABLE = "unavailable"
JSON_ACCEPT = "application/json"
SSE_ACCEPT = "text/event-stream"
SSE_DATA_PREFIX = b"data:"
SSE_DONE = "[DONE]"
CHAT_PATH = "/v1/chat/completions"
REQUESTS_PATH = "/requests"
# Chat completion ids prefix the bare public id `GET /requests` reports.
CHAT_ID_PREFIX = "chatcmpl-"
MILLISECONDS_PER_SECOND = 1000.0

# The `--stop-sequence` probe: a greedy counting reply its stop cuts short.
STOP_COUNT_TO = 60
STOP_PROMPT = f"Count from 1 to {STOP_COUNT_TO}, separated by commas."
# The reply the prompt asks for; a stop sequence must occur in it.
STOP_REFERENCE = ", ".join(str(number) for number in range(1, STOP_COUNT_TO + 1))
DEFAULT_STOP_SEQUENCE = ", 20,"
STOP_FINISH_REASON = "stop"
STOP_LABEL = "stop"
# The request's `GET /requests` record lands after native completion, which
# can follow the client's done, so it is polled for this long.
REQUEST_RECORD_WAIT_SECONDS = 5.0
REQUEST_RECORD_POLL_SECONDS = 0.1
REQUEST_RECORD_POLLS = round(REQUEST_RECORD_WAIT_SECONDS / REQUEST_RECORD_POLL_SECONDS)

# Common English words that most tokenizers encode as one token after a space,
# so the word count approximates the requested prompt token count.
FILLER_WORDS = (
    "river stone garden window bridge market letter silver morning paper "
    "forest engine harbor yellow candle winter number planet rabbit castle "
    "orange pocket summer valley ticket button mirror island basket doctor "
    "copper meadow signal travel corner shadow pillow hammer jacket marble "
    "lemon rocket tunnel violet wagon anchor bottle carpet desert feather "
    "ginger helmet jungle kettle ladder magnet needle oyster pepper quilt "
    "saddle timber velvet walnut zipper album bakery cactus dragon falcon"
).split()

# The cumulative `/status` counters diffed around each request.
PREFILL_TOKENS = ("metrics", "prefill_input_tokens")
PREFILL_WALL = ("metrics", "prefill_wall_ms")
DECODE_TOKENS = ("metrics", "decode_output_tokens")
DECODE_WALL = ("metrics", "decode_wall_ms")
DRAFTED = ("metrics", "drafted_tokens")
ACCEPTED = ("metrics", "accepted_draft_tokens")
PREFILL_ROWS = ("scheduler", "prefill_rows")
PREFILL_BATCHES = ("scheduler", "prefill_batches")
REUSED = ("cache", "reused_tokens")
SUSPENSIONS = ("cache", "resource_suspensions")
REPLAY = ("cache", "resource_replay_tokens")
DENIED = ("memory_governor", "denied_reservations")
PROBE_PATHS = (
    PREFILL_TOKENS,
    PREFILL_WALL,
    DECODE_TOKENS,
    DECODE_WALL,
    DRAFTED,
    ACCEPTED,
    PREFILL_ROWS,
    PREFILL_BATCHES,
    REUSED,
    SUSPENSIONS,
    REPLAY,
    DENIED,
)
# The governor's host page sample, read from the after-snapshot rather than
# diffed: (record key, `/status` path). An absent key reads as None.
HOST_READINGS = (
    ("host_available_bytes", ("memory_governor", "host_available_bytes")),
    ("host_inactive_bytes", ("memory_governor", "host_inactive_bytes")),
    ("host_wired_bytes", ("memory_governor", "host_wired_bytes")),
    ("host_compressor_bytes", ("memory_governor", "host_compressor_bytes")),
)

# (record key, table label) in display order.
REPORT_ROWS = (
    ("prompt_tokens", "prompt tokens"),
    ("completion_tokens", "completion tokens"),
    ("prefill_tokens_per_second", "prefill tok/s"),
    ("decode_tokens_per_second", "decode tok/s"),
    ("draft_acceptance", "draft acceptance"),
    ("average_prefill_chunk_rows", "avg prefill chunk rows"),
    ("cache_matched_tokens", "cache matched tokens"),
    ("resource_suspensions", "resource suspensions"),
    ("replay_tokens", "replay tokens"),
    ("denied_reservations", "denied reservations"),
    ("host_available_bytes", "host available bytes"),
    ("host_inactive_bytes", "host inactive bytes"),
    ("host_wired_bytes", "host wired bytes"),
    ("host_compressor_bytes", "host compressor bytes"),
    ("http_wall_ms", "HTTP wall ms"),
)
# The rows a `--stop-sequence` run adds below REPORT_ROWS.
STOP_ROWS = (
    ("stop_to_done_ms", "stop to done ms"),
    ("cycle_ms", "cycle ms"),
)


class ProbeError(RuntimeError):
    """The server refused a request or never produced a fresh `/status`.

    `status` is the HTTP status of a refused request, None otherwise.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ProbeClient:
    """Minimal JSON-over-HTTP client for one Splash server."""

    # --------------------------------------------------------------------------
    def __init__(self, base_url: str, api_key: str | None):
        """Bind to `base_url` (scheme://host:port), sending `api_key` as Bearer."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # --------------------------------------------------------------------------
    def request(self, method: str, path: str, body=None, timeout=60.0) -> dict:
        """Send one request and return its decoded JSON body.

        Raises ProbeError on a connection failure, a non-2xx status or a body
        that is not JSON.
        """
        response = self._open(method, path, body, JSON_ACCEPT, timeout)
        try:
            with response:
                raw = response.read()
        except OSError as error:
            raise ProbeError(f"{method} {path}: {error}") from error
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProbeError(f"{method} {path} returned invalid JSON") from error

    # --------------------------------------------------------------------------
    def stream(self, path: str, body: dict, timeout: float):
        """POST `body` to `path` and yield `(arrival, event)` per SSE event.

        `arrival` is `time.monotonic()` just after the event's line was read.
        Comment and blank lines are skipped and iteration ends at ``[DONE]``.
        Raises ProbeError as `request` does, and on an event that is not JSON
        or is the server's in-stream error event.
        """
        response = self._open("POST", path, body, SSE_ACCEPT, timeout)
        with response:
            try:
                for line in response:
                    if not line.startswith(SSE_DATA_PREFIX):
                        continue
                    payload = (
                        line[len(SSE_DATA_PREFIX) :].strip().decode(errors="replace")
                    )
                    if payload == SSE_DONE:
                        return
                    yield time.monotonic(), sse_event(payload)
            except OSError as error:
                raise ProbeError(f"POST {path} stream: {error}") from error

    # --------------------------------------------------------------------------
    def _open(self, method: str, path: str, body, accept: str, timeout):
        """Send one request and return its open response for the caller to read.

        Raises ProbeError on a connection failure or a non-2xx status, carrying
        the HTTP status of a refused request.
        """
        headers = {"Accept": accept}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        prepared = urllib.request.Request(
            self.base_url + path,
            data=payload,
            headers=headers,
            method=method,
        )
        try:
            return urllib.request.urlopen(prepared, timeout=timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise ProbeError(
                f"{method} {path}: HTTP {error.code}: {detail}", error.code
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise ProbeError(f"{method} {path}: {error}") from error


# ------------------------------------------------------------------------------
def sse_event(payload: str) -> dict:
    """Decode one SSE `data:` payload; ProbeError on bad JSON or an error event."""
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ProbeError(f"stream event is not JSON: {payload[:200]}") from error
    if isinstance(event, dict) and "error" in event:
        raise ProbeError(f"stream failed: {json.dumps(event['error'])[:500]}")
    return event


# ------------------------------------------------------------------------------
def fresh_status(client: ProbeClient) -> dict:
    """Return a `/status` snapshot the server marked fresh.

    A stale snapshot (`transport.status_stale`) carries counters from an older
    native refresh and would corrupt the diff, so it is retried at most
    STALE_SNAPSHOT_RETRIES times with a short sleep; then ProbeError.
    """
    for attempt in range(STALE_SNAPSHOT_RETRIES + 1):
        if attempt:
            time.sleep(STALE_RETRY_DELAY_SECONDS)
        snapshot = client.request("GET", "/status", timeout=STATUS_TIMEOUT_SECONDS)
        transport = snapshot.get("transport")
        if not (isinstance(transport, dict) and transport.get("status_stale")):
            return snapshot
    raise ProbeError(
        f"/status stayed stale after {STALE_SNAPSHOT_RETRIES} retries; "
        "the native runtime is not answering status requests"
    )


# ------------------------------------------------------------------------------
def ratio(numerator: float, denominator: float):
    """Return numerator / denominator, or None when the denominator is not positive."""
    return numerator / denominator if denominator > 0 else None


# ------------------------------------------------------------------------------
def rate_per_second(tokens: float, wall_ms: float):
    """Return tokens per second over `wall_ms`, or None for a zero wall time."""
    return ratio(tokens * 1000.0, wall_ms)


# ------------------------------------------------------------------------------
def status_reading(snapshot: dict, path):
    """Return the number at a nested `/status` key path, or None when absent.

    Unlike a diffed counter, a point reading an older server does not export
    must not pass for 0, so a missing key or non-numeric leaf reads as None.
    """
    value = snapshot
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


# ------------------------------------------------------------------------------
def probe_record(label: str, before: dict, after: dict, reply: dict, wall_ms):
    """Build one request's report row from its bracketing `/status` snapshots.

    Every rate and counter comes from the snapshot diff; only the token counts
    come from the reply's `usage`. A rate whose denominator delta is 0 is None.
    The host memory readings come from the after-snapshot alone; the finish
    reason and content come from the reply's first choice.
    """
    delta = status_delta(before, after, PROBE_PATHS)
    usage = reply.get("usage") or {}
    choices = reply.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return {
        "request": label,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_tokens_per_second": rate_per_second(
            delta[PREFILL_TOKENS], delta[PREFILL_WALL]
        ),
        "decode_tokens_per_second": rate_per_second(
            delta[DECODE_TOKENS], delta[DECODE_WALL]
        ),
        "draft_acceptance": ratio(delta[ACCEPTED], delta[DRAFTED]),
        "average_prefill_chunk_rows": ratio(
            delta[PREFILL_ROWS], delta[PREFILL_BATCHES]
        ),
        "cache_matched_tokens": delta[REUSED],
        "resource_suspensions": delta[SUSPENSIONS],
        "replay_tokens": delta[REPLAY],
        "denied_reservations": delta[DENIED],
        **{key: status_reading(after, path) for key, path in HOST_READINGS},
        "http_wall_ms": wall_ms,
        "finish_reason": choices[0].get("finish_reason"),
        "content": message.get("content"),
    }


# ------------------------------------------------------------------------------
def cold_prompt(prompt_tokens: int, max_tokens: int, nonce: str) -> str:
    """Return a prompt of about `prompt_tokens` tokens that no cache can match.

    The nonce leads so even the first block differs from every earlier prompt;
    the closing instruction asks for more output than `max_tokens` allows so
    decode runs to the cap.
    """
    generator = random.Random(nonce)
    words = generator.choices(
        FILLER_WORDS, k=max(1, prompt_tokens - PROMPT_FRAME_TOKENS)
    )
    instruction = f"Write the integers from 1 to {max_tokens * 2}, one per line."
    return f"{nonce}\n{' '.join(words)}\n{instruction}"


# ------------------------------------------------------------------------------
def resolve_model(client: ProbeClient, model: str | None) -> str:
    """Return `model`, or the first id `/v1/models` lists when it is None."""
    if model:
        return model
    listed = client.request("GET", "/v1/models").get("data") or []
    if not listed or not listed[0].get("id"):
        raise ProbeError("/v1/models lists no model; pass --model")
    return listed[0]["id"]


# ------------------------------------------------------------------------------
def run_probe(
    client: ProbeClient, model: str, prompt: str, max_tokens, timeout, repeat=True
):
    """Send `prompt` cold, then its exact repeat; return the report rows.

    With `repeat` False only the cold request is sent. Each `/status` snapshot
    is taken outside the timed window.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    records = []
    for label in ("cold", "repeat") if repeat else ("cold",):
        try:
            records.append(timed_request(client, label, body, timeout))
        except ProbeError as error:
            # A template that cannot turn thinking off refuses before any
            # native work; measure with its default thinking mode instead.
            if label != "cold" or not refuses_thinking_mode(error):
                raise
            drop_reasoning_effort(body)
            records.append(timed_request(client, label, body, timeout))
    return records


# ------------------------------------------------------------------------------
def timed_request(client: ProbeClient, label: str, body: dict, timeout) -> dict:
    """Send `body` once and return its report row, diffed from `/status`."""
    before = fresh_status(client)
    started = time.monotonic()
    reply = client.request("POST", "/v1/chat/completions", body, timeout=timeout)
    wall_ms = (time.monotonic() - started) * 1000.0
    after = fresh_status(client)
    return probe_record(label, before, after, reply, wall_ms)


# ------------------------------------------------------------------------------
def drop_reasoning_effort(body: dict) -> None:
    """Remove `reasoning_effort` from `body`, noting the fallback on stderr."""
    del body["reasoning_effort"]
    print(
        "live_probe: template cannot disable thinking; "
        "probing with its default reasoning effort",
        file=sys.stderr,
    )


# ------------------------------------------------------------------------------
def run_stop_probe(
    client: ProbeClient, model: str, stop_sequence: str, max_tokens, timeout
) -> dict:
    """Stream the counting prompt once with `stop_sequence`; return its record.

    The record carries every REPORT_ROWS and STOP_ROWS key: the token counts
    come from the stream's usage and the HTTP wall from the probe's clock; no
    `/status` diff applies, so the other REPORT_ROWS keys are None.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": STOP_PROMPT}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "stop": [stop_sequence],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        reading = StopStream.read(client, body, timeout)
    except ProbeError as error:
        if not refuses_thinking_mode(error):
            raise
        drop_reasoning_effort(body)
        reading = StopStream.read(client, body, timeout)
    record = dict.fromkeys(key for key, _label in REPORT_ROWS + STOP_ROWS)
    record.update(
        request=STOP_LABEL,
        prompt_tokens=reading.usage.get("prompt_tokens"),
        completion_tokens=reading.usage.get("completion_tokens"),
        http_wall_ms=reading.wall_ms,
        stop_to_done_ms=reading.stop_to_done_ms(),
        cycle_ms=request_cycle_ms(client, reading.response_id),
        finish_reason=reading.finish_reason,
        content="".join(reading.content),
    )
    return record


class StopStream:
    """The arrival times and fields of one streamed chat completion."""

    # --------------------------------------------------------------------------
    def __init__(self):
        """Start empty: no content, no finish, no usage yet."""
        self.response_id = None
        self.content = []
        self.last_content_at = None
        self.finish_reason = None
        self.finished_at = None
        self.usage = {}
        self.wall_ms = None

    # --------------------------------------------------------------------------
    @classmethod
    def read(cls, client: ProbeClient, body: dict, timeout) -> StopStream:
        """POST the streaming `body` and fold its events until ``[DONE]``."""
        reading = cls()
        started = time.monotonic()
        for arrival, event in client.stream(CHAT_PATH, body, timeout):
            reading.observe(arrival, event)
        reading.wall_ms = (time.monotonic() - started) * MILLISECONDS_PER_SECOND
        return reading

    # --------------------------------------------------------------------------
    def observe(self, arrival: float, event: dict) -> None:
        """Fold one chunk: its id, content delta, finish reason and usage."""
        self.response_id = event.get("id") or self.response_id
        self.usage = event.get("usage") or self.usage
        for choice in event.get("choices") or ():
            text = (choice.get("delta") or {}).get("content")
            if text:
                self.content.append(text)
                self.last_content_at = arrival
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
                self.finished_at = arrival

    # --------------------------------------------------------------------------
    def stop_to_done_ms(self):
        """Return the finish chunk's arrival minus the last content delta's, in ms.

        It includes decoding the stop text the server held back before it
        matched. None without both chunks.
        """
        if self.last_content_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.last_content_at) * MILLISECONDS_PER_SECOND


# ------------------------------------------------------------------------------
def request_cycle_ms(client: ProbeClient, response_id: str | None):
    """Return the request's decode wall per cycle from its `GET /requests` record.

    A cycle is a draft cycle, or a decode step when the request drafted none.
    A record still missing after the bounded wait gives None with a note on
    stderr; a record with no cycles gives None.
    """
    record = None
    if response_id:
        record = wait_for_request_record(
            client, response_id.removeprefix(CHAT_ID_PREFIX)
        )
    if record is None:
        print(
            f"live_probe: no GET {REQUESTS_PATH} record for {response_id} within "
            f"{REQUEST_RECORD_WAIT_SECONDS:g} s; cycle ms unavailable",
            file=sys.stderr,
        )
        return None
    engine = record.get("engine") or {}
    cycles = record.get("draft_cycles") or engine.get("decode_steps") or 0
    return ratio(engine.get("decode_wall_ms") or 0, cycles)


# ------------------------------------------------------------------------------
def wait_for_request_record(client: ProbeClient, public_id: str):
    """Poll `GET /requests` until it lists `public_id`; None once the wait ends.

    Polls REQUEST_RECORD_POLLS + 1 times, REQUEST_RECORD_POLL_SECONDS apart.
    """
    for attempt in range(REQUEST_RECORD_POLLS + 1):
        if attempt:
            time.sleep(REQUEST_RECORD_POLL_SECONDS)
        listed = client.request("GET", REQUESTS_PATH, timeout=STATUS_TIMEOUT_SECONDS)
        for record in reversed(listed.get("requests") or []):
            if record.get("id") == public_id:
                return record
    return None


# ------------------------------------------------------------------------------
def stop_failure(record: dict, stop_sequence: str, max_tokens: int) -> str | None:
    """Return why `record` is not a stop-sequence finish, or None when it is.

    `finish_reason` `stop` alone does not show it, since an EOS finish reports
    it too: the completion must also end before `max_tokens` with content that
    ends just before the stop text's place in the counting reply.
    """
    if record["finish_reason"] != STOP_FINISH_REASON:
        return (
            f"finish_reason is {record['finish_reason']!r}, not {STOP_FINISH_REASON!r}"
        )
    completion_tokens = record["completion_tokens"]
    if not isinstance(completion_tokens, int) or completion_tokens >= max_tokens:
        return f"usage.completion_tokens {completion_tokens} is not below {max_tokens}"
    expected = STOP_REFERENCE[: STOP_REFERENCE.index(stop_sequence)]
    if not (record["content"] or "").endswith(expected):
        return f"content does not end just before the stop text {stop_sequence!r}"
    return None


# ------------------------------------------------------------------------------
def refuses_thinking_mode(error: ProbeError) -> bool:
    """True when the server refused the requested thinking mode."""
    return error.status == 400 and THINKING_MODE_REFUSAL in str(error)


# ------------------------------------------------------------------------------
def format_value(value) -> str:
    """Render one table cell; None reads as unavailable."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(int(value))


# ------------------------------------------------------------------------------
def format_table(records: list[dict], rows=REPORT_ROWS) -> str:
    """Render the records as a fixed-width table, one column per request.

    `rows` holds the (record key, label) pairs shown, in order.
    """
    width = max(len(label) for _, label in rows)
    header = f"{'':<{width}}" + "".join(
        f"  {record['request']:>12}" for record in records
    )
    lines = [header]
    for key, label in rows:
        cells = "".join(f"  {format_value(record[key]):>12}" for record in records)
        lines.append(f"{label:<{width}}{cells}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
def format_replies(records: list[dict]) -> str:
    """Render each request's finish reason and reply content, one line each."""
    return "\n".join(
        f"{record['request']}: finish_reason {record['finish_reason']}, "
        f"content {record['content']!r}"
        for record in records
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
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument(
        "--prompt-file",
        help="send this file's text as the user message, once, with no repeat "
        "(--prompt-tokens is then ignored)",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--stop-sequence",
        nargs="?",
        const=DEFAULT_STOP_SEQUENCE,
        help="instead stream one counting reply with this stop sequence "
        f"(default {DEFAULT_STOP_SEQUENCE!r}) and report stop-to-done and cycle ms; "
        "exits non-zero unless the stop sequence ended it",
    )
    parser.add_argument("--json", action="store_true", help="print the records as JSON")
    args = parser.parse_args(argv)
    if args.stop_sequence is not None:
        if args.prompt_file:
            parser.error("--stop-sequence and --prompt-file cannot be combined")
        if STOP_REFERENCE.find(args.stop_sequence) <= 0:
            parser.error(
                "--stop-sequence must occur after the start of the counting reply "
                f"{STOP_REFERENCE[:40]!r}..."
            )
    if not args.prompt_file and args.prompt_tokens < MINIMUM_PROMPT_TOKENS:
        parser.error(f"--prompt-tokens must be at least {MINIMUM_PROMPT_TOKENS}")
    if args.max_tokens < 1 or args.request_timeout <= 0:
        parser.error("--max-tokens and --request-timeout must be positive")
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must start with http:// or https://")
    return args


# ------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Run the probe and print the report; return the process exit code."""
    args = parse_args(argv)
    client = ProbeClient(args.base_url, args.api_key)
    if args.stop_sequence is not None:
        return main_stop(client, args)
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as prompt_file:
                prompt = prompt_file.read()
        except OSError as error:
            print(f"live_probe: cannot read --prompt-file: {error}", file=sys.stderr)
            return 1
    else:
        prompt = cold_prompt(args.prompt_tokens, args.max_tokens, uuid.uuid4().hex)
    try:
        model = resolve_model(client, args.model)
        records = run_probe(
            client,
            model,
            prompt,
            args.max_tokens,
            args.request_timeout,
            repeat=not args.prompt_file,
        )
    except ProbeError as error:
        print(f"live_probe: {error}", file=sys.stderr)
        return 1
    print_report(model, records, args.json)
    return 0


# ------------------------------------------------------------------------------
def main_stop(client: ProbeClient, args) -> int:
    """Run the `--stop-sequence` probe and print its report; return the exit code.

    The report prints even when the finish was not the stop sequence's; the
    exit code is then 1.
    """
    try:
        model = resolve_model(client, args.model)
        record = run_stop_probe(
            client, model, args.stop_sequence, args.max_tokens, args.request_timeout
        )
    except ProbeError as error:
        print(f"live_probe: {error}", file=sys.stderr)
        return 1
    print_report(model, [record], args.json, REPORT_ROWS + STOP_ROWS)
    failure = stop_failure(record, args.stop_sequence, args.max_tokens)
    if failure:
        print(f"live_probe: not a stop-sequence finish: {failure}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------------------
def print_report(model: str, records: list[dict], as_json: bool, rows=REPORT_ROWS):
    """Print the records as JSON, or as the `rows` table with replies below it."""
    if as_json:
        print(json.dumps({"model": model, "requests": records}, indent=2))
    else:
        print(f"model: {model}")
        print(format_table(records, rows))
        print(format_replies(records))


if __name__ == "__main__":
    raise SystemExit(main())
