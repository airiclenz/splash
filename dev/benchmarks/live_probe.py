"""Probe a running Splash server's prefill and decode rates.

Run as ``python -m dev.benchmarks.live_probe --base-url http://127.0.0.1:8000``.
Sends one cold prompt and then its exact repeat, and reports per request the
native rates and cache/pressure counters diffed from ``GET /status`` taken just
before and just after it. Never starts a server; the counters are server-wide,
so run it against an otherwise idle server.
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
STALE_SNAPSHOT_RETRIES = 3
STALE_RETRY_DELAY_SECONDS = 0.15
API_KEY_ENVIRONMENT = "SPLASH_API_KEY"
UNAVAILABLE = "unavailable"

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
    ("http_wall_ms", "HTTP wall ms"),
)


class ProbeError(RuntimeError):
    """The server refused a request or never produced a fresh `/status`."""


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
        headers = {"Accept": "application/json"}
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
            with urllib.request.urlopen(prepared, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise ProbeError(f"{method} {path}: HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise ProbeError(f"{method} {path}: {error}") from error
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProbeError(f"{method} {path} returned invalid JSON") from error


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
def probe_record(label: str, before: dict, after: dict, reply: dict, wall_ms):
    """Build one request's report row from its bracketing `/status` snapshots.

    Every rate and counter comes from the snapshot diff; only the token counts
    come from the reply's `usage`. A rate whose denominator delta is 0 is None.
    """
    delta = status_delta(before, after, PROBE_PATHS)
    usage = reply.get("usage") or {}
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
        "http_wall_ms": wall_ms,
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
def run_probe(client: ProbeClient, model: str, prompt: str, max_tokens, timeout):
    """Send `prompt` cold, then its exact repeat; return both report rows.

    Each `/status` snapshot is taken outside the timed window.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    records = []
    for label in ("cold", "repeat"):
        before = fresh_status(client)
        started = time.monotonic()
        reply = client.request("POST", "/v1/chat/completions", body, timeout=timeout)
        wall_ms = (time.monotonic() - started) * 1000.0
        after = fresh_status(client)
        records.append(probe_record(label, before, after, reply, wall_ms))
    return records


# ------------------------------------------------------------------------------
def format_value(value) -> str:
    """Render one table cell; None reads as unavailable."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(int(value))


# ------------------------------------------------------------------------------
def format_table(records: list[dict]) -> str:
    """Render the records as a fixed-width table, one column per request."""
    width = max(len(label) for _, label in REPORT_ROWS)
    header = f"{'':<{width}}" + "".join(
        f"  {record['request']:>12}" for record in records
    )
    lines = [header]
    for key, label in REPORT_ROWS:
        cells = "".join(f"  {format_value(record[key]):>12}" for record in records)
        lines.append(f"{label:<{width}}{cells}")
    return "\n".join(lines)


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
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--json", action="store_true", help="print the records as JSON")
    args = parser.parse_args(argv)
    if args.prompt_tokens < MINIMUM_PROMPT_TOKENS:
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
    try:
        model = resolve_model(client, args.model)
        prompt = cold_prompt(args.prompt_tokens, args.max_tokens, uuid.uuid4().hex)
        records = run_probe(
            client, model, prompt, args.max_tokens, args.request_timeout
        )
    except ProbeError as error:
        print(f"live_probe: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"model": model, "requests": records}, indent=2))
    else:
        print(f"model: {model}")
        print(format_table(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
