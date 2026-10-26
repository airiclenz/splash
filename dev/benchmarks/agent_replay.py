"""Find and replay the agent conversations in a Splash request recording.

Run as ``python -m dev.benchmarks.agent_replay chains --recording <file>``.
The recording is the JSON-lines file `server/server.py --record-requests`
writes, one ``{"received_unix_ms", "path", "body"}`` object per accepted
generation request. `chains` lists its request chains, largest first: runs of
requests on one path in which each request's message list starts with the
previous request's full message list, as an agent sends a growing history.

``run --recording <file> --chain <i> --base-url <url> --label <name> --output
<json>`` replays one chain serially and streaming at temperature 0 against an
otherwise idle server, and writes per turn the usage token counts, the client
TTFT and the native rates diffed from ``GET /status`` around the turn. A
per-run nonce line leads every turn's rendered prompt, so turn 1 is cold and no
run reuses another run's cache. ``summarize <json>...`` prints, per label, the
per-turn medians and whole-chain totals of those runs. `chains` and `summarize`
never contact a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from dev.benchmarks.contended_probe import sse_events
from dev.benchmarks.http_regression import status_delta
from dev.benchmarks.live_probe import (
    ACCEPTED,
    DECODE_TOKENS,
    DECODE_WALL,
    DRAFTED,
    PREFILL_WALL,
    REUSED,
    ProbeClient,
    ProbeError,
    fresh_status,
    ratio,
)

CHAT_PATH = "/v1/chat/completions"
ANTHROPIC_PATH = "/v1/messages"
RESPONSES_PATH = "/v1/responses"
# The body field holding each generation path's message list.
MESSAGE_FIELDS = {
    CHAT_PATH: "messages",
    ANTHROPIC_PATH: "messages",
    RESPONSES_PATH: "input",
}
# A `/v1/responses` body naming this field carries only the new turn, its
# history lives server side, so it cannot be placed in a chain.
PREVIOUS_RESPONSE_FIELD = "previous_response_id"
CHARACTERS_PER_TOKEN = 4
# A lone request is no conversation to replay.
MINIMUM_CHAIN_TURNS = 2
NO_DIGEST = b""

REQUEST_TIMEOUT_SECONDS = 1800.0
API_KEY_ENVIRONMENT = "SPLASH_API_KEY"
MILLISECONDS_PER_SECOND = 1000.0
# A follow-up turn has lost its prefix when its cached tokens fall more than
# this below the previous turn's prompt tokens.
PREFIX_LOSS_SLACK_TOKENS = 1024
NONCE_LABEL = "agent-replay run"
# Requests the native runtime admitted; exactly one per replayed turn.
ADMISSIONS = ("requests", "submitted")
TURN_PATHS = (
    PREFILL_WALL,
    DECODE_TOKENS,
    DECODE_WALL,
    DRAFTED,
    ACCEPTED,
    REUSED,
    ADMISSIONS,
)
# The completion-limit fields each path reads, the winning one first: the
# server prefers chat `max_completion_tokens` over `max_tokens`.
MAX_TOKEN_FIELDS = {
    CHAT_PATH: ("max_completion_tokens", "max_tokens"),
    ANTHROPIC_PATH: ("max_tokens",),
    RESPONSES_PATH: ("max_output_tokens",),
}
# The body field holding a path's system text; chat sends it as a message.
SYSTEM_FIELDS = {ANTHROPIC_PATH: "system", RESPONSES_PATH: "instructions"}
CHAT_TOKEN_DELTA_FIELDS = ("content", "reasoning_content", "tool_calls")
ANTHROPIC_TOKEN_DELTA_FIELDS = ("text", "thinking", "partial_json")
RESPONSES_TOKEN_EVENTS = frozenset(
    {
        "response.output_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
    }
)
RESPONSES_FINAL_EVENTS = frozenset({"response.completed", "response.incomplete"})
RESPONSES_FAILED_EVENT = "response.failed"
UNAVAILABLE = "n/a"


@dataclass(frozen=True)
class Turn:
    """One recorded request of a chain, without its body."""

    # --------------------------------------------------------------------------
    # 1-based line of the request in the recording.
    line: int
    # --------------------------------------------------------------------------
    # When the server accepted the request.
    received_unix_ms: int
    # --------------------------------------------------------------------------
    # Length of the request's message list.
    messages: int
    # --------------------------------------------------------------------------
    # Characters of every string in the request's message list.
    characters: int


@dataclass
class Chain:
    """A run of requests on one path, each extending the previous history."""

    # --------------------------------------------------------------------------
    # The generation path every turn was posted to.
    path: str
    # --------------------------------------------------------------------------
    # The turns in recording order; never empty.
    turns: list[Turn] = field(default_factory=list)

    # --------------------------------------------------------------------------
    def final_characters(self) -> int:
        """Characters of the final turn's message list."""
        return self.turns[-1].characters

    # --------------------------------------------------------------------------
    def final_estimated_tokens(self) -> int:
        """The final turn's prompt size, estimated as characters/4."""
        return self.final_characters() // CHARACTERS_PER_TOKEN


@dataclass(frozen=True)
class RecordedRequest:
    """One recording line reduced to what chain finding needs."""

    # --------------------------------------------------------------------------
    # The generation path, without a query string.
    path: str
    # --------------------------------------------------------------------------
    # When the server accepted the request.
    received_unix_ms: int
    # --------------------------------------------------------------------------
    # The request's message list, or None when it cannot join a chain.
    messages: list | None


# ------------------------------------------------------------------------------
def message_list(path: str, body: dict) -> list | None:
    """Return the history `body` sends on `path`, or None if it has none.

    A `/v1/responses` string `input` is its shorthand for one user message.
    Unknown paths, `previous_response_id` bodies and non-list histories (a
    request the server went on to reject) yield None.
    """
    message_field = MESSAGE_FIELDS.get(path)
    if message_field is None:
        return None
    if path == RESPONSES_PATH and body.get(PREVIOUS_RESPONSE_FIELD) is not None:
        return None
    messages = body.get(message_field)
    if path == RESPONSES_PATH and isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages if isinstance(messages, list) else None


# ------------------------------------------------------------------------------
def parse_line(text: str, line: int) -> RecordedRequest:
    """Decode one recording line; raise ProbeError when it is malformed."""
    try:
        record = json.loads(text)
    except ValueError as error:
        raise ProbeError(f"recording line {line} is not JSON: {error}") from error
    if not isinstance(record, dict):
        raise ProbeError(f"recording line {line} is not a JSON object")
    received = record.get("received_unix_ms")
    path = record.get("path")
    body = record.get("body")
    if isinstance(received, bool) or not isinstance(received, int):
        raise ProbeError(f"recording line {line} has no integer received_unix_ms")
    if not isinstance(path, str) or not isinstance(body, dict):
        raise ProbeError(
            f"recording line {line} needs a string path and an object body"
        )
    path = path.partition("?")[0]
    return RecordedRequest(path, received, message_list(path, body))


# ------------------------------------------------------------------------------
def read_recording(recording: str):
    """Yield (line, RecordedRequest) for each request in `recording`.

    Blank lines are skipped. An unterminated final line that is not JSON is a
    write still in progress and is skipped; any other malformed line, or an
    unreadable file, raises ProbeError.
    """
    try:
        with open(recording, encoding="utf-8") as stream:
            for line, text in enumerate(stream, start=1):
                if not text.strip():
                    continue
                try:
                    yield line, parse_line(text, line)
                except ProbeError:
                    if text.endswith("\n"):
                        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ProbeError(f"cannot read recording {recording}: {error}") from error


# ------------------------------------------------------------------------------
def string_characters(value) -> int:
    """Characters of every string nested in `value`; keys are not counted."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(string_characters(item) for item in value.values())
    if isinstance(value, list):
        return sum(string_characters(item) for item in value)
    return 0


# ------------------------------------------------------------------------------
def prefix_digests(messages: list) -> list[bytes]:
    """Return digests[k] identifying `messages[:k]`, for k = 0..len(messages).

    Each digest folds the previous one with the message's canonical JSON, so
    equal digests mean equal prefixes, whatever the key order of a message.
    """
    digests = [NO_DIGEST]
    for message in messages:
        encoded = json.dumps(message, sort_keys=True).encode("ascii")
        digests.append(hashlib.sha256(digests[-1] + encoded).digest())
    return digests


# ------------------------------------------------------------------------------
def find_chains(requests) -> list[Chain]:
    """Group (line, RecordedRequest) pairs into chains, largest final first.

    A request joins the chain on its path whose last message list is the
    longest proper prefix of its own; otherwise it starts a chain. When two
    chains end in the same list, the later one is extended. Chains of fewer
    than MINIMUM_CHAIN_TURNS turns are dropped; ties keep recording order.
    Cost: one hash per message and one dict probe per proper prefix, so time
    is linear in the recording and memory holds no request body.
    """
    chains: list[Chain] = []
    # (path, prefix length, prefix digest) of each chain's last message list.
    chain_ends: dict[tuple[str, int, bytes], int] = {}
    for line, request in requests:
        if request.messages is None:
            continue
        digests = prefix_digests(request.messages)
        chain_index = None
        for length in range(len(request.messages) - 1, 0, -1):
            chain_index = chain_ends.get((request.path, length, digests[length]))
            if chain_index is not None:
                del chain_ends[(request.path, length, digests[length])]
                break
        if chain_index is None:
            chain_index = len(chains)
            chains.append(Chain(request.path))
        chains[chain_index].turns.append(
            Turn(
                line,
                request.received_unix_ms,
                len(request.messages),
                string_characters(request.messages),
            )
        )
        chain_ends[(request.path, len(request.messages), digests[-1])] = chain_index
    found = [chain for chain in chains if len(chain.turns) >= MINIMUM_CHAIN_TURNS]
    return sorted(found, key=Chain.final_characters, reverse=True)


# ------------------------------------------------------------------------------
def chain_document(index: int, chain: Chain) -> dict:
    """Render one chain as the JSON object `chains --json` prints."""
    return {
        "index": index,
        "path": chain.path,
        "turn_count": len(chain.turns),
        "final_characters": chain.final_characters(),
        "final_estimated_tokens": chain.final_estimated_tokens(),
        "first_received_unix_ms": chain.turns[0].received_unix_ms,
        "last_received_unix_ms": chain.turns[-1].received_unix_ms,
        "turns": [
            {
                "line": turn.line,
                "received_unix_ms": turn.received_unix_ms,
                "messages": turn.messages,
            }
            for turn in chain.turns
        ],
    }


CHAIN_COLUMNS = (
    ("index", "index"),
    ("turn_count", "turns"),
    ("path", "path"),
    ("final_characters", "final chars"),
    ("final_estimated_tokens", "est. tokens"),
    ("first_received_unix_ms", "first received ms"),
    ("last_received_unix_ms", "last received ms"),
)


# ------------------------------------------------------------------------------
def format_grid(rows: list[list[str]]) -> str:
    """Render rows of string cells as right-aligned fixed-width columns."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.rjust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


# ------------------------------------------------------------------------------
def format_chains(documents: list[dict]) -> str:
    """Render chain documents as a fixed-width table, one row per chain."""
    rows = [[label for _, label in CHAIN_COLUMNS]]
    rows += [[str(document[key]) for key, _ in CHAIN_COLUMNS] for document in documents]
    return format_grid(rows)


# ------------------------------------------------------------------------------
def list_chains(args) -> int:
    """Print the recording's chains as a table or JSON; return the exit code."""
    chains = find_chains(read_recording(args.recording))
    documents = [chain_document(index, chain) for index, chain in enumerate(chains)]
    if args.json:
        print(json.dumps({"recording": args.recording, "chains": documents}, indent=2))
    elif documents:
        print(format_chains(documents))
    else:
        print(f"no chains of {MINIMUM_CHAIN_TURNS}+ turns in {args.recording}")
    return 0


@dataclass
class StreamReading:
    """What one replayed turn's stream reported, filled as its events arrive."""

    # --------------------------------------------------------------------------
    # Monotonic seconds of the first event carrying generated output, or None.
    first_token_at: float | None = None
    # --------------------------------------------------------------------------
    # The usage the stream reported; None until its usage event arrives.
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None

    # --------------------------------------------------------------------------
    def mark_token(self) -> None:
        """Record now as the first-token time unless one is already set."""
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()


# ------------------------------------------------------------------------------
def observe_chat(event: dict, reading: StreamReading) -> None:
    """Fold one chat-completions stream chunk into `reading`.

    Usage arrives only in the `stream_options.include_usage` chunk.
    """
    for choice in event.get("choices") or ():
        delta = choice.get("delta") or {}
        if any(delta.get(name) for name in CHAT_TOKEN_DELTA_FIELDS):
            reading.mark_token()
    usage = event.get("usage")
    if isinstance(usage, dict):
        reading.prompt_tokens = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details") or {}
        reading.cached_tokens = details.get("cached_tokens", 0)
        reading.completion_tokens = usage.get("completion_tokens")


# ------------------------------------------------------------------------------
def observe_anthropic(event: dict, reading: StreamReading) -> None:
    """Fold one Anthropic `/v1/messages` stream event into `reading`.

    `message_start` usage splits the prompt into uncached `input_tokens` and
    `cache_read_input_tokens`; `message_delta` carries the output tokens.
    """
    kind = event.get("type")
    if kind == "message_start":
        usage = (event.get("message") or {}).get("usage") or {}
        if usage.get("input_tokens") is None:
            return
        cached = usage.get("cache_read_input_tokens") or 0
        reading.prompt_tokens = usage["input_tokens"] + cached
        reading.cached_tokens = cached
    elif kind == "content_block_delta":
        delta = event.get("delta") or {}
        if any(delta.get(name) for name in ANTHROPIC_TOKEN_DELTA_FIELDS):
            reading.mark_token()
    elif kind == "message_delta":
        reading.completion_tokens = (event.get("usage") or {}).get("output_tokens")


# ------------------------------------------------------------------------------
def observe_responses(event: dict, reading: StreamReading) -> None:
    """Fold one `/v1/responses` stream event into `reading`.

    Raises ProbeError on `response.failed`, whose error sits inside `response`.
    """
    kind = event.get("type")
    if kind in RESPONSES_TOKEN_EVENTS and event.get("delta"):
        reading.mark_token()
    elif kind in RESPONSES_FINAL_EVENTS:
        usage = (event.get("response") or {}).get("usage") or {}
        reading.prompt_tokens = usage.get("input_tokens")
        details = usage.get("input_tokens_details") or {}
        reading.cached_tokens = details.get("cached_tokens", 0)
        reading.completion_tokens = usage.get("output_tokens")
    elif kind == RESPONSES_FAILED_EVENT:
        error = (event.get("response") or {}).get("error")
        raise ProbeError(f"stream failed: {json.dumps(error)[:500]}")


STREAM_OBSERVERS = {
    CHAT_PATH: observe_chat,
    ANTHROPIC_PATH: observe_anthropic,
    RESPONSES_PATH: observe_responses,
}


# ------------------------------------------------------------------------------
def prefixed_text(content, line: str):
    """Return `content` with `line` before its first text; ProbeError if none.

    `content` is a string or a list of parts, one of them holding a "text".
    """
    if isinstance(content, str):
        return line + content
    for part in content if isinstance(content, list) else ():
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            part["text"] = line + part["text"]
            return content
    raise ProbeError("cannot place the run nonce: the leading content has no text")


# ------------------------------------------------------------------------------
def place_nonce(path: str, body: dict, nonce: str) -> None:
    """Put the nonce line at the earliest text the chat template renders.

    The tool block renders before the system text, so the nonce leads
    `tools[0]`'s description when tools are sent, else the Anthropic `system`
    or Responses `instructions`, else the first message's text.
    """
    line = f"{nonce}\n"
    tools = body.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[0], dict):
        function = tools[0].get("function")
        holder = function if isinstance(function, dict) else tools[0]
        holder["description"] = line + (holder.get("description") or "")
        return
    system_field = SYSTEM_FIELDS.get(path)
    if system_field and body.get(system_field):
        body[system_field] = prefixed_text(body[system_field], line)
        return
    message_field = MESSAGE_FIELDS[path]
    messages = body.get(message_field)
    if isinstance(messages, str):
        body[message_field] = line + messages
        return
    first = messages[0] if isinstance(messages, list) and messages else None
    if not isinstance(first, dict):
        raise ProbeError("cannot place the run nonce: the body has no first message")
    first["content"] = prefixed_text(first.get("content"), line)


# ------------------------------------------------------------------------------
def cap_max_tokens(path: str, body: dict, cap: int | None) -> None:
    """Cap the completion limit `path` reads at `cap`, setting it when absent.

    Every limit field the body sends is set, so none outranks the cap.
    """
    if cap is None:
        return
    names = MAX_TOKEN_FIELDS[path]
    recorded = next((body[name] for name in names if body.get(name)), None)
    limit = cap if not isinstance(recorded, int) else min(recorded, cap)
    for name in names:
        if name == names[0] or name in body:
            body[name] = limit


# ------------------------------------------------------------------------------
def replay_body(path: str, body: dict, nonce: str, cap: int | None) -> dict:
    """Turn a recorded body into its replay: greedy, streaming, nonce-led.

    Mutates and returns `body`; every field not named here is sent as recorded.
    """
    body["temperature"] = 0
    body["stream"] = True
    if path == CHAT_PATH:
        options = body.get("stream_options")
        body["stream_options"] = {
            **(options if isinstance(options, dict) else {}),
            "include_usage": True,
        }
    cap_max_tokens(path, body, cap)
    place_nonce(path, body, nonce)
    return body


# ------------------------------------------------------------------------------
def open_stream(client: ProbeClient, path: str, body: dict, timeout: float):
    """POST a streaming `body` to `path` and return the open HTTP response.

    Raises ProbeError, with the HTTP status when the server refused it.
    """
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    prepared = urllib.request.Request(
        client.base_url + path,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        return urllib.request.urlopen(prepared, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise ProbeError(
            f"POST {path}: HTTP {error.code}: {detail}", error.code
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise ProbeError(f"POST {path}: {error}") from error


# ------------------------------------------------------------------------------
def read_stream(response, observe) -> StreamReading:
    """Read `response` to its end, folding each event with `observe`.

    The stream is never closed early: closing it would cancel the turn.
    Raises ProbeError on a stream error or a stream that reported no usage.
    """
    reading = StreamReading()
    with response:
        try:
            for event in sse_events(response):
                observe(event, reading)
        except OSError as error:
            raise ProbeError(f"stream: {error}") from error
    if reading.prompt_tokens is None or reading.completion_tokens is None:
        raise ProbeError("the stream ended without reporting usage")
    return reading


# ------------------------------------------------------------------------------
def with_rates(record: dict) -> dict:
    """Return `record` plus the rates derived from its raw counts.

    The tail prefill rate is uncached prompt tokens over the prefill wall.
    """
    uncached = record["prompt_tokens"] - record["cached_tokens"]
    return {
        **record,
        "tail_prefill_tokens_per_second": ratio(
            uncached * MILLISECONDS_PER_SECOND, record["prefill_wall_ms"]
        ),
        "decode_tokens_per_second": ratio(
            record["decode_tokens"] * MILLISECONDS_PER_SECOND,
            record["decode_wall_ms"],
        ),
        "draft_acceptance": ratio(
            record["accepted_draft_tokens"], record["drafted_tokens"]
        ),
    }


# ------------------------------------------------------------------------------
def turn_record(number, turn, reading, started, ended, delta, previous_prompt):
    """Build one turn's record from its stream reading and `/status` delta.

    `started`/`ended` bracket the request in monotonic seconds. Raises
    ProbeError unless the delta counts exactly one admission: another request
    on the server would corrupt every counter. Turn 1 (`previous_prompt`
    None) has no `prefix_lost` verdict.
    """
    admissions = delta[ADMISSIONS]
    if admissions != 1:
        raise ProbeError(
            f"turn {number}: /status counted {admissions:g} admissions, not 1; "
            "replay against an otherwise idle server"
        )
    ttft_ms = None
    if reading.first_token_at is not None:
        ttft_ms = (reading.first_token_at - started) * MILLISECONDS_PER_SECOND
    prefix_lost = None
    if previous_prompt is not None:
        prefix_lost = reading.cached_tokens < previous_prompt - PREFIX_LOSS_SLACK_TOKENS
    return with_rates(
        {
            "turn": number,
            "line": turn.line,
            "received_unix_ms": turn.received_unix_ms,
            "prompt_tokens": reading.prompt_tokens,
            "cached_tokens": reading.cached_tokens,
            "completion_tokens": reading.completion_tokens,
            "ttft_ms": ttft_ms,
            "http_wall_ms": (ended - started) * MILLISECONDS_PER_SECOND,
            "prefill_wall_ms": delta[PREFILL_WALL],
            "decode_tokens": delta[DECODE_TOKENS],
            "decode_wall_ms": delta[DECODE_WALL],
            "drafted_tokens": delta[DRAFTED],
            "accepted_draft_tokens": delta[ACCEPTED],
            "cache_reused_tokens": delta[REUSED],
            "prefix_lost": prefix_lost,
        }
    )


# ------------------------------------------------------------------------------
def chain_bodies(recording: str, chain: Chain):
    """Yield the recorded body of each of `chain`'s turns, in turn order.

    Reads the recording once and holds one body at a time.
    """
    wanted = {turn.line for turn in chain.turns}
    try:
        with open(recording, encoding="utf-8") as stream:
            for line, text in enumerate(stream, start=1):
                if line in wanted:
                    yield json.loads(text)["body"]
    except (OSError, UnicodeDecodeError, ValueError, KeyError) as error:
        raise ProbeError(f"cannot read recording {recording}: {error}") from error


# ------------------------------------------------------------------------------
def replay_chain(client: ProbeClient, chain: Chain, bodies, nonce, cap, timeout):
    """Replay `chain` turn by turn, each bracketed by fresh `/status` reads.

    Generated replies are discarded: each next turn is sent as recorded.
    Returns the turn records.
    """
    records = []
    previous_prompt = None
    for number, (turn, body) in enumerate(zip(chain.turns, bodies), start=1):
        body = replay_body(chain.path, body, nonce, cap)
        before = fresh_status(client)
        started = time.monotonic()
        response = open_stream(client, chain.path, body, timeout)
        reading = read_stream(response, STREAM_OBSERVERS[chain.path])
        ended = time.monotonic()
        after = fresh_status(client)
        delta = status_delta(before, after, TURN_PATHS)
        records.append(
            turn_record(number, turn, reading, started, ended, delta, previous_prompt)
        )
        previous_prompt = reading.prompt_tokens
    return records


# ------------------------------------------------------------------------------
def selected_chain(recording: str, index: int) -> Chain:
    """Return the chain `chains` lists at `index`; ProbeError when out of range."""
    chains = find_chains(read_recording(recording))
    if not 0 <= index < len(chains):
        raise ProbeError(
            f"--chain {index} is out of range: {recording} has {len(chains)} chains"
        )
    return chains[index]


# ------------------------------------------------------------------------------
def run_replay(args) -> int:
    """Replay one chain, write its run JSON, print its summary; return 0."""
    chain = selected_chain(args.recording, args.chain)
    client = ProbeClient(args.base_url, args.api_key)
    nonce = f"{NONCE_LABEL} {uuid.uuid4().hex}"
    records = replay_chain(
        client,
        chain,
        chain_bodies(args.recording, chain),
        nonce,
        args.max_tokens,
        args.request_timeout,
    )
    identity = chain_document(args.chain, chain)
    del identity["turns"]
    document = {
        "label": args.label,
        "recording": args.recording,
        "base_url": args.base_url,
        "chain": identity,
        "nonce": nonce,
        "max_tokens": args.max_tokens,
        "first_turn_cold": records[0]["cached_tokens"] == 0,
        "turns": records,
    }
    if not document["first_turn_cold"]:
        print(
            f"agent_replay: turn 1 reused {records[0]['cached_tokens']} cached "
            "tokens; it was not cold",
            file=sys.stderr,
        )
    try:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
    except OSError as error:
        raise ProbeError(f"cannot write {args.output}: {error}") from error
    print(format_summary([document]))
    return 0


# The raw per-turn counts a whole-chain total sums.
SUMMED_KEYS = (
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "ttft_ms",
    "http_wall_ms",
    "prefill_wall_ms",
    "decode_tokens",
    "decode_wall_ms",
    "drafted_tokens",
    "accepted_draft_tokens",
    "cache_reused_tokens",
)
# (record key, table label) in display order.
SUMMARY_COLUMNS = (
    ("prompt_tokens", "prompt"),
    ("cached_tokens", "cached"),
    ("completion_tokens", "completion"),
    ("ttft_ms", "TTFT ms"),
    ("prefill_wall_ms", "prefill ms"),
    ("tail_prefill_tokens_per_second", "tail prefill tok/s"),
    ("decode_tokens_per_second", "decode tok/s"),
    ("drafted_tokens", "drafted"),
    ("accepted_draft_tokens", "accepted"),
    ("draft_acceptance", "acceptance"),
)


# ------------------------------------------------------------------------------
def chain_totals(turns: list[dict]) -> dict:
    """Sum a run's raw turn counts and derive the whole-chain rates.

    A count no turn reported sums to 0; `prefix_lost` is true when any turn
    lost its prefix, None for a one-turn run.
    """
    totals = {
        key: sum(turn[key] for turn in turns if turn.get(key) is not None)
        for key in SUMMED_KEYS
    }
    verdicts = [
        turn["prefix_lost"] for turn in turns if turn["prefix_lost"] is not None
    ]
    totals["prefix_lost"] = any(verdicts) if verdicts else None
    return with_rates(totals)


# ------------------------------------------------------------------------------
def format_cell(value) -> str:
    """Render one summary cell; None reads as unavailable."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(int(value))


# ------------------------------------------------------------------------------
def summary_row(name: str, records: list[dict]) -> list[str]:
    """Return one table row: the median of each column, then the loss count."""
    cells = [name]
    for key, _ in SUMMARY_COLUMNS:
        values = [record[key] for record in records if record.get(key) is not None]
        cells.append(format_cell(statistics.median(values) if values else None))
    verdicts = [record["prefix_lost"] for record in records]
    judged = [verdict for verdict in verdicts if verdict is not None]
    cells.append(f"{sum(judged)}/{len(judged)}" if judged else UNAVAILABLE)
    return cells


# ------------------------------------------------------------------------------
def format_label(label: str, runs: list[dict]) -> str:
    """Render one label's runs: per-turn medians, then the median chain total."""
    caps = ", ".join(sorted({str(run.get("max_tokens")) for run in runs}))
    rows = [["turn", *(name for _, name in SUMMARY_COLUMNS), "prefix lost"]]
    for index in range(max(len(run["turns"]) for run in runs)):
        turns = [run["turns"][index] for run in runs if index < len(run["turns"])]
        rows.append(summary_row(str(index + 1), turns))
    rows.append(summary_row("total", [chain_totals(run["turns"]) for run in runs]))
    return f"{label}: {len(runs)} run(s), max tokens {caps}\n{format_grid(rows)}"


# ------------------------------------------------------------------------------
def format_summary(documents: list[dict]) -> str:
    """Render the run documents grouped by label, in first-seen label order."""
    labels: dict[str, list[dict]] = {}
    for document in documents:
        labels.setdefault(document["label"], []).append(document)
    return "\n\n".join(format_label(label, runs) for label, runs in labels.items())


# ------------------------------------------------------------------------------
def load_run(path: str) -> dict:
    """Read one `run` output file; ProbeError when it is not one."""
    try:
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ProbeError(f"cannot read run {path}: {error}") from error
    if not (
        isinstance(document, dict)
        and isinstance(document.get("label"), str)
        and isinstance(document.get("chain"), dict)
        and isinstance(document.get("turns"), list)
        and document["turns"]
    ):
        raise ProbeError(f"{path} is not an agent_replay run")
    return document


# ------------------------------------------------------------------------------
def summarize_runs(args) -> int:
    """Print the runs' per-label summary; return the exit code.

    Raises ProbeError when the runs replay different chains, since their
    turns would then not compare at the same depth.
    """
    documents = [load_run(path) for path in args.runs]
    identities = {
        (
            document["chain"].get("path"),
            document["chain"].get("first_received_unix_ms"),
            document["chain"].get("turn_count"),
        )
        for document in documents
    }
    if len(identities) > 1:
        raise ProbeError(
            f"the runs replay different chains: {sorted(map(str, identities))}"
        )
    print(format_summary(documents))
    return 0


# ------------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse the command line into a subcommand and its options."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    chains = commands.add_parser("chains", help="list the request chains")
    chains.add_argument("--recording", required=True, help="--record-requests file")
    chains.add_argument("--json", action="store_true", help="print the chains as JSON")
    chains.set_defaults(handler=list_chains)
    run = commands.add_parser("run", help="replay one chain against a server")
    run.add_argument("--recording", required=True, help="--record-requests file")
    run.add_argument(
        "--chain", type=int, required=True, help="chain index as `chains` lists it"
    )
    run.add_argument(
        "--base-url", required=True, help="running server, e.g. http://127.0.0.1:1111"
    )
    run.add_argument("--label", required=True, help="the build this run measures")
    run.add_argument("--output", required=True, help="run JSON to write")
    run.add_argument(
        "--max-tokens",
        type=int,
        help="cap each turn's completion limit (default: as recorded)",
    )
    run.add_argument(
        "--api-key",
        default=os.environ.get(API_KEY_ENVIRONMENT),
        help=f"Bearer key (default: ${API_KEY_ENVIRONMENT})",
    )
    run.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS)
    run.set_defaults(handler=run_replay)
    summarize = commands.add_parser("summarize", help="compare run JSONs by label")
    summarize.add_argument("runs", nargs="+", help="`run` output files")
    summarize.set_defaults(handler=summarize_runs)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.max_tokens is not None and args.max_tokens < 1:
            parser.error("--max-tokens must be positive")
        if args.request_timeout <= 0:
            parser.error("--request-timeout must be positive")
        if not args.base_url.startswith(("http://", "https://")):
            parser.error("--base-url must start with http:// or https://")
    return args


# ------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Run the chosen subcommand; return the process exit code."""
    args = parse_args(argv)
    try:
        return args.handler(args)
    except ProbeError as error:
        print(f"agent_replay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
