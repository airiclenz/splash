"""Find and replay the agent conversations in a Splash request recording.

Run as ``python -m dev.benchmarks.agent_replay chains --recording <file>``.
The recording is the JSON-lines file `server/server.py --record-requests`
writes, one ``{"received_unix_ms", "path", "body"}`` object per accepted
generation request. `chains` lists its request chains, largest first: runs of
requests on one path in which each request's message list starts with the
previous request's full message list, as an agent sends a growing history.
Never contacts a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field

from dev.benchmarks.live_probe import ProbeError

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
def format_chains(documents: list[dict]) -> str:
    """Render chain documents as a fixed-width table, one row per chain."""
    rows = [[label for _, label in CHAIN_COLUMNS]]
    rows += [[str(document[key]) for key, _ in CHAIN_COLUMNS] for document in documents]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.rjust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


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


# ------------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse the command line into a subcommand and its options."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    chains = commands.add_parser("chains", help="list the request chains")
    chains.add_argument("--recording", required=True, help="--record-requests file")
    chains.add_argument("--json", action="store_true", help="print the chains as JSON")
    chains.set_defaults(handler=list_chains)
    return parser.parse_args(argv)


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
