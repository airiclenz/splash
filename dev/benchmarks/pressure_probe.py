"""Probe a running Splash server under host memory pressure.

Run as ``python -m dev.benchmarks.pressure_probe --base-url http://127.0.0.1:8000``.
Warms N long fixed-nonce conversations one at a time, then sends them all
concurrently, each extended by more tokens and with a decode budget. Samples
``GET /status`` and ``kern.memorystatus_vm_pressure_level`` every few seconds
and reports per lane its cached tokens and wall, the server-wide counter deltas
of each phase, and the time spent in server and OS memory pressure.

A swap guard stops the probe when ``vm_stat`` swapouts grow past a cap, to
spare the SSD; the server keeps working on requests already sent. Never starts
a server; the counters are server-wide, so run it against an otherwise idle
server.

The burst prompt extends the warm prompt's filler words but drops its closing
instruction, so it diverges just before the warm prompt's end: only a state
published at or below the divergence can be reused.

``--agent-turns N`` drives one conversation instead, the way an agent does:
each turn is sent right after the previous one finishes and carries the
previous prompt, the returned reply and ``--tool-tokens`` filler words, so the
whole earlier conversation is reusable. Each turn is its own phase, reported
with its cached tokens, TTFT, counter deltas and OS warn seconds; the probe
exits AGENT_COLD_TURN_EXIT_CODE when a turn after the first reuses 0 tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid

from dev.benchmarks.live_probe import (
    API_KEY_ENVIRONMENT,
    FILLER_WORDS,
    ProbeClient,
    ProbeError,
    format_value,
    fresh_status,
    resolve_model,
    status_reading,
)

DEFAULT_LANES = 2
DEFAULT_WARM_TOKENS = 50000
DEFAULT_EXTEND_TOKENS = 8000
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOOL_TOKENS = 2000
DEFAULT_SAMPLE_SECONDS = 3.0
DEFAULT_SWAP_GUARD_MB = 512
DEFAULT_TIMEOUT_SECONDS = 7200.0
WARM_MAX_TOKENS = 8
SWAP_POLL_SECONDS = 1.0
STATUS_TIMEOUT_SECONDS = 30.0
MINIMUM_WARM_TOKENS = 64
# Tokens left for the nonce, the closing instruction and the chat template.
PROMPT_FRAME_TOKENS = 16
WARM_INSTRUCTION = "Reply with the word done."
OS_PRESSURE_SYSCTL = "kern.memorystatus_vm_pressure_level"
# kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical.
OS_PRESSURE_WARN = 2
SERVER_NORMAL = "normal"
SWAP_ABORT_EXIT_CODE = 3
AGENT_COLD_TURN_EXIT_CODE = 4
AGENT_LANE = 0
# The chat reply path of the server-side TTFT (prefill wall, no network).
TTFT_PATH = ("metrics", "request_latency", "start_to_first_token_ms")
# Arrival to engine admission: the time a lane waited unadmitted, which the
# engine TTFT above leaves out.
QUEUE_PATH = ("metrics", "request_latency", "queue_to_start_ms")
BYTES_PER_MB = 2**20

# The `/status` causes that split checkpoint_publication_failures.
CHECKPOINT_FAILURE_CAUSES = (
    "missed_boundary",
    "retire_blocked",
    "snapshot_denied",
    "decode_overshoot",
    "error",
)
# (record key, `/status` path) of the cumulative counters diffed per phase.
PHASE_COUNTERS = (
    ("prefill_tokens", ("metrics", "prefill_input_tokens")),
    ("prefill_wall_ms", ("metrics", "prefill_wall_ms")),
    ("decode_tokens", ("metrics", "decode_output_tokens")),
    ("decode_wall_ms", ("metrics", "decode_wall_ms")),
    ("reused_tokens", ("cache", "reused_tokens")),
    ("suspensions", ("cache", "resource_suspensions")),
    ("replay_tokens", ("cache", "resource_replay_tokens")),
    ("decode_suspensions", ("cache", "resource_decode_suspensions")),
    ("decode_replay_tokens", ("cache", "resource_decode_replay_tokens")),
    ("denied_reservations", ("memory_governor", "denied_reservations")),
    (
        "denied_host_reservations",
        ("memory_governor", "denied_host_reservations"),
    ),
    (
        "denied_engine_reservations",
        ("memory_governor", "denied_engine_reservations"),
    ),
    ("state_evictions", ("state", "evictions")),
    ("checkpoint_evictions", ("state", "checkpoint_evictions")),
    ("checkpoint_retirements", ("state", "checkpoint_retirements")),
    ("kv_evictions", ("kv", "evictions")),
    ("checkpoint_publication_failures", ("cache", "checkpoint_publication_failures")),
    *(
        (
            f"checkpoint_failures_{cause}",
            ("cache", "checkpoint_publication_failure_causes", cause),
        )
        for cause in CHECKPOINT_FAILURE_CAUSES
    ),
    (
        "replay_publication_failures",
        ("cache", "replay_state_publication_failures"),
    ),
    ("capacity_failures", ("metrics", "capacity_failures")),
)
# (sample key, `/status` path) of the point readings each sample keeps.
SAMPLE_READINGS = (
    ("suspended", ("admission", "suspended")),
    ("waiting_memory", ("admission", "waiting_memory")),
    ("waiting_schedule", ("admission", "waiting_schedule")),
    ("governor_headroom_bytes", ("memory_governor", "headroom_bytes")),
    ("host_available_bytes", ("memory_governor", "host_available_bytes")),
    ("host_headroom_bytes", ("memory_governor", "host_headroom_bytes")),
    ("host_wired_bytes", ("memory_governor", "host_wired_bytes")),
    ("host_compressor_bytes", ("memory_governor", "host_compressor_bytes")),
    ("kv_pages_cache", ("kv", "pages_cache")),
    ("kv_pages_resident", ("kv", "pages_resident")),
)
# (summary key, sample key) of the readings whose minimum the summary reports.
MINIMUM_READINGS = (
    ("min_governor_headroom_bytes", "governor_headroom_bytes"),
    ("min_host_available_bytes", "host_available_bytes"),
    ("min_host_headroom_bytes", "host_headroom_bytes"),
)
LANE_COLUMNS = (
    ("lane", "lane"),
    ("phase", "phase"),
    ("prompt_tokens", "prompt"),
    ("cached_tokens", "cached"),
    ("completion_tokens", "completion"),
    ("queue_ms", "queue ms"),
    ("ttft_ms", "ttft ms"),
    ("wall_s", "wall s"),
)


# ------------------------------------------------------------------------------
def counter_delta(before: dict, after: dict) -> dict:
    """Diff the PHASE_COUNTERS between two `/status` snapshots.

    A counter the server does not export reads as None rather than 0, so an
    older build cannot pass for one that counted nothing.
    """
    delta = {}
    for key, path in PHASE_COUNTERS:
        start, end = status_reading(before, path), status_reading(after, path)
        delta[key] = None if start is None or end is None else end - start
    return delta


# ------------------------------------------------------------------------------
def status_sample(snapshot: dict, os_level, phase: str, at: float) -> dict:
    """Return one pressure sample from a `/status` snapshot and the OS level."""
    pressure = snapshot.get("memory_pressure")
    return {
        "t": at,
        "phase": phase,
        "server_pressure": pressure if isinstance(pressure, str) else None,
        "os_level": os_level,
        **{key: status_reading(snapshot, path) for key, path in SAMPLE_READINGS},
    }


# ------------------------------------------------------------------------------
def pressure_summary(samples: list[dict]) -> dict:
    """Summarise the pressure samples of one probe run.

    Time in a state is the sum of the intervals that start in it, so the last
    sample contributes no time. Failed samples (no server pressure) are skipped.
    """
    usable = [sample for sample in samples if sample.get("server_pressure")]
    summary = {
        "samples": len(usable),
        "server_warning_s": 0.0,
        "os_warn_s": 0.0,
        "os_warn_while_server_normal_s": 0.0,
        "server_flips": 0,
    }
    for current, following in zip(usable, usable[1:]):
        interval = following["t"] - current["t"]
        server_raised = current["server_pressure"] != SERVER_NORMAL
        os_raised = (current.get("os_level") or 0) >= OS_PRESSURE_WARN
        if server_raised:
            summary["server_warning_s"] += interval
        if os_raised:
            summary["os_warn_s"] += interval
            if not server_raised:
                summary["os_warn_while_server_normal_s"] += interval
        if following["server_pressure"] != current["server_pressure"]:
            summary["server_flips"] += 1
    for summary_key, sample_key in MINIMUM_READINGS:
        readings = [s[sample_key] for s in usable if s.get(sample_key) is not None]
        summary[summary_key] = min(readings) if readings else None
    return summary


# ------------------------------------------------------------------------------
def parse_vm_stat(text: str) -> tuple[int, int]:
    """Return (page size in bytes, cumulative swapouts in pages) from `vm_stat`."""
    page = re.search(r"page size of (\d+) bytes", text)
    swapouts = re.search(r"Swapouts:\s+(\d+)", text)
    if not page or not swapouts:
        raise ProbeError("vm_stat output carries no page size or swapout count")
    return int(page.group(1)), int(swapouts.group(1))


# ------------------------------------------------------------------------------
def read_vm_stat() -> tuple[int, int]:
    """Run `vm_stat` and return (page size, swapouts in pages)."""
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    return parse_vm_stat(output.stdout)


# ------------------------------------------------------------------------------
def read_os_pressure_level():
    """Return `kern.memorystatus_vm_pressure_level`, or None when unreadable."""
    try:
        output = subprocess.run(
            ["sysctl", "-n", OS_PRESSURE_SYSCTL], capture_output=True, text=True
        )
        return int(output.stdout.strip())
    except (OSError, ValueError):
        return None


class SwapGuard:
    """Trips once `vm_stat` swapouts grow past a byte cap over the first reading."""

    # --------------------------------------------------------------------------
    def __init__(self, cap_bytes: int, read=read_vm_stat):
        """Take the baseline reading now; `read` returns (page size, swapouts)."""
        self.cap_bytes = cap_bytes
        self.read = read
        self.page_bytes, self.baseline_pages = read()
        self.grown_bytes = 0
        self.tripped = threading.Event()

    # --------------------------------------------------------------------------
    def check(self) -> bool:
        """Take one reading; trip and return True once growth exceeds the cap."""
        _page_bytes, pages = self.read()
        self.grown_bytes = (pages - self.baseline_pages) * self.page_bytes
        if self.grown_bytes > self.cap_bytes:
            self.tripped.set()
        return self.tripped.is_set()

    # --------------------------------------------------------------------------
    def watch(self, stop: threading.Event) -> None:
        """Check every SWAP_POLL_SECONDS until tripped or `stop` is set."""
        while not stop.is_set() and not self.check():
            stop.wait(SWAP_POLL_SECONDS)


class Sampler:
    """Samples `/status` and the OS pressure level on a background thread."""

    # --------------------------------------------------------------------------
    def __init__(self, client: ProbeClient, interval: float, sink=None):
        """Sample `client` every `interval` s; append each sample to `sink` as JSON."""
        self.client = client
        self.interval = interval
        self.sink = sink
        self.phase = "start"
        self.samples: list[dict] = []

    # --------------------------------------------------------------------------
    def sample_once(self) -> None:
        """Take one sample; a failed `/status` read is kept as an error row."""
        at = time.time()
        try:
            snapshot = self.client.request(
                "GET", "/status", timeout=STATUS_TIMEOUT_SECONDS
            )
            sample = status_sample(snapshot, read_os_pressure_level(), self.phase, at)
        except ProbeError as error:
            sample = {"t": at, "phase": self.phase, "error": str(error)}
        self.samples.append(sample)
        if self.sink:
            self.sink.write(json.dumps(sample) + "\n")
            self.sink.flush()

    # --------------------------------------------------------------------------
    def run(self, stop: threading.Event) -> None:
        """Sample until `stop` is set."""
        while not stop.is_set():
            self.sample_once()
            stop.wait(self.interval)


# ------------------------------------------------------------------------------
def lane_words(nonce: str, count: int) -> list[str]:
    """Return the lane's first `count` filler words, fixed by its nonce."""
    return random.Random(nonce).choices(FILLER_WORDS, k=count)


# ------------------------------------------------------------------------------
def lane_prompt(nonce: str, words: list[str], instruction: str) -> str:
    """Return a lane prompt: nonce, filler words, closing instruction."""
    return f"{nonce}\n{' '.join(words)}\n{instruction}"


# ------------------------------------------------------------------------------
def chat_body(model: str, prompt: str, max_tokens: int) -> dict:
    """Return a greedy, non-thinking chat request for `prompt`."""
    return conversation_body(model, [{"role": "user", "content": prompt}], max_tokens)


# ------------------------------------------------------------------------------
def conversation_body(model: str, messages: list[dict], max_tokens: int) -> dict:
    """Return a greedy, non-thinking chat request continuing `messages`."""
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }


# ------------------------------------------------------------------------------
def lane_record(lane: int, phase: str, reply, wall_s: float, error) -> dict:
    """Build one lane's report row from its reply (None when it failed)."""
    usage = (reply or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "lane": lane,
        "phase": phase,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "queue_ms": status_reading(reply or {}, QUEUE_PATH),
        "ttft_ms": status_reading(reply or {}, TTFT_PATH),
        "wall_s": round(wall_s, 1),
        "error": error,
    }


# ------------------------------------------------------------------------------
def send_lane(client, lane, phase, body, timeout, records, replies=None) -> None:
    """Send one lane's request and append its row to `records`.

    When `replies` is given, the decoded reply (None on failure) is stored
    there under `lane`. A failure is also printed at once, so its server
    message survives a run that stops before the report.
    """
    started = time.monotonic()
    reply, error = None, None
    try:
        reply = client.request("POST", "/v1/chat/completions", body, timeout=timeout)
    except ProbeError as caught:
        error = str(caught)
        print(f"pressure_probe: lane {lane} {phase}: {error}", file=sys.stderr)
    records.append(lane_record(lane, phase, reply, time.monotonic() - started, error))
    if replies is not None:
        replies[lane] = reply


# ------------------------------------------------------------------------------
def run_phase(client, sampler, guard, phase, bodies, timeout, replies=None) -> dict:
    """Send `bodies` (lane -> request) concurrently and wait for all or the guard.

    Returns the phase row: lane rows, counter deltas and whether the swap guard
    stopped it. On a trip the unfinished lanes are left to the server and the
    after-snapshot is taken at once. `replies`, when given, receives each
    finished lane's decoded reply.
    """
    sampler.phase = phase
    before = fresh_status(client)
    records: list[dict] = []
    threads = [
        threading.Thread(
            target=send_lane,
            args=(client, lane, phase, body, timeout, records, replies),
            daemon=True,
        )
        for lane, body in bodies.items()
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        if guard.tripped.wait(SWAP_POLL_SECONDS):
            break
    after = fresh_status(client)
    return {
        "phase": phase,
        "lanes": sorted(records, key=lambda record: record["lane"]),
        "swap_aborted": guard.tripped.is_set(),
        **counter_delta(before, after),
    }


# ------------------------------------------------------------------------------
def run_probe(client, model, args, sampler, guard) -> list[dict]:
    """Warm each lane alone (unless `args.nonces` names warmed ones), then burst.

    Stops once the swap guard has tripped or a warm-up fails.
    """
    warm_words = max(1, args.warm_tokens - PROMPT_FRAME_TOKENS)
    burst_words = warm_words + args.extend_tokens
    instruction = f"Write the integers from 1 to {args.max_tokens * 2}, one per line."
    phases = []
    for lane, nonce in enumerate(args.nonces if args.warm else []):
        if guard.tripped.is_set():
            return phases
        words = lane_words(nonce, warm_words)
        body = chat_body(
            model, lane_prompt(nonce, words, WARM_INSTRUCTION), WARM_MAX_TOKENS
        )
        phase = run_phase(
            client, sampler, guard, f"warm{lane}", {lane: body}, args.request_timeout
        )
        phases.append(phase)
        if phase["swap_aborted"] or phase["lanes"][0]["error"]:
            return phases
    if guard.tripped.is_set():
        return phases
    bodies = {
        lane: chat_body(
            model,
            lane_prompt(nonce, lane_words(nonce, burst_words), instruction),
            args.max_tokens,
        )
        for lane, nonce in enumerate(args.nonces)
    }
    phases.append(
        run_phase(client, sampler, guard, "burst", bodies, args.request_timeout)
    )
    return phases


# ------------------------------------------------------------------------------
def reply_text(reply) -> str | None:
    """Return the assistant content of a chat reply, or None when it has none."""
    choices = (reply or {}).get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else None


# ------------------------------------------------------------------------------
def run_agent(client, model, args, sampler, guard) -> list[dict]:
    """Drive one conversation for `args.agent_turns` back-to-back turns.

    Turn 0 is the warm-up-sized prompt; each later turn resends the whole
    conversation, the previous reply and `args.tool_tokens` new filler words,
    as an agent does after a tool call. Stops once the swap guard has tripped,
    a turn fails or a reply carries no content to resend.
    """
    nonce = args.nonces[0]
    instruction = f"Write the integers from 1 to {args.max_tokens * 2}, one per line."
    words = lane_words(nonce, max(1, args.warm_tokens - PROMPT_FRAME_TOKENS))
    messages = [{"role": "user", "content": lane_prompt(nonce, words, instruction)}]
    phases = []
    for turn in range(args.agent_turns):
        if guard.tripped.is_set():
            return phases
        body = conversation_body(model, messages, args.max_tokens)
        replies: dict = {}
        phase = run_phase(
            client,
            sampler,
            guard,
            f"turn{turn}",
            {AGENT_LANE: body},
            args.request_timeout,
            replies,
        )
        phases.append(phase)
        content = reply_text(replies.get(AGENT_LANE))
        record = phase["lanes"][0] if phase["lanes"] else None
        if phase["swap_aborted"] or record is None or record["error"]:
            return phases
        if content is None:
            record["error"] = "reply carries no message content to resend"
            return phases
        tool_words = lane_words(f"{nonce}-{turn}", args.tool_tokens)
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"{' '.join(tool_words)}\n{instruction}"},
        ]
    return phases


# ------------------------------------------------------------------------------
def has_cold_turn(phases: list[dict]) -> bool:
    """Return whether a finished turn after the first reused 0 cached tokens."""
    return any(
        record["cached_tokens"] == 0
        for phase in phases[1:]
        for record in phase["lanes"]
    )


# ------------------------------------------------------------------------------
def phase_os_warn_seconds(samples: list[dict], phase: str) -> float:
    """Return the OS warn seconds of the samples taken during `phase`."""
    taken = [sample for sample in samples if sample.get("phase") == phase]
    return pressure_summary(taken)["os_warn_s"]


# ------------------------------------------------------------------------------
def format_report(document: dict) -> str:
    """Render the probe document as fixed-width text."""
    lines = [f"model: {document['model']}", f"nonces: {' '.join(document['nonces'])}"]
    header = "".join(f"{label:>12}" for _, label in LANE_COLUMNS)
    lines += ["", header]
    for phase in document["phases"]:
        for record in phase["lanes"]:
            cells = (
                record[key]
                if isinstance(record[key], str)
                else format_value(record[key])
                for key, _ in LANE_COLUMNS
            )
            lines.append("".join(f"{cell:>12}" for cell in cells))
            if record["error"]:
                lines.append(f"    error: {record['error']}")
    width = max(len(key) for key, _ in PHASE_COUNTERS)
    names = [phase["phase"] for phase in document["phases"]]
    lines += ["", f"{'':<{width}}" + "".join(f"{name:>12}" for name in names)]
    for key, _ in PHASE_COUNTERS:
        cells = "".join(
            f"{format_value(phase[key]):>12}" for phase in document["phases"]
        )
        lines.append(f"{key:<{width}}{cells}")
    cells = "".join(
        f"{format_value(phase['os_warn_s']):>12}" for phase in document["phases"]
    )
    lines.append(f"{'os_warn_s':<{width}}{cells}")
    lines.append("")
    for key, value in document["pressure"].items():
        lines.append(f"{key:<{width}}{format_value(value):>12}")
    if document["swap_aborted"]:
        grown = document["swap_grown_bytes"] / BYTES_PER_MB
        lines.append(f"\nswap guard tripped: swapouts grew {grown:.0f} MB")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="running server, e.g. http://127.0.0.1:8000"
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
        "--lanes", type=int, help=f"default {DEFAULT_LANES}, or one per --nonces value"
    )
    parser.add_argument("--warm-tokens", type=int, default=DEFAULT_WARM_TOKENS)
    parser.add_argument("--extend-tokens", type=int, default=DEFAULT_EXTEND_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--agent-turns",
        type=int,
        help="drive one conversation for this many back-to-back turns instead",
    )
    parser.add_argument(
        "--tool-tokens",
        type=int,
        default=DEFAULT_TOOL_TOKENS,
        help="filler words each agent turn adds after the reply",
    )
    parser.add_argument(
        "--nonces", nargs="+", help="reuse lanes an earlier run warmed; skips warm-up"
    )
    parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)
    parser.add_argument(
        "--swap-guard-mb",
        type=int,
        default=DEFAULT_SWAP_GUARD_MB,
        help="stop once swapouts grow by more than this",
    )
    parser.add_argument(
        "--samples", help="append every pressure sample to this JSONL file"
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if args.warm_tokens < MINIMUM_WARM_TOKENS:
        parser.error(f"--warm-tokens must be at least {MINIMUM_WARM_TOKENS}")
    if args.nonces is not None and args.lanes not in (None, len(args.nonces)):
        parser.error(
            f"--lanes {args.lanes} does not match the {len(args.nonces)} --nonces values"
        )
    if args.lanes is None:
        args.lanes = DEFAULT_LANES if args.nonces is None else len(args.nonces)
    if min(args.lanes, args.extend_tokens, args.max_tokens, args.swap_guard_mb) < 1:
        parser.error(
            "--lanes, --extend-tokens, --max-tokens and --swap-guard-mb must be positive"
        )
    if args.agent_turns is not None:
        if min(args.agent_turns, args.tool_tokens) < 1:
            parser.error("--agent-turns and --tool-tokens must be positive")
        if args.nonces is not None:
            parser.error("--agent-turns starts its own conversation; drop --nonces")
        args.lanes = 1
    if args.sample_seconds <= 0 or args.request_timeout <= 0:
        parser.error("--sample-seconds and --request-timeout must be positive")
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must start with http:// or https://")
    args.warm = args.nonces is None
    if args.warm:
        args.nonces = [uuid.uuid4().hex for _ in range(args.lanes)]
    return args


# ------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Run the probe and print the report; return the process exit code.

    Exits SWAP_ABORT_EXIT_CODE when the swap guard stopped the run, else
    AGENT_COLD_TURN_EXIT_CODE when an agent turn after the first reused 0 tokens.
    """
    args = parse_args(argv)
    client = ProbeClient(args.base_url, args.api_key)
    stop = threading.Event()
    sink = open(args.samples, "a") if args.samples else None
    threads = []
    try:
        model = resolve_model(client, args.model)
        guard = SwapGuard(args.swap_guard_mb * BYTES_PER_MB, read_vm_stat)
        sampler = Sampler(client, args.sample_seconds, sink)
        threads = [
            threading.Thread(target=target, args=(stop,), daemon=True)
            for target in (guard.watch, sampler.run)
        ]
        for thread in threads:
            thread.start()
        print(f"pressure_probe: nonces {' '.join(args.nonces)}", file=sys.stderr)
        run = run_probe if args.agent_turns is None else run_agent
        phases = run(client, model, args, sampler, guard)
    except (ProbeError, subprocess.CalledProcessError) as error:
        print(f"pressure_probe: {error}", file=sys.stderr)
        return 1
    finally:
        stop.set()
        # A sample in flight finishes before its sink closes.
        for thread in threads:
            thread.join(STATUS_TIMEOUT_SECONDS)
        if sink:
            sink.close()
    for phase in phases:
        phase["os_warn_s"] = phase_os_warn_seconds(sampler.samples, phase["phase"])
    document = {
        "model": model,
        "nonces": args.nonces,
        "phases": phases,
        "pressure": pressure_summary(sampler.samples),
        "swap_aborted": guard.tripped.is_set(),
        "swap_grown_bytes": guard.grown_bytes,
    }
    print(json.dumps(document, indent=2) if args.json else format_report(document))
    if document["swap_aborted"]:
        return SWAP_ABORT_EXIT_CODE
    if args.agent_turns is not None and has_cold_turn(phases):
        return AGENT_COLD_TURN_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
