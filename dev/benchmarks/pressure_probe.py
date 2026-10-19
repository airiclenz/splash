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
BYTES_PER_MB = 2**20

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
    ("state_evictions", ("state", "evictions")),
    ("checkpoint_evictions", ("state", "checkpoint_evictions")),
    ("checkpoint_retirements", ("state", "checkpoint_retirements")),
    ("kv_evictions", ("kv", "evictions")),
    ("checkpoint_publication_failures", ("cache", "checkpoint_publication_failures")),
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
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
        "wall_s": round(wall_s, 1),
        "error": error,
    }


# ------------------------------------------------------------------------------
def send_lane(client, lane, phase, body, timeout, records) -> None:
    """Send one lane's request and append its row to `records`."""
    started = time.monotonic()
    reply, error = None, None
    try:
        reply = client.request("POST", "/v1/chat/completions", body, timeout=timeout)
    except ProbeError as caught:
        error = str(caught)
    records.append(lane_record(lane, phase, reply, time.monotonic() - started, error))


# ------------------------------------------------------------------------------
def run_phase(client, sampler, guard, phase, bodies, timeout) -> dict:
    """Send `bodies` (lane -> request) concurrently and wait for all or the guard.

    Returns the phase row: lane rows, counter deltas and whether the swap guard
    stopped it. On a trip the unfinished lanes are left to the server and the
    after-snapshot is taken at once.
    """
    sampler.phase = phase
    before = fresh_status(client)
    records: list[dict] = []
    threads = [
        threading.Thread(
            target=send_lane,
            args=(client, lane, phase, body, timeout, records),
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
    parser.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    parser.add_argument("--warm-tokens", type=int, default=DEFAULT_WARM_TOKENS)
    parser.add_argument("--extend-tokens", type=int, default=DEFAULT_EXTEND_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
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
    if min(args.lanes, args.extend_tokens, args.max_tokens, args.swap_guard_mb) < 1:
        parser.error(
            "--lanes, --extend-tokens, --max-tokens and --swap-guard-mb must be positive"
        )
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

    Exits SWAP_ABORT_EXIT_CODE when the swap guard stopped the run.
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
        phases = run_probe(client, model, args, sampler, guard)
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
    document = {
        "model": model,
        "nonces": args.nonces,
        "phases": phases,
        "pressure": pressure_summary(sampler.samples),
        "swap_aborted": guard.tripped.is_set(),
        "swap_grown_bytes": guard.grown_bytes,
    }
    print(json.dumps(document, indent=2) if args.json else format_report(document))
    return SWAP_ABORT_EXIT_CODE if document["swap_aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
