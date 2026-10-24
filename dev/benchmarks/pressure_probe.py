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

``--ballast-mb N`` holds N MB of incompressible host memory in a child process
from the burst phase's start until it ends (with ``--agent-turns``, from the
second turn's start until the last turn ends), so the server serves under host
pressure. ``--ballast-mb auto`` engages only once the server holds a request in
prefill or decode, then grows the ballast a step at a time, each sized from a
fresh ``vm_stat`` host reading taken as the server's ``memory_governor`` does,
to keep the host just past the server's warning margin. A step never exceeds a
quarter of the swap guard's cap nor the swap room left below half of it, and
swapouts are re-read before each step: growth stops at half the cap and a step
is shed while swapouts still rise past three quarters of it. The child wires its pages
(mlock), or touches the ones the host would not wire, so they stay resident. It
exits when its pipe to the probe closes, so a killed probe leaves no ballast
behind.

``--require-warning-suspension`` and ``--require-waiting-memory`` assert that
the pressure paths ran: the probe exits WARNING_SUSPENSION_MISSING_EXIT_CODE or
WAITING_MEMORY_MISSING_EXIT_CODE when the samples show none.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import multiprocessing
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
WARNING_SUSPENSION_MISSING_EXIT_CODE = 5
WAITING_MEMORY_MISSING_EXIT_CODE = 6
SERVER_WARNING = "warning"
AGENT_LANE = 0
# The chat reply path of the server-side TTFT (prefill wall, no network).
TTFT_PATH = ("metrics", "request_latency", "start_to_first_token_ms")
# Arrival to engine admission: the time a lane waited unadmitted, which the
# engine TTFT above leaves out.
QUEUE_PATH = ("metrics", "request_latency", "queue_to_start_ms")
BYTES_PER_MB = 2**20
BALLAST_AUTO = "auto"
# The agent turn the ballast is held from: the first turn warms the conversation.
BALLAST_AGENT_TURN = 1
# os.urandom chunk the ballast is filled with: bounds the transient copy. Also
# the smallest target `auto` still grows toward.
BALLAST_CHUNK_BYTES = 64 * BYTES_PER_MB
# The most `auto` grows by between two host readings.
BALLAST_STEP_BYTES = 256 * BYTES_PER_MB
# Share of the swap guard's cap one `auto` step holds at most: a step that
# swapped out in full, seen a tick late, still leaves swapouts at most at
# BALLAST_SWAP_SHED_SHARE of the cap, short of the trip.
BALLAST_STEP_GUARD_SHARE = 0.25
# Share of the swap guard's cap at which `auto` stops growing the ballast.
BALLAST_SWAP_GROW_SHARE = 0.5
# Share past which `auto` sheds its newest step while swapouts still rise.
BALLAST_SWAP_SHED_SHARE = 0.75
BALLAST_GROW = "grow"
BALLAST_SHED = "shed"
# How often the ballast child re-reads pages the host would not wire.
BALLAST_TOUCH_SECONDS = 1.0
BALLAST_POLL_SECONDS = 0.5
BALLAST_EXIT_SECONDS = 5.0
# MemoryPlan.hpp EngineMemoryPolicy::minimumHostWarningMarginBytes.
MINIMUM_HOST_WARNING_MARGIN_BYTES = 256 * BYTES_PER_MB
# How far `auto` pushes host headroom below the warning margin.
BALLAST_OVERSHOOT_BYTES = 256 * BYTES_PER_MB
# `vm_stat` labels of the page categories estimateHostAvailableMemory reads.
VM_STAT_USED_PAGES = (
    "Pages active",
    "Pages inactive",
    "Pages speculative",
    "Pages wired down",
    "Pages occupied by compressor",
)
VM_STAT_RECLAIMABLE_PAGES = ("File-backed pages", "Pages purgeable")

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
    ("resume_restored_tokens", ("cache", "resource_resume_restored_tokens")),
    ("resume_state_evictions", ("cache", "resource_resume_state_evictions")),
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
    ("resource_suspensions", ("cache", "resource_suspensions")),
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
def parse_host_available(text: str, physical_bytes: int) -> int:
    """Return the host's available bytes from `vm_stat` output.

    Physical memory less the used pages, crediting file-backed and purgeable
    pages; mirrors estimateHostAvailableMemory
    (runtime/engine/MemoryGovernor.cpp), so it reads what the server's next
    host sample would. Keep the two in step. Raises ProbeError when a page
    category is missing.
    """
    page_bytes, _swapouts = parse_vm_stat(text)
    counts = {}
    for label in VM_STAT_USED_PAGES + VM_STAT_RECLAIMABLE_PAGES:
        found = re.search(rf"{label}:\s+(\d+)", text)
        if not found:
            raise ProbeError(f"vm_stat output carries no '{label}' count")
        counts[label] = int(found.group(1))
    used_pages = sum(counts[label] for label in VM_STAT_USED_PAGES) - sum(
        counts[label] for label in VM_STAT_RECLAIMABLE_PAGES
    )
    if used_pages < 0:
        return 0
    return max(0, physical_bytes - used_pages * page_bytes)


# ------------------------------------------------------------------------------
def read_host_available() -> int:
    """Run `vm_stat` and return the host's available bytes, read now."""
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    physical_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return parse_host_available(output.stdout, physical_bytes)


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
def host_warning_margin_bytes(reserve_bytes: int) -> int:
    """Return the host cushion's warning margin for a host reserve.

    Mirrors EngineMemoryPolicy::hostCushionForReserve
    (runtime/engine/MemoryPlan.hpp); keep the two in step.
    """
    return max(reserve_bytes // 4, MINIMUM_HOST_WARNING_MARGIN_BYTES)


# ------------------------------------------------------------------------------
def auto_ballast(snapshot: dict, host_available_bytes: int) -> dict:
    """Return how much more `--ballast-mb auto` holds, from a fresh host reading.

    The host headroom is computed as the server does (MemoryGovernor::
    hostHeadroomBytes): `host_available_bytes`, read from `vm_stat` just now,
    less the `/status` host_reserve_bytes and reserved_bytes, floored at 0.
    The `/status` host_headroom_bytes is not used: the server refreshes its
    host sample only every few tens of seconds, so a ballast sized from it
    overshoots by every step taken in between.

    N = headroom - the warning margin + BALLAST_OVERSHOOT_BYTES, floored at 0,
    so holding N more drops the headroom just past the server's warning
    margin; Ballast.step grows toward it a bounded step at a time. Raises
    ProbeError when the server exports neither reserve reading.
    """
    governor = ("memory_governor",)
    reserve = status_reading(snapshot, (*governor, "host_reserve_bytes"))
    reserved = status_reading(snapshot, (*governor, "reserved_bytes"))
    if reserve is None or reserved is None:
        raise ProbeError(
            "/status memory_governor carries no host_reserve_bytes or "
            "reserved_bytes to size --ballast-mb auto"
        )
    headroom = max(0, host_available_bytes - reserve - reserved)
    margin = host_warning_margin_bytes(reserve)
    ballast_bytes = max(0, headroom - margin + BALLAST_OVERSHOOT_BYTES)
    return {
        "mb": ballast_bytes // BYTES_PER_MB,
        "host_headroom_bytes": headroom,
        "host_reserve_bytes": reserve,
        "warning_margin_bytes": margin,
    }


# ------------------------------------------------------------------------------
def auto_step_bytes(target_bytes: int, swap_grown_bytes: int, swap_cap_bytes: int):
    """Return the page-aligned size of the next `auto` growth step; 0 holds.

    The least of the auto_ballast target, BALLAST_STEP_BYTES,
    BALLAST_STEP_GUARD_SHARE of the swap guard's cap and the swap room left
    below BALLAST_SWAP_GROW_SHARE of it. A step whose bytes all push other
    pages to swap therefore lifts swapouts no further than the share where
    growth stops, so `swap_limited` engages before the guard trips.
    """
    swap_room = int(swap_cap_bytes * BALLAST_SWAP_GROW_SHARE) - swap_grown_bytes
    step_bytes = min(
        target_bytes,
        BALLAST_STEP_BYTES,
        int(swap_cap_bytes * BALLAST_STEP_GUARD_SHARE),
        swap_room,
    )
    return max(0, step_bytes - step_bytes % mmap.PAGESIZE)


# ------------------------------------------------------------------------------
def lanes_resident(snapshot: dict) -> bool:
    """Return whether the server holds a request in prefill or decode.

    `--ballast-mb auto` engages only then, so it is never sized from an idle
    server's headroom. Raises ProbeError when the server lacks either count.
    """
    counts = [
        status_reading(snapshot, ("scheduler", key))
        for key in ("prefilling", "decoding")
    ]
    if None in counts:
        raise ProbeError(
            "/status scheduler carries no prefilling or decoding count to "
            "engage --ballast-mb auto"
        )
    return sum(counts) > 0


# ------------------------------------------------------------------------------
def fill_incompressible(region) -> None:
    """Overwrite a writable buffer with random bytes, chunk by chunk.

    Random pages defeat the macOS memory compressor, which would absorb zero
    or repeated pages without raising host pressure.
    """
    size = len(region)
    for offset in range(0, size, BALLAST_CHUNK_BYTES):
        length = min(BALLAST_CHUNK_BYTES, size - offset)
        region[offset : offset + length] = os.urandom(length)


# ------------------------------------------------------------------------------
def wire_pages(region) -> bool:
    """Wire a writable buffer's pages into RAM with mlock(2).

    Wired pages are neither compressed nor swapped out, so an idle ballast
    keeps pressing on the host. Returns False, after saying why on stderr,
    when the host refuses (the wire limit, or no mlock); the caller then keeps
    the pages warm by touching them.
    """
    try:
        anchor = ctypes.c_char.from_buffer(region)
        address = ctypes.addressof(anchor)
        # Drop the buffer export at once, or the region could not be closed.
        del anchor
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(len(region)))
        reason = os.strerror(ctypes.get_errno())
    except (AttributeError, OSError, TypeError, ValueError) as error:
        result, reason = -1, str(error)
    if result != 0:
        print(f"pressure_probe: ballast pages not wired: {reason}", file=sys.stderr)
    return result == 0


# ------------------------------------------------------------------------------
def touch_pages(region) -> None:
    """Read one byte of every page, so the host keeps unwired pages resident."""
    region[:: mmap.PAGESIZE]


# ------------------------------------------------------------------------------
def held_bytes(regions: list) -> tuple[int, int]:
    """Return (bytes held, bytes wired) of the ballast child's (region, wired) list."""
    held = sum(len(region) for region, _ in regions)
    wired = sum(len(region) for region, is_wired in regions if is_wired)
    return held, wired


# ------------------------------------------------------------------------------
def apply_ballast_command(regions: list, command: str, byte_count: int) -> None:
    """Grow the ballast by `byte_count` filled, wired bytes, or shed its newest region."""
    if command == BALLAST_GROW:
        # Anonymous memory is mapped lazily, so the fill touches each page once.
        region = mmap.mmap(-1, byte_count)
        fill_incompressible(region)
        regions.append((region, wire_pages(region)))
    elif regions:
        regions.pop()[0].close()


# ------------------------------------------------------------------------------
def hold_ballast(control) -> None:
    """Hold ballast regions as the probe commands, until its pipe closes.

    The ballast child's body. `control` is its end of a pipe to the probe,
    carrying (BALLAST_GROW, byte count) or (BALLAST_SHED, 0) commands; each is
    answered with (bytes held, bytes wired). Between commands the child touches
    the pages it could not wire every BALLAST_TOUCH_SECONDS. It returns once
    the probe closes its end or dies.
    """
    regions: list = []
    try:
        while True:
            if not control.poll(BALLAST_TOUCH_SECONDS):
                for region, is_wired in regions:
                    if not is_wired:
                        touch_pages(region)
                continue
            command, byte_count = control.recv()
            apply_ballast_command(regions, command, byte_count)
            control.send(held_bytes(regions))
    except (EOFError, OSError):
        # The probe closed its end or exited: EOF is the release signal.
        pass
    finally:
        for region, _ in regions:
            region.close()


class BallastChild:
    """One spawned process holding the ballast until its pipe closes."""

    # --------------------------------------------------------------------------
    def __init__(self):
        """Spawn the child; it holds nothing until the first `resize`."""
        # Spawn, not fork: the probe runs sampler and guard threads.
        context = multiprocessing.get_context("spawn")
        self.control, child_end = context.Pipe()
        self.process = context.Process(
            target=hold_ballast, args=(child_end,), daemon=True
        )
        self.process.start()
        child_end.close()

    # --------------------------------------------------------------------------
    def resize(self, command: str, byte_count: int, stop: threading.Event):
        """Send one command; return (bytes held, bytes wired) once it applied.

        Returns None when `stop` is set or the child ends first.
        """
        try:
            self.control.send((command, byte_count))
            while not self.control.poll(BALLAST_POLL_SECONDS):
                if stop.is_set() or not self.process.is_alive():
                    return None
            return self.control.recv()
        except (EOFError, OSError):
            return None

    # --------------------------------------------------------------------------
    def close(self) -> None:
        """Close the pipe so the child exits; terminate it if it lingers."""
        self.control.close()
        self.process.join(BALLAST_EXIT_SECONDS)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


class Ballast:
    """Holds `--ballast-mb` of host memory between `arm` and `release`.

    Runs on the probe's main thread only. `arm` marks the phase the ballast is
    held from; a fixed size is filled there, before the phase's requests. An
    `auto` ballast is driven by `tend`, which run_phase calls about every
    SWAP_POLL_SECONDS: it engages once the server holds a request in prefill or
    decode, then each tick re-reads the host with `read_host_available()` and
    grows the ballast toward the auto_ballast target by at most one
    auto_step_bytes step, checking swapouts before every step and stopping
    short of the swap guard. `start_child()` returns an object offering
    `resize(command, byte_count, stop)` and `close()`. `arm` and `release` are
    idempotent, so phase edges, guard trips and `main` may all release.
    """

    # --------------------------------------------------------------------------
    def __init__(self, request, client, start_child, read_host_available):
        """`request` is None (no ballast), BALLAST_AUTO, or a size in MB.

        `read_host_available()` returns the host's available bytes, read now.
        """
        self.request = request
        self.client = client
        self.start_child = start_child
        self.read_host_available = read_host_available
        self.child = None
        self.record = None
        self.armed_at = 0.0
        self.held_bytes = 0
        self.swap_seen_bytes = 0
        self.is_engaged = False
        self.has_failed = False

    # --------------------------------------------------------------------------
    def arm(self, phase: str, guard: SwapGuard) -> None:
        """Hold the ballast from the start of `phase` on; once only.

        A fixed size is filled now and this returns once the child holds it,
        the guard tripped or the child failed; the report's `held` says which.
        Starts nothing once the guard has tripped.
        """
        if self.request is None or self.record is not None:
            return
        if guard.tripped.is_set():
            return
        self.record = {
            "requested": self.request,
            "phase": phase,
            "held": False,
            "mb": 0,
            "wired_mb": 0,
        }
        self.armed_at = time.monotonic()
        if self.request == BALLAST_AUTO:
            self.record["swap_limited"] = False
            return
        self.resize(BALLAST_GROW, self.request * BYTES_PER_MB, guard)

    # --------------------------------------------------------------------------
    def tend(self, guard: SwapGuard) -> None:
        """Take one `auto` step from a fresh `/status` and host reading.

        A no-op for a fixed or no ballast. Before engaging, waits for
        lanes_resident and then records the engage-time readings. A failed
        `/status` read skips the step.
        """
        if self.request != BALLAST_AUTO or self.record is None or self.has_failed:
            return
        if guard.tripped.is_set():
            return
        try:
            status = fresh_status(self.client)
        except ProbeError as error:
            print(f"pressure_probe: ballast step skipped: {error}", file=sys.stderr)
            return
        if not self.is_engaged and not lanes_resident(status):
            return
        sizing = auto_ballast(status, self.read_host_available())
        if not self.is_engaged:
            self.is_engaged = True
            engaged = {key: value for key, value in sizing.items() if key != "mb"}
            engaged["engage_s"] = round(time.monotonic() - self.armed_at, 1)
            self.record.update(engaged)
            self.swap_seen_bytes = guard.grown_bytes
        self.step(sizing["mb"] * BYTES_PER_MB, guard)

    # --------------------------------------------------------------------------
    def step(self, target_bytes: int, guard: SwapGuard) -> None:
        """Shed, hold or grow the `auto` ballast by one step.

        Takes a fresh swap reading first. Sheds the newest region while
        swapouts still rise past BALLAST_SWAP_SHED_SHARE of the guard's cap;
        grows no more once they reach BALLAST_SWAP_GROW_SHARE; otherwise grows
        by auto_step_bytes toward `target_bytes`, only while that target
        reaches BALLAST_CHUNK_BYTES.
        """
        if guard.check():
            return
        grown = guard.grown_bytes
        is_swap_rising = grown > self.swap_seen_bytes
        self.swap_seen_bytes = grown
        if grown >= guard.cap_bytes * BALLAST_SWAP_SHED_SHARE:
            self.record["swap_limited"] = True
            if is_swap_rising and self.held_bytes > 0:
                self.resize(BALLAST_SHED, 0, guard)
            return
        if grown >= guard.cap_bytes * BALLAST_SWAP_GROW_SHARE:
            self.record["swap_limited"] = True
            return
        if target_bytes < BALLAST_CHUNK_BYTES:
            return
        step_bytes = auto_step_bytes(target_bytes, grown, guard.cap_bytes)
        if step_bytes > 0:
            self.resize(BALLAST_GROW, step_bytes, guard)

    # --------------------------------------------------------------------------
    def resize(self, command: str, byte_count: int, guard: SwapGuard) -> None:
        """Send one command to the child, starting it first, and record the result.

        `mb` and `wired_mb` keep the peak held; a child that ends or a guard
        trip stops the ballast from changing again.
        """
        if self.child is None:
            self.child = self.start_child()
        reply = self.child.resize(command, byte_count, guard.tripped)
        if reply is None:
            self.has_failed = True
            print("pressure_probe: ballast not held", file=sys.stderr)
            return
        self.held_bytes, wired_bytes = reply
        record = self.record
        record["held"] = record["held"] or self.held_bytes > 0
        record["mb"] = max(record["mb"], self.held_bytes // BYTES_PER_MB)
        record["wired_mb"] = max(record["wired_mb"], wired_bytes // BYTES_PER_MB)

    # --------------------------------------------------------------------------
    def release(self) -> None:
        """End the ballast child, if one runs."""
        child, self.child = self.child, None
        if child is not None:
            child.close()


# ------------------------------------------------------------------------------
def has_warning_suspension(samples: list[dict]) -> bool:
    """Return whether resource suspensions rose between two warning samples.

    Only consecutive usable samples (those with a server pressure) count, and
    both must read `warning`.
    """
    usable = [sample for sample in samples if sample.get("server_pressure")]
    for current, following in zip(usable, usable[1:]):
        pressures = (current["server_pressure"], following["server_pressure"])
        if pressures != (SERVER_WARNING, SERVER_WARNING):
            continue
        before = current.get("resource_suspensions")
        after = following.get("resource_suspensions")
        if before is not None and after is not None and after > before:
            return True
    return False


# ------------------------------------------------------------------------------
def has_waiting_memory(samples: list[dict]) -> bool:
    """Return whether some sample shows a request waiting for memory."""
    return any((sample.get("waiting_memory") or 0) > 0 for sample in samples)


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
def run_phase(
    client, sampler, guard, phase, bodies, timeout, replies=None, ballast=None
) -> dict:
    """Send `bodies` (lane -> request) concurrently and wait for all or the guard.

    Returns the phase row: lane rows, counter deltas and whether the swap guard
    stopped it. On a trip the unfinished lanes are left to the server and the
    after-snapshot is taken at once. `replies`, when given, receives each
    finished lane's decoded reply. `ballast`, when given, is tended about every
    SWAP_POLL_SECONDS while the lanes run.
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
        if ballast is not None:
            ballast.tend(guard)
    after = fresh_status(client)
    return {
        "phase": phase,
        "lanes": sorted(records, key=lambda record: record["lane"]),
        "swap_aborted": guard.tripped.is_set(),
        **counter_delta(before, after),
    }


# ------------------------------------------------------------------------------
def run_probe(client, model, args, sampler, guard, ballast) -> list[dict]:
    """Warm each lane alone (unless `args.nonces` names warmed ones), then burst.

    `ballast` is held for the burst and tended while it runs. Stops once the
    swap guard has tripped or a warm-up fails.
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
    bodies = {
        lane: chat_body(
            model,
            lane_prompt(nonce, lane_words(nonce, burst_words), instruction),
            args.max_tokens,
        )
        for lane, nonce in enumerate(args.nonces)
    }
    try:
        ballast.arm("burst", guard)
        if guard.tripped.is_set():
            return phases
        phases.append(
            run_phase(
                client,
                sampler,
                guard,
                "burst",
                bodies,
                args.request_timeout,
                ballast=ballast,
            )
        )
    finally:
        ballast.release()
    return phases


# ------------------------------------------------------------------------------
def reply_text(reply) -> str | None:
    """Return the assistant content of a chat reply, or None when it has none."""
    choices = (reply or {}).get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else None


# ------------------------------------------------------------------------------
def run_agent(client, model, args, sampler, guard, ballast) -> list[dict]:
    """Drive one conversation for `args.agent_turns` back-to-back turns.

    Turn 0 is the warm-up-sized prompt; each later turn resends the whole
    conversation, the previous reply and `args.tool_tokens` new filler words,
    as an agent does after a tool call. `ballast` is held from the start of
    turn BALLAST_AGENT_TURN until the last turn ends. Stops once the swap guard
    has tripped, a turn fails or a reply carries no content to resend.
    """
    nonce = args.nonces[0]
    instruction = f"Write the integers from 1 to {args.max_tokens * 2}, one per line."
    words = lane_words(nonce, max(1, args.warm_tokens - PROMPT_FRAME_TOKENS))
    messages = [{"role": "user", "content": lane_prompt(nonce, words, instruction)}]
    phases = []
    try:
        for turn in range(args.agent_turns):
            if turn == BALLAST_AGENT_TURN:
                ballast.arm(f"turn{turn}", guard)
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
                ballast,
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
    finally:
        ballast.release()
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
    for key, value in (document["ballast"] or {}).items():
        cell = value if isinstance(value, str) else format_value(value)
        lines.append(f"{'ballast_' + key:<{width}}{cell:>12}")
    if document["swap_aborted"]:
        grown = document["swap_grown_bytes"] / BYTES_PER_MB
        lines.append(f"\nswap guard tripped: swapouts grew {grown:.0f} MB")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
def ballast_request(text: str):
    """Parse `--ballast-mb`: BALLAST_AUTO or a positive size in MB."""
    if text == BALLAST_AUTO:
        return BALLAST_AUTO
    try:
        megabytes = int(text)
    except ValueError:
        megabytes = 0
    if megabytes < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive size in MB or {BALLAST_AUTO}, got {text!r}"
        )
    return megabytes


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
        "--ballast-mb",
        type=ballast_request,
        help="hold this much incompressible host memory during the burst (with "
        f"--agent-turns, from the second turn on); {BALLAST_AUTO} grows it in "
        "steps once a request is in prefill or decode, from fresh vm_stat "
        "readings, to keep the host past the server's warning margin, stopping "
        "short of the swap guard",
    )
    parser.add_argument(
        "--require-warning-suspension",
        action="store_true",
        help=f"exit {WARNING_SUSPENSION_MISSING_EXIT_CODE} unless a resource "
        "suspension happened while the server reported warning",
    )
    parser.add_argument(
        "--require-waiting-memory",
        action="store_true",
        help=f"exit {WAITING_MEMORY_MISSING_EXIT_CODE} unless a sample shows a "
        "request waiting for memory",
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
def exit_code(args, document: dict, samples: list[dict]) -> int:
    """Return the probe's exit code; the swap guard is checked first.

    SWAP_ABORT_EXIT_CODE when the swap guard stopped the run, else
    AGENT_COLD_TURN_EXIT_CODE when an agent turn after the first reused 0
    tokens, else WARNING_SUSPENSION_MISSING_EXIT_CODE or
    WAITING_MEMORY_MISSING_EXIT_CODE when a require-flag's path never ran.
    """
    if document["swap_aborted"]:
        return SWAP_ABORT_EXIT_CODE
    if args.agent_turns is not None and has_cold_turn(document["phases"]):
        return AGENT_COLD_TURN_EXIT_CODE
    if args.require_warning_suspension and not has_warning_suspension(samples):
        return WARNING_SUSPENSION_MISSING_EXIT_CODE
    if args.require_waiting_memory and not has_waiting_memory(samples):
        return WAITING_MEMORY_MISSING_EXIT_CODE
    return 0


# ------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Run the probe and print the report; return the process exit code.

    See exit_code; 1 when the server or `vm_stat` could not be read. The
    ballast is released on every exit path, an interrupt included.
    """
    args = parse_args(argv)
    client = ProbeClient(args.base_url, args.api_key)
    ballast = Ballast(args.ballast_mb, client, BallastChild, read_host_available)
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
        phases = run(client, model, args, sampler, guard, ballast)
    except (ProbeError, subprocess.CalledProcessError) as error:
        print(f"pressure_probe: {error}", file=sys.stderr)
        return 1
    finally:
        ballast.release()
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
        "ballast": ballast.record,
    }
    print(json.dumps(document, indent=2) if args.json else format_report(document))
    return exit_code(args, document, sampler.samples)


if __name__ == "__main__":
    raise SystemExit(main())
