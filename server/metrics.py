"""Usage, metrics and Prometheus text rendering for request results."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

from .latency import prometheus_latency


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


MEMORY_PRESSURE_STATES = ("normal", "warning", "critical")
# Status.cpp writes these as 0 when the host sample failed, so they are only
# exported while memory_governor.host_measurement_valid is true.
HOST_MEMORY_METRICS = {
    "splash_host_headroom_bytes": ("memory_governor", "host_headroom_bytes"),
    "splash_host_available_bytes": ("memory_governor", "host_available_bytes"),
}


@dataclass(frozen=True)
class LabelledCounter:
    """A counter family whose samples each come from their own status path."""

    help_text: str
    label: str
    paths: dict


# Each gauge or counter Prometheus reads from the status, by the path of
# keys to its value.
PROMETHEUS_SERIES = {
    "splash_ready": ("ready",),
    "splash_metal_healthy": ("metal", "healthy"),
    "splash_transport_pending": ("transport", "pending"),
    "splash_tokenizer_decode_fallbacks_total": ("tokenizer", "decode_fallbacks"),
    "splash_frontend_active": ("frontend", "active"),
    "splash_frontend_waiting": ("frontend", "waiting"),
    "splash_requests_submitted_total": ("requests", "submitted"),
    "splash_requests_completed_total": ("requests", "completed"),
    "splash_requests_cancelled_total": ("requests", "cancelled"),
    "splash_requests_failed_total": ("requests", "failed"),
    "splash_scheduler_queued": ("scheduler", "queued"),
    "splash_scheduler_waiting_resources": ("scheduler", "waiting_resources"),
    "splash_scheduler_waiting_prefix": ("scheduler", "waiting_prefix"),
    "splash_cache_resource_suspensions_total": ("cache", "resource_suspensions"),
    "splash_cache_priority_suspensions_total": ("cache", "priority_suspensions"),
    "splash_cache_resource_resumptions_total": ("cache", "resource_resumptions"),
    "splash_cache_resource_replay_tokens_total": (
        "cache",
        "resource_replay_tokens",
    ),
    "splash_cache_resource_decode_suspensions_total": (
        "cache",
        "resource_decode_suspensions",
    ),
    "splash_cache_resource_decode_replay_tokens_total": (
        "cache",
        "resource_decode_replay_tokens",
    ),
    "splash_cache_resource_resume_restored_tokens_total": (
        "cache",
        "resource_resume_restored_tokens",
    ),
    "splash_cache_resource_resume_state_evictions_total": (
        "cache",
        "resource_resume_state_evictions",
    ),
    "splash_cache_resource_queued_prefix_evictions_total": (
        "cache",
        "resource_queued_prefix_evictions",
    ),
    "splash_cache_resource_queued_depth_losses_total": (
        "cache",
        "resource_queued_depth_losses",
    ),
    "splash_cache_resource_queued_depth_lost_tokens_total": (
        "cache",
        "resource_queued_depth_lost_tokens",
    ),
    "splash_scheduler_prefilling": ("scheduler", "prefilling"),
    "splash_scheduler_decoding": ("scheduler", "decoding"),
    "splash_scheduler_waiting_mask": ("scheduler", "waiting_mask"),
    "splash_scheduler_prefill_batches_total": ("scheduler", "prefill_batches"),
    "splash_scheduler_prefill_rows_total": ("scheduler", "prefill_rows"),
    "splash_scheduler_decode_batches_total": ("scheduler", "decode_batches"),
    "splash_scheduler_decode_b1_total": (
        "scheduler",
        "decode_batches_by_width",
        "b1",
    ),
    "splash_scheduler_decode_b2_total": (
        "scheduler",
        "decode_batches_by_width",
        "b2",
    ),
    "splash_scheduler_decode_b3_total": (
        "scheduler",
        "decode_batches_by_width",
        "b3",
    ),
    "splash_scheduler_decode_b4_total": (
        "scheduler",
        "decode_batches_by_width",
        "b4",
    ),
    "splash_kv_pages_allocated": ("kv", "pages_allocated"),
    "splash_kv_evictions_total": ("kv", "evictions"),
    "splash_kv_pages_active": ("kv", "pages_active"),
    "splash_kv_pages_cache": ("kv", "pages_cache"),
    "splash_kv_free_allocated_pages": ("kv", "pages_free"),
    "splash_kv_allocated_bytes": ("kv", "allocated_bytes"),
    "splash_kv_reclaimable_bytes": ("kv", "reclaimable_bytes"),
    "splash_kv_extent_allocations_total": ("kv", "extent_allocations"),
    "splash_kv_extent_releases_total": ("kv", "extent_releases"),
    "splash_kv_extent_allocate_max_milliseconds": (
        "kv",
        "extent_allocate_max_ms",
    ),
    "splash_kv_extent_release_max_milliseconds": (
        "kv",
        "extent_release_max_ms",
    ),
    "splash_kv_extent_compactions_total": ("kv", "extent_compactions"),
    "splash_kv_pages_moved_total": ("kv", "pages_moved"),
    "splash_kv_extent_compact_max_milliseconds": (
        "kv",
        "extent_compact_max_ms",
    ),
    "splash_state_entries": ("state", "entries"),
    "splash_state_pinned": ("state", "pinned"),
    "splash_state_in_use": ("state", "in_use"),
    "splash_state_in_use_evictions_total": ("state", "in_use_evictions"),
    "splash_state_bytes": ("state", "bytes"),
    "splash_state_active_lanes": ("state", "active_lanes"),
    "splash_state_publications_total": ("state", "publications"),
    "splash_state_evictions_total": ("state", "evictions"),
    "splash_disk_capacity_bytes": ("disk", "capacity_bytes"),
    "splash_disk_limit_bytes": ("disk", "limit_bytes"),
    "splash_disk_used_bytes": ("disk", "used_bytes"),
    "splash_disk_volume_available_bytes": ("disk", "volume_available_bytes"),
    "splash_disk_write_budget_bytes": ("disk", "write_budget_bytes"),
    "splash_disk_read_bytes_total": ("disk", "read_bytes"),
    "splash_disk_written_bytes_total": ("disk", "written_bytes"),
    "splash_cache_hits_total": ("cache", "hits"),
    "splash_cache_cold_misses_total": ("cache", "cold_misses"),
    "splash_cache_reused_tokens_total": ("cache", "reused_tokens"),
    "splash_cache_deduplicated_kv_tokens_total": ("cache", "deduplicated_kv_tokens"),
    "splash_cache_lazy_junctions_total": ("cache", "lazy_junctions"),
    "splash_target_prefill_rows_total": (
        "draft_context",
        "target_prefill_rows",
    ),
    "splash_draft_context_prompt_end_rows_total": (
        "draft_context",
        "prompt_end_rows",
    ),
    "splash_draft_context_materialization_rows_total": (
        "draft_context",
        "materialization_rows",
    ),
    "splash_draft_context_avoided_rows_total": (
        "draft_context",
        "avoided_rows",
    ),
    "splash_draft_state_restore_skipped_total": (
        "draft_context",
        "restore_skipped",
    ),
    "splash_draft_state_resets_total": ("draft_context", "resets"),
    "splash_constraint_mask_overlap_batches_total": (
        "constraint_masks",
        "overlap_batches",
    ),
    "splash_constraint_mask_overlap_requests_total": (
        "constraint_masks",
        "overlap_requests",
    ),
    "splash_constraint_mask_target_forward_gpu_milliseconds": (
        "constraint_masks",
        "last_target_forward_gpu_ms",
    ),
    "splash_constraint_mask_residual_wait_milliseconds": (
        "constraint_masks",
        "last_residual_wait_ms",
    ),
    "splash_image_encodes_total": ("images", "encodes"),
    "splash_image_embedding_reuses_total": ("images", "embedding_reuses"),
    "splash_image_embedding_evictions_total": ("images", "embedding_evictions"),
    "splash_image_arena_bytes": ("images", "arena_bytes"),
    "splash_image_cached_bytes": ("images", "cached_bytes"),
    "splash_memory_current_bytes": ("memory_actual", "current_bytes"),
    "splash_memory_peak_bytes": ("memory_actual", "peak_bytes"),
    "splash_memory_denied_reservations_total": (
        "memory_governor",
        "denied_reservations",
    ),
    "splash_memory_limit_bytes": ("memory_governor", "limit_bytes"),
    "splash_memory_headroom_bytes": ("memory_governor", "headroom_bytes"),
    "splash_admission_waiting_memory": ("admission", "waiting_memory"),
    "splash_admission_waiting_concurrency": ("admission", "waiting_concurrency"),
    "splash_admission_waiting_schedule": ("admission", "waiting_schedule"),
    "splash_admission_held_behind_refusal": ("admission", "held_behind_refusal"),
    "splash_admission_restoring": ("admission", "restoring"),
    "splash_admission_suspended": ("admission", "suspended"),
    "splash_admission_oldest_wait_milliseconds": ("admission", "oldest_wait_ms"),
    "splash_ttft_p50_milliseconds": ("metrics", "ttft_ms", "p50"),
    "splash_ttft_p95_milliseconds": ("metrics", "ttft_ms", "p95"),
    "splash_itl_p50_milliseconds": ("metrics", "itl_ms", "p50"),
    "splash_itl_p95_milliseconds": ("metrics", "itl_ms", "p95"),
    "splash_prefill_input_tokens_total": ("metrics", "prefill_input_tokens"),
    "splash_prefill_wall_milliseconds_total": ("metrics", "prefill_wall_ms"),
    "splash_prefill_tokens_per_second": (
        "metrics",
        "prefill_tokens_per_second",
    ),
    "splash_prefill_cold_tokens_total": ("metrics", "prefill_cold_tokens"),
    "splash_prefill_cold_wall_milliseconds_total": (
        "metrics",
        "prefill_cold_wall_ms",
    ),
    "splash_prefill_cold_tokens_per_second": (
        "metrics",
        "prefill_cold_tokens_per_second",
    ),
    "splash_prefill_cached_tokens_total": ("metrics", "prefill_cached_tokens"),
    "splash_prefill_cached_wall_milliseconds_total": (
        "metrics",
        "prefill_cached_wall_ms",
    ),
    "splash_prefill_cached_tokens_per_second": (
        "metrics",
        "prefill_cached_tokens_per_second",
    ),
    "splash_decode_output_tokens_total": ("metrics", "decode_output_tokens"),
    "splash_decode_wall_milliseconds_total": ("metrics", "decode_wall_ms"),
    "splash_decode_cycle_milliseconds_total": ("metrics", "decode_cycle_ms"),
    "splash_decode_tokens_per_second": (
        "metrics",
        "decode_tokens_per_second",
    ),
    "splash_drafted_tokens_total": ("metrics", "drafted_tokens"),
    "splash_accepted_draft_tokens_total": ("metrics", "accepted_draft_tokens"),
    "splash_draft_acceptance_ratio": ("metrics", "draft_acceptance_rate"),
    "splash_capacity_failures_total": ("metrics", "capacity_failures"),
    "splash_metal_failures_total": ("metrics", "metal_failures"),
    "splash_response_store_entries": ("response_store", "entries"),
    "splash_response_store_bytes": ("response_store", "bytes"),
}


# Counter families with a label, each sample from its own status path.
PROMETHEUS_FAMILIES = {
    "splash_state_evictions_by_cause_total": LabelledCounter(
        help_text=(
            "State cache evictions by cause: a pressure reclaim pass, growth "
            "after a budget or KV capacity denial, idle backing reused while "
            "growth is paused, a denied snapshot, or housekeeping (disk "
            "quota, invalidation or dropped subtrees)."
        ),
        label="cause",
        paths={
            cause: ("state", "eviction_causes", cause)
            for cause in (
                "pressure_pass",
                "growth",
                "paused_backing",
                "snapshot",
                "housekeeping",
            )
        },
    ),
    "splash_cache_checkpoint_publication_failures_total": LabelledCounter(
        help_text=(
            "Checkpoint publication failures by cause: a blocked retirement "
            "or a denied snapshot."
        ),
        label="cause",
        paths={
            cause: ("cache", "checkpoint_publication_failure_causes", cause)
            for cause in (
                "retire_blocked",
                "snapshot_denied",
            )
        },
    ),
    "splash_memory_denied_reservations_by_cause_total": LabelledCounter(
        help_text=(
            "Denied memory reservations by the bound that refused them: host "
            "pressure or the engine budget."
        ),
        label="cause",
        paths={
            "host": ("memory_governor", "denied_host_reservations"),
            "engine": ("memory_governor", "denied_engine_reservations"),
        },
    ),
}


# Metrics that only mean something while the runtime proposes draft tokens.
_SPECULATION_METRICS = frozenset(("splash_draft_acceptance_ratio",))


def _is_speculation_disabled(status):
    """True only when the status explicitly reports speculative decoding off."""
    metrics = status.get("metrics")
    return isinstance(metrics, dict) and metrics.get("speculative_decoding") is False


def _status_value(status, path):
    current = status
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool):
        return 1 if current else 0
    if is_finite_number(current):
        return current
    return None


def _labelled_counter_lines(name, family, status):
    """HELP, TYPE and one sample per present label; nothing without any."""
    samples = []
    for label_value, path in family.paths.items():
        metric_value = _status_value(status, path)
        if metric_value is not None:
            samples.append(f'{name}{{{family.label}="{label_value}"}} {metric_value}')
    if not samples:
        return []
    return [f"# HELP {name} {family.help_text}", f"# TYPE {name} counter", *samples]


# The `/status` `build` fields `splash_build_info` labels, in label order.
_BUILD_IDENTITY_FIELDS = ("version", "commit", "build_id")


def _label_value(text):
    """Escape a Prometheus label value: backslash, double quote and newline."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_info_lines(status):
    """HELP, TYPE and one `splash_build_info` sample; nothing without `build`."""
    build = status.get("build")
    if not isinstance(build, dict):
        return []
    labels = []
    for field in _BUILD_IDENTITY_FIELDS:
        value = build.get(field)
        if not isinstance(value, str):
            return []
        labels.append(f'{field}="{_label_value(value)}"')
    return [
        "# HELP splash_build_info The build this server runs.",
        "# TYPE splash_build_info gauge",
        f"splash_build_info{{{','.join(labels)}}} 1",
    ]


def prometheus_metrics(status):
    """Render low-cardinality metrics directly from the native status."""
    lines = [
        "# HELP splash_info Splash runtime metrics.",
        "# TYPE splash_info gauge",
        'splash_info{runtime="native"} 1',
    ]
    lines.extend(_build_info_lines(status))
    pressure = status.get("memory_pressure")
    for state in MEMORY_PRESSURE_STATES:
        lines.append(
            f'splash_memory_pressure{{state="{state}"}} {1 if pressure == state else 0}'
        )
    governor = status.get("memory_governor")
    if not isinstance(governor, dict):
        governor = {}
    system_pressure = governor.get("system_pressure")
    if isinstance(system_pressure, str):
        for state in MEMORY_PRESSURE_STATES:
            gauge_value = 1 if system_pressure == state else 0
            lines.append(
                f'splash_system_memory_pressure{{state="{state}"}} {gauge_value}'
            )
    thermal = status.get("thermal_state")
    for state in ("nominal", "fair", "serious", "critical"):
        lines.append(
            f'splash_thermal_state{{state="{state}"}} {1 if thermal == state else 0}'
        )
    gated = status.get("disk", {}).get("writes_gated")
    for state in ("no", "budget", "closed"):
        lines.append(
            f'splash_disk_writes_gated{{state="{state}"}} {1 if gated == state else 0}'
        )
    is_speculation_disabled = _is_speculation_disabled(status)
    for name, path in PROMETHEUS_SERIES.items():
        if is_speculation_disabled and name in _SPECULATION_METRICS:
            continue
        metric_value = _status_value(status, path)
        if metric_value is not None:
            lines.append(f"{name} {metric_value}")
    for name, family in PROMETHEUS_FAMILIES.items():
        lines.extend(_labelled_counter_lines(name, family, status))
    if governor.get("host_measurement_valid") is True:
        for name, path in HOST_MEMORY_METRICS.items():
            metric_value = _status_value(status, path)
            if metric_value is not None:
                lines.append(f"{name} {metric_value}")
    lines.extend(prometheus_latency(status.get("latency", {})))
    return "\n".join(lines) + "\n"


def usage_dict(result, job):
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": result.cache.matched_tokens},
        "completion_tokens_details": {"reasoning_tokens": job.reasoning_tokens},
    }


def timings_dict(result):
    """llama-server-compatible counts and request lifecycle timings.

    Counts are final totals. Rates exclude cached prompt tokens and the first
    emission respectively: those tokens precede the intervals being measured.
    These are elapsed request intervals, not isolated GPU execution times.
    """
    latency = result.metrics["request_latency"]
    prompt_ms = latency.get("start_to_first_token_ms", 0.0)
    prompt_rate = result.prefill_tokens * 1000.0 / prompt_ms if prompt_ms else 0.0
    timings = {
        "prompt_n": result.prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_second": prompt_rate if math.isfinite(prompt_rate) else 0.0,
        "predicted_n": result.completion_tokens,
        "predicted_ms": latency.get("first_token_to_done_ms", 0.0),
        "predicted_per_second": latency.get("stream_tokens_per_second", 0.0),
        "cache_n": result.cache.matched_tokens,
    }
    if result.drafted_tokens:
        timings.update(
            draft_n=result.drafted_tokens,
            draft_n_accepted=result.accepted_draft_tokens,
        )
    return timings


def metrics_dict(result):
    # The first native emission can contain a whole speculative block. Its
    # generation precedes TTFT, so none of those tokens belong in the interval
    # from first emission to Done. Native command throughput lives in /status.
    decode_tokens = max(0, result.completion_tokens - result.first_token_batch_tokens)
    latency = {}
    intervals = (
        ("start_to_first_token_ms", result.start_to_first_token_ms),
        ("first_token_to_done_ms", result.first_token_to_done_ms),
        ("wall_ms", result.request_wall_ms),
    )
    for name, milliseconds in intervals:
        if math.isfinite(milliseconds) and milliseconds >= 0:
            latency[name] = milliseconds
    if (
        result.completion_tokens > 0
        and "wall_ms" in latency
        and "first_token_to_done_ms" in latency
        and latency["wall_ms"] >= latency["first_token_to_done_ms"]
    ):
        latency["ttft_ms"] = latency["wall_ms"] - latency["first_token_to_done_ms"]
        if (
            "start_to_first_token_ms" in latency
            and latency["ttft_ms"] >= latency["start_to_first_token_ms"]
        ):
            latency["queue_to_start_ms"] = (
                latency["ttft_ms"] - latency["start_to_first_token_ms"]
            )
    if (
        result.first_token_batch_tokens > 0
        and decode_tokens > 0
        and latency.get("first_token_to_done_ms", 0) > 0
    ):
        rate = decode_tokens * 1000.0 / latency["first_token_to_done_ms"]
        if math.isfinite(rate):
            latency["stream_tokens_per_second"] = rate
    cache_info = result.cache
    cache = {
        "status": cache_info.status,
        "matched_tokens": cache_info.matched_tokens,
        "deduplicated_tokens": cache_info.deduplicated_tokens,
        "lane": cache_info.lane if cache_info.lane >= 0 else None,
    }
    metrics = {
        "prefill": {"tokens": result.prefill_tokens},
        "decode": {"tokens": decode_tokens},
        # Native DoneEvent intervals are request lifecycle timing, not summed
        # executor command/GPU time. Native batch throughput lives in /status.
        "request_latency": latency,
        "cache": cache,
        "engine": engine_dict(result.engine_timing),
    }
    if result.drafted_tokens:
        metrics["speculation"] = {
            "drafted_tokens": result.drafted_tokens,
            "accepted_draft_tokens": result.accepted_draft_tokens,
            "cycles": result.draft_cycles,
        }
    return metrics


# Microseconds per millisecond, for the engine record's wall fields.
MICROS_PER_MILLISECOND = 1000.0


def engine_dict(timing):
    """Render an engine timing record with its walls in milliseconds.

    A request's prefill and decode walls are the full wall of every batch it
    rode in, so concurrent requests' walls overlap rather than sum.
    """
    return {
        "prefill_start_tokens": timing.prefill_start_tokens,
        "prefilled_tokens": timing.prefilled_tokens,
        "prefill_wall_ms": timing.prefill_wall_micros / MICROS_PER_MILLISECOND,
        "decode_wall_ms": timing.decode_wall_micros / MICROS_PER_MILLISECOND,
        "decode_steps": timing.decode_steps,
        "suspensions": timing.suspensions,
        "decode_suspensions": timing.decode_suspensions,
        "replay_tokens": timing.replay_tokens,
    }


# The newest finished-request records `GET /requests` serves.
REQUEST_RECORDS_CAPACITY = 1024


class RequestRecords:
    """A bounded, thread-safe ring of finished-request records.

    Appending is O(1) and evicts the oldest record once the ring holds
    `capacity`. Records must not be mutated after they are appended: readers
    receive the same dicts, outside the lock.
    """

    def __init__(self, capacity=REQUEST_RECORDS_CAPACITY):
        """Create an empty ring holding at most `capacity` records."""
        if capacity <= 0:
            raise ValueError("request record capacity must be positive")
        self._records = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, record):
        """Add one finished request's record, evicting the oldest when full."""
        with self._lock:
            self._records.append(record)

    def finished_after(self, since_unix_ms=None):
        """Return the held records oldest first, optionally only those whose
        `finished_unix_ms` is later than `since_unix_ms`."""
        with self._lock:
            records = list(self._records)
        if since_unix_ms is None:
            return records
        return [
            record for record in records if record["finished_unix_ms"] > since_unix_ms
        ]
