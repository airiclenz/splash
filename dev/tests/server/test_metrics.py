import unittest

from server.metrics import prometheus_metrics

HOST_BYTE_METRICS = ("splash_host_headroom_bytes", "splash_host_available_bytes")
SYSTEM_PRESSURE_METRIC = "splash_system_memory_pressure"


def governor_status(**fields):
    return {"memory_governor": fields}


class HostMemoryMetricsTests(unittest.TestCase):
    def test_host_memory_fields_render_exact_lines(self):
        lines = prometheus_metrics(
            governor_status(
                host_measurement_valid=True,
                host_headroom_bytes=1073741824,
                host_available_bytes=8589934592,
                system_pressure="warning",
            )
        ).splitlines()

        self.assertIn("splash_host_headroom_bytes 1073741824", lines)
        self.assertIn("splash_host_available_bytes 8589934592", lines)
        self.assertIn('splash_system_memory_pressure{state="normal"} 0', lines)
        self.assertIn('splash_system_memory_pressure{state="warning"} 1', lines)
        self.assertIn('splash_system_memory_pressure{state="critical"} 0', lines)

    def test_missing_host_memory_fields_render_no_lines(self):
        for status in ({}, governor_status(), governor_status(limit_bytes=1)):
            with self.subTest(status=status):
                text = prometheus_metrics(status)

                for name in HOST_BYTE_METRICS + (SYSTEM_PRESSURE_METRIC,):
                    self.assertNotIn(name, text)

    def test_invalid_host_measurement_omits_host_byte_lines(self):
        text = prometheus_metrics(
            governor_status(
                host_measurement_valid=False,
                host_headroom_bytes=0,
                host_available_bytes=0,
                system_pressure="normal",
            )
        )

        for name in HOST_BYTE_METRICS:
            self.assertNotIn(name, text)
        self.assertIn('splash_system_memory_pressure{state="normal"} 1', text)

    def test_stale_fallback_status_renders_no_system_pressure(self):
        text = prometheus_metrics({"schema_version": 1, "ready": False})

        self.assertNotIn(SYSTEM_PRESSURE_METRIC, text)
        self.assertIn('splash_memory_pressure{state="normal"} 0', text)


FIXTURE_STATUS = {
    "schema_version": 5,
    "ready": True,
    "memory_pressure": "warning",
    "metal": {"healthy": True},
    "transport": {"pending": 2},
    "frontend": {"active": 1, "waiting": 0},
    "requests": {"submitted": 40, "completed": 37, "cancelled": 2, "failed": 1},
    "scheduler": {
        "queued": 3,
        "decoding": 2,
        "decode_batches_by_width": {"b1": 10, "b2": 5, "b4": 1},
    },
    "cache": {"hits": 12, "resource_replay_tokens": 4096, "reused_tokens": "n/a"},
    "kv": {"pages_free": 512},
    "state": {"entries": 4, "pinned": False},
    "memory_governor": {
        "limit_bytes": 34359738368,
        "headroom_bytes": 2147483648,
        "host_measurement_valid": True,
        "host_headroom_bytes": 1073741824,
        "host_available_bytes": 8589934592,
        "system_pressure": "normal",
    },
    "metrics": {
        "ttft_ms": {"p50": 120.25, "p95": 480.0},
        "decode_tokens_per_second": 55.5,
        "draft_acceptance_rate": float("nan"),
    },
    "response_store": {"entries": 6, "bytes": 2048},
}

# The rendering of FIXTURE_STATUS before the metric table became a module
# constant: the output must stay byte-identical.
FIXTURE_RENDERING = """\
# HELP splash_info Splash runtime metrics.
# TYPE splash_info gauge
splash_info{runtime="native"} 1
splash_memory_pressure{state="normal"} 0
splash_memory_pressure{state="warning"} 1
splash_memory_pressure{state="critical"} 0
splash_system_memory_pressure{state="normal"} 1
splash_system_memory_pressure{state="warning"} 0
splash_system_memory_pressure{state="critical"} 0
splash_thermal_state{state="nominal"} 0
splash_thermal_state{state="fair"} 0
splash_thermal_state{state="serious"} 0
splash_thermal_state{state="critical"} 0
splash_disk_writes_gated{state="no"} 0
splash_disk_writes_gated{state="budget"} 0
splash_disk_writes_gated{state="closed"} 0
splash_ready 1
splash_metal_healthy 1
splash_transport_pending 2
splash_frontend_active 1
splash_frontend_waiting 0
splash_requests_submitted_total 40
splash_requests_completed_total 37
splash_requests_cancelled_total 2
splash_requests_failed_total 1
splash_scheduler_queued 3
splash_cache_resource_replay_tokens_total 4096
splash_scheduler_decoding 2
splash_scheduler_decode_b1_total 10
splash_scheduler_decode_b2_total 5
splash_scheduler_decode_b4_total 1
splash_kv_free_allocated_pages 512
splash_state_entries 4
splash_state_pinned 0
splash_cache_hits_total 12
splash_memory_limit_bytes 34359738368
splash_memory_headroom_bytes 2147483648
splash_ttft_p50_milliseconds 120.25
splash_ttft_p95_milliseconds 480.0
splash_decode_tokens_per_second 55.5
splash_response_store_entries 6
splash_response_store_bytes 2048
splash_host_headroom_bytes 1073741824
splash_host_available_bytes 8589934592
"""


class TokenizerMetricsTests(unittest.TestCase):
    def test_decode_fallbacks_render_as_a_counter(self):
        lines = prometheus_metrics({"tokenizer": {"decode_fallbacks": 3}}).splitlines()

        self.assertIn("splash_tokenizer_decode_fallbacks_total 3", lines)

    def test_missing_tokenizer_status_renders_no_counter(self):
        self.assertNotIn(
            "splash_tokenizer_decode_fallbacks_total", prometheus_metrics({})
        )


class DraftStateResetMetricsTests(unittest.TestCase):
    def test_resets_render_as_one_counter(self):
        lines = prometheus_metrics({"draft_context": {"resets": 2}}).splitlines()
        self.assertIn("splash_draft_state_resets_total 2", lines)

    def test_missing_resets_render_no_line(self):
        self.assertNotIn("splash_draft_state_resets_total", prometheus_metrics({}))


class DecodeSuspensionMetricsTests(unittest.TestCase):
    def test_decode_suspension_counters_render_after_replay_tokens(self):
        lines = prometheus_metrics(
            {
                "cache": {
                    "resource_replay_tokens": 4096,
                    "resource_decode_suspensions": 2,
                    "resource_decode_replay_tokens": 3072,
                }
            }
        ).splitlines()

        start = lines.index("splash_cache_resource_replay_tokens_total 4096")
        self.assertEqual(
            lines[start + 1 : start + 3],
            [
                "splash_cache_resource_decode_suspensions_total 2",
                "splash_cache_resource_decode_replay_tokens_total 3072",
            ],
        )

    def test_resume_counters_render_after_decode_replay_tokens(self):
        lines = prometheus_metrics(
            {
                "cache": {
                    "resource_decode_replay_tokens": 3072,
                    "resource_resume_restored_tokens": 8192,
                    "resource_resume_state_evictions": 1,
                }
            }
        ).splitlines()

        start = lines.index("splash_cache_resource_decode_replay_tokens_total 3072")
        self.assertEqual(
            lines[start + 1 : start + 3],
            [
                "splash_cache_resource_resume_restored_tokens_total 8192",
                "splash_cache_resource_resume_state_evictions_total 1",
            ],
        )

    def test_queued_prefix_evictions_render_after_resume_state_evictions(self):
        lines = prometheus_metrics(
            {
                "cache": {
                    "resource_resume_state_evictions": 1,
                    "resource_queued_prefix_evictions": 2,
                }
            }
        ).splitlines()

        start = lines.index("splash_cache_resource_resume_state_evictions_total 1")
        self.assertEqual(
            lines[start + 1],
            "splash_cache_resource_queued_prefix_evictions_total 2",
        )

    def test_queued_depth_loss_renders_after_queued_prefix_evictions(self):
        lines = prometheus_metrics(
            {
                "cache": {
                    "resource_queued_prefix_evictions": 2,
                    "resource_queued_depth_losses": 1,
                    "resource_queued_depth_lost_tokens": 64,
                }
            }
        ).splitlines()

        start = lines.index("splash_cache_resource_queued_prefix_evictions_total 2")
        self.assertEqual(
            lines[start + 1 : start + 3],
            [
                "splash_cache_resource_queued_depth_losses_total 1",
                "splash_cache_resource_queued_depth_lost_tokens_total 64",
            ],
        )

    def test_prefill_split_renders_after_prefill_rate(self):
        lines = prometheus_metrics(
            {
                "metrics": {
                    "prefill_tokens_per_second": 2500,
                    "prefill_cold_tokens": 3000,
                    "prefill_cold_wall_ms": 1500,
                    "prefill_cold_tokens_per_second": 2000,
                    "prefill_cached_tokens": 1096,
                    "prefill_cached_wall_ms": 274,
                    "prefill_cached_tokens_per_second": 4000,
                }
            }
        ).splitlines()

        start = lines.index("splash_prefill_tokens_per_second 2500")
        self.assertEqual(
            lines[start + 1 : start + 7],
            [
                "splash_prefill_cold_tokens_total 3000",
                "splash_prefill_cold_wall_milliseconds_total 1500",
                "splash_prefill_cold_tokens_per_second 2000",
                "splash_prefill_cached_tokens_total 1096",
                "splash_prefill_cached_wall_milliseconds_total 274",
                "splash_prefill_cached_tokens_per_second 4000",
            ],
        )

    def test_missing_decode_suspension_counters_render_no_lines(self):
        text = prometheus_metrics({"cache": {"resource_replay_tokens": 1}})

        self.assertNotIn("splash_cache_resource_decode_", text)


class CheckpointFailureCauseMetricsTests(unittest.TestCase):
    def test_failure_causes_render_as_one_labelled_counter(self):
        causes = {
            "retire_blocked": 2,
            "snapshot_denied": 3,
        }
        lines = prometheus_metrics(
            {"cache": {"checkpoint_publication_failure_causes": causes}}
        ).splitlines()

        name = "splash_cache_checkpoint_publication_failures_total"
        start = lines.index(f"# TYPE {name} counter")
        self.assertTrue(lines[start - 1].startswith(f"# HELP {name} "))
        self.assertEqual(
            lines[start + 1 : start + 3],
            [f'{name}{{cause="{cause}"}} {count}' for cause, count in causes.items()],
        )

    def test_missing_failure_causes_render_no_lines(self):
        for status in ({}, {"cache": {"checkpoint_publication_failures": 2}}):
            with self.subTest(status=status):
                self.assertNotIn(
                    "splash_cache_checkpoint_publication_failures_total",
                    prometheus_metrics(status),
                )


STATE_EVICTIONS_METRIC = "splash_state_evictions_total"
STATE_EVICTIONS_BY_CAUSE_METRIC = "splash_state_evictions_by_cause_total"


class StateEvictionCauseMetricsTests(unittest.TestCase):
    def test_eviction_causes_render_beside_the_unlabelled_total(self):
        causes = {
            "pressure_pass": 1,
            "growth": 2,
            "paused_backing": 3,
            "snapshot": 4,
            "housekeeping": 6,
        }
        lines = prometheus_metrics(
            {"state": {"evictions": 16, "eviction_causes": causes}}
        ).splitlines()

        self.assertIn(f"{STATE_EVICTIONS_METRIC} 16", lines)
        name = STATE_EVICTIONS_BY_CAUSE_METRIC
        start = lines.index(f"# TYPE {name} counter")
        self.assertTrue(lines[start - 1].startswith(f"# HELP {name} "))
        self.assertEqual(
            [line for line in lines if line.startswith(f"{name}{{")],
            [f'{name}{{cause="{cause}"}} {count}' for cause, count in causes.items()],
        )
        self.assertEqual(
            lines[start + 1 : start + 1 + len(causes)],
            [f'{name}{{cause="{cause}"}} {count}' for cause, count in causes.items()],
        )

    def test_missing_eviction_causes_render_no_lines(self):
        for status in ({}, {"state": {"evictions": 2}}):
            with self.subTest(status=status):
                text = prometheus_metrics(status)

                self.assertNotIn(STATE_EVICTIONS_BY_CAUSE_METRIC, text)


DENIED_TOTAL_METRIC = "splash_memory_denied_reservations_total"
DENIED_BY_CAUSE_METRIC = "splash_memory_denied_reservations_by_cause_total"


class DeniedReservationCauseMetricsTests(unittest.TestCase):
    def test_denial_causes_render_beside_the_unlabelled_total(self):
        lines = prometheus_metrics(
            governor_status(
                denied_reservations=5,
                denied_host_reservations=3,
                denied_engine_reservations=2,
            )
        ).splitlines()

        self.assertIn(f"{DENIED_TOTAL_METRIC} 5", lines)
        start = lines.index(f"# TYPE {DENIED_BY_CAUSE_METRIC} counter")
        self.assertTrue(
            lines[start - 1].startswith(f"# HELP {DENIED_BY_CAUSE_METRIC} ")
        )
        self.assertEqual(
            lines[start + 1 : start + 3],
            [
                f'{DENIED_BY_CAUSE_METRIC}{{cause="host"}} 3',
                f'{DENIED_BY_CAUSE_METRIC}{{cause="engine"}} 2',
            ],
        )

    def test_unlabelled_total_keeps_its_own_family(self):
        lines = prometheus_metrics(
            governor_status(
                denied_reservations=5,
                denied_host_reservations=3,
                denied_engine_reservations=2,
            )
        ).splitlines()

        total_lines = [line for line in lines if DENIED_TOTAL_METRIC in line]
        by_cause = [line for line in lines if DENIED_BY_CAUSE_METRIC in line]
        self.assertEqual(total_lines, [f"{DENIED_TOTAL_METRIC} 5"])
        self.assertEqual(len(by_cause), 4)
        self.assertTrue(by_cause[0].startswith("# HELP "))
        self.assertTrue(by_cause[1].startswith("# TYPE "))

    def test_status_without_causes_renders_no_cause_lines(self):
        text = prometheus_metrics(governor_status(denied_reservations=5))

        self.assertNotIn(DENIED_BY_CAUSE_METRIC, text)
        self.assertIn(f"{DENIED_TOTAL_METRIC} 5", text.splitlines())


ACCEPTANCE_RATIO_METRIC = "splash_draft_acceptance_ratio"


def speculation_status(is_speculative):
    return {
        "metrics": {
            "speculative_decoding": is_speculative,
            "draft_acceptance_rate": 0.75,
        }
    }


class SpeculationMetricsTests(unittest.TestCase):
    def test_acceptance_ratio_is_omitted_when_speculation_is_off(self):
        text = prometheus_metrics(speculation_status(False))

        self.assertNotIn(ACCEPTANCE_RATIO_METRIC, text)

    def test_acceptance_ratio_renders_when_speculation_is_on(self):
        lines = prometheus_metrics(speculation_status(True)).splitlines()

        self.assertIn(f"{ACCEPTANCE_RATIO_METRIC} 0.75", lines)

    def test_acceptance_ratio_renders_when_status_predates_the_field(self):
        lines = prometheus_metrics(
            {"metrics": {"draft_acceptance_rate": 0.75}}
        ).splitlines()

        self.assertIn(f"{ACCEPTANCE_RATIO_METRIC} 0.75", lines)


class StatusMetricsRenderingTests(unittest.TestCase):
    def test_fixture_renders_exactly_as_before(self):
        self.assertEqual(prometheus_metrics(FIXTURE_STATUS), FIXTURE_RENDERING)

    def test_repeated_renders_are_identical(self):
        self.assertEqual(
            prometheus_metrics(FIXTURE_STATUS), prometheus_metrics(FIXTURE_STATUS)
        )


BUILD_STATUS = {
    "build": {"version": "source", "commit": "ebbc670-dirty", "build_id": "src-1"}
}


class BuildInfoMetricsTests(unittest.TestCase):
    def test_build_identity_renders_as_an_info_gauge_after_splash_info(self):
        lines = prometheus_metrics(BUILD_STATUS).splitlines()

        self.assertEqual(
            lines[3:6],
            [
                "# HELP splash_build_info The build this server runs.",
                "# TYPE splash_build_info gauge",
                'splash_build_info{version="source",commit="ebbc670-dirty",'
                'build_id="src-1"} 1',
            ],
        )

    def test_label_values_are_escaped(self):
        status = {"build": {"version": 'a"b\\c\nd', "commit": "x", "build_id": "y"}}

        self.assertIn(
            'splash_build_info{version="a\\"b\\\\c\\nd",commit="x",build_id="y"} 1',
            prometheus_metrics(status),
        )

    def test_missing_or_partial_build_renders_no_lines(self):
        for status in ({}, {"build": None}, {"build": {"version": "1.0"}}):
            with self.subTest(status=status):
                self.assertNotIn("splash_build_info", prometheus_metrics(status))


if __name__ == "__main__":
    unittest.main()
