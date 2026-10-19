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
    "kv": {"pages_total": 1024, "pages_free": 512, "unmap_last_ms": 1.5},
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
splash_kv_pages_total 1024
splash_kv_pages_free 512
splash_kv_unmap_last_ms 1.5
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


class StatusMetricsRenderingTests(unittest.TestCase):
    def test_fixture_renders_exactly_as_before(self):
        self.assertEqual(prometheus_metrics(FIXTURE_STATUS), FIXTURE_RENDERING)

    def test_repeated_renders_are_identical(self):
        self.assertEqual(
            prometheus_metrics(FIXTURE_STATUS), prometheus_metrics(FIXTURE_STATUS)
        )


if __name__ == "__main__":
    unittest.main()
