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


if __name__ == "__main__":
    unittest.main()
