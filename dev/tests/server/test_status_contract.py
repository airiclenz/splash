"""The engine's status document as the server reads it, and the routes
that report it: the probes, /status and /metrics."""

import json
import os
import time
import unittest
from pathlib import Path

from dev.tests import native_peer
from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase
from server import protocol as wire
from server import release_identity
from server import server as api
from server.metrics import PROMETHEUS_SERIES, prometheus_metrics

# A representative status document, every section of it set, which
# runtime_status_test.cpp checks the engine writes.
GOLDEN = Path(__file__).parents[1] / "engine/status_golden.json"
# The sections the server adds to the engine's document before it renders it.
SERVER_SECTIONS = {"transport", "frontend", "response_store", "tokenizer"}


class StatusContractTests(unittest.TestCase):
    def test_every_series_renders_from_the_engine_status(self):
        status = json.loads(GOLDEN.read_text())
        self.assertEqual(status["schema_version"], wire.STATUS_SCHEMA_VERSION)
        lines = prometheus_metrics(status).splitlines()
        rendered = {line.split()[0] for line in lines if not line.startswith("#")}
        self.assertEqual(
            [
                name
                for name, path in PROMETHEUS_SERIES.items()
                if path[0] not in SERVER_SECTIONS and name not in rendered
            ],
            [],
        )
        self.assertIn('splash_memory_pressure{state="normal"} 1', lines)
        self.assertIn('splash_thermal_state{state="fair"} 1', lines)
        self.assertIn('splash_thermal_state{state="nominal"} 0', lines)
        self.assertIn('splash_disk_writes_gated{state="budget"} 1', lines)
        self.assertIn('splash_disk_writes_gated{state="no"} 0', lines)
        self.assertIn("splash_disk_read_bytes_total 12345", lines)
        self.assertIn("splash_disk_written_bytes_total 67890", lines)


class StatusRouteTests(HarnessTestCase):
    def test_probes_and_status_report_a_ready_engine(self):
        harness = self.harness(FakeRuntime())
        status, _, payload = harness.request("GET", "/health")
        health = json.loads(payload)
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.assertEqual(health["model"], "test-model")
        self.assertEqual(health["build"], release_identity.UNRESOLVED.as_dict())
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ready"}))
        status, _, payload = harness.request("GET", "/status")
        snapshot = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["schema_version"], wire.STATUS_SCHEMA_VERSION)
        self.assertIs(snapshot["ready"], True)
        self.assertEqual(snapshot["memory_pressure"], "normal")
        self.assertEqual(snapshot["thermal_state"], "nominal")
        self.assertIs(snapshot["metal"]["healthy"], True)
        self.assertEqual(
            snapshot["transport"],
            {
                "ready": True,
                "recovering": False,
                "stopped": False,
                "pending": 0,
                "pending_limit": 4,
                "restarts": 0,
                "last_crash_trace": None,
                "status_stale": False,
                "status_age_ms": 0.0,
            },
        )

    def test_metrics_route_renders_the_engine_status(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        runtime.status_event = native_peer.status_event(
            2,
            thermal_state="serious",
            requests={"submitted": 7, "completed": 5},
            admission={
                "waiting_memory": 2,
                "waiting_concurrency": 1,
                "held_behind_refusal": 4,
                "restoring": 1,
                "suspended": 1,
                "oldest_wait_ms": 1250.0,
            },
            scheduler={
                "queued": 1,
                "waiting_resources": 3,
                "waiting_prefix": 2,
                "prefilling": 2,
                "decoding": 1,
                "waiting_mask": 0,
                "prefill_batches": 10,
                "prefill_rows": 2048,
                "decode_batches": 7,
                "decode_batches_by_width": {
                    "b1": 1,
                    "b2": 2,
                    "b3": 3,
                    "b4": 1,
                },
            },
            kv={
                "pages_allocated": 8,
                "pages_active": 4,
                "pages_cache": 4,
                "pages_free": 2,
                "allocated_bytes": 8192,
                "extent_allocate_max_ms": 2.5,
                "extent_release_max_ms": 0.75,
                "extent_compact_max_ms": 1.5,
            },
            state={
                "entries": 2,
                "pinned": 1,
                "in_use": 1,
                "in_use_evictions": 3,
                "bytes": 4096,
                "active_lanes": 2,
                "publications": 3,
                "evictions": 1,
            },
            cache={
                "hits": 7,
                "cold_misses": 4,
                "reused_tokens": 1024,
                "lazy_junctions": 2,
                "priority_suspensions": 3,
            },
            draft_context={
                "target_prefill_rows": 10000,
                "prompt_end_rows": 2048,
                "materialization_rows": 31,
                "avoided_rows": 7921,
                "restore_skipped": 1,
                "resets": 2,
            },
            constraint_masks={
                "overlap_batches": 5,
                "overlap_requests": 8,
                "last_target_forward_gpu_ms": 72.5,
                "total_target_forward_gpu_ms": 250.0,
                "last_residual_wait_ms": 1.5,
                "total_residual_wait_ms": 12.0,
            },
            memory_actual={
                "current_bytes": 700,
                "peak_bytes": 800,
            },
            memory_governor={"limit_bytes": 1000, "headroom_bytes": 200},
            metrics={
                "ttft_ms": {"p50": 10.0, "p95": 12.5, "samples": 5},
                "prefill_input_tokens": 2048,
                "prefill_wall_ms": 500.0,
                "prefill_tokens_per_second": 4096.0,
                "decode_output_tokens": 32,
                "decode_wall_ms": 64.0,
                "decode_cycle_ms": 80.0,
                "decode_tokens_per_second": 500.0,
                "draft_acceptance_rate": 0.875,
                "capacity_failures": 1,
                "metal_failures": 2,
            },
        )
        status, content_type, payload = harness.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/plain; version=0.0.4; charset=utf-8")
        metrics = payload.decode().splitlines()
        self.assertIn("splash_ready 1", metrics)
        self.assertIn('splash_memory_pressure{state="normal"} 1', metrics)
        self.assertIn('splash_thermal_state{state="serious"} 1', metrics)
        self.assertIn('splash_thermal_state{state="nominal"} 0', metrics)
        self.assertIn("splash_requests_submitted_total 7", metrics)
        self.assertIn("splash_scheduler_waiting_resources 3", metrics)
        self.assertIn("splash_scheduler_waiting_prefix 2", metrics)
        self.assertIn("splash_admission_waiting_memory 2", metrics)
        self.assertIn("splash_admission_waiting_concurrency 1", metrics)
        self.assertIn("splash_admission_held_behind_refusal 4", metrics)
        self.assertIn("splash_admission_restoring 1", metrics)
        self.assertIn("splash_admission_suspended 1", metrics)
        self.assertIn("splash_admission_oldest_wait_milliseconds 1250.0", metrics)
        self.assertIn("splash_scheduler_prefill_rows_total 2048", metrics)
        self.assertIn("splash_scheduler_decode_b3_total 3", metrics)
        self.assertIn("splash_kv_pages_allocated 8", metrics)
        self.assertIn("splash_kv_free_allocated_pages 2", metrics)
        self.assertFalse([line for line in metrics if "splash_kv_pages_free" in line])
        self.assertIn("splash_kv_allocated_bytes 8192", metrics)
        self.assertIn("splash_kv_extent_allocate_max_milliseconds 2.5", metrics)
        self.assertIn("splash_kv_extent_release_max_milliseconds 0.75", metrics)
        self.assertIn("splash_kv_extent_compact_max_milliseconds 1.5", metrics)
        self.assertFalse([line for line in metrics if "_max_ms " in line])
        self.assertIn("splash_state_entries 2", metrics)
        self.assertIn("splash_state_in_use 1", metrics)
        self.assertIn("splash_state_in_use_evictions_total 3", metrics)
        self.assertIn("splash_state_active_lanes 2", metrics)
        self.assertIn("splash_cache_hits_total 7", metrics)
        self.assertIn("splash_cache_cold_misses_total 4", metrics)
        self.assertIn("splash_cache_reused_tokens_total 1024", metrics)
        self.assertIn("splash_cache_lazy_junctions_total 2", metrics)
        self.assertIn("splash_cache_priority_suspensions_total 3", metrics)
        self.assertIn("splash_target_prefill_rows_total 10000", metrics)
        self.assertIn("splash_draft_context_prompt_end_rows_total 2048", metrics)
        self.assertIn("splash_draft_context_avoided_rows_total 7921", metrics)
        self.assertIn("splash_draft_state_restore_skipped_total 1", metrics)
        self.assertIn("splash_constraint_mask_overlap_batches_total 5", metrics)
        self.assertIn("splash_constraint_mask_overlap_requests_total 8", metrics)
        self.assertIn(
            "splash_constraint_mask_target_forward_gpu_milliseconds 72.5", metrics
        )
        self.assertIn("splash_constraint_mask_residual_wait_milliseconds 1.5", metrics)
        self.assertIn("splash_prefill_input_tokens_total 2048", metrics)
        self.assertIn("splash_prefill_tokens_per_second 4096.0", metrics)
        self.assertIn("splash_decode_output_tokens_total 32", metrics)
        self.assertIn("splash_decode_wall_milliseconds_total 64.0", metrics)
        self.assertIn("splash_decode_cycle_milliseconds_total 80.0", metrics)
        self.assertIn("splash_decode_tokens_per_second 500.0", metrics)
        self.assertIn("splash_capacity_failures_total 1", metrics)
        self.assertIn("splash_metal_failures_total 2", metrics)
        self.assertIn("splash_response_store_entries 0", metrics)
        self.assertIn("splash_memory_headroom_bytes 200", metrics)
        self.assertIn("splash_ttft_p95_milliseconds 12.5", metrics)
        self.assertIn("splash_draft_acceptance_ratio 0.875", metrics)

    def test_health_stays_live_when_native_is_not_ready(self):
        runtime = FakeRuntime()
        runtime.closed = True
        runtime.ready = False
        harness = self.harness(runtime)
        status, _, payload = harness.request("GET", "/health")
        health = json.loads(payload)
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.assertIs(health["ready"], False)
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual(
            (status, json.loads(payload)), (503, {"status": "unavailable"})
        )
        status, _, payload = harness.request("GET", "/status")
        self.assertFalse(json.loads(payload)["ready"])

    def test_status_identifies_the_http_instance_independently_of_readiness(self):
        first = self.harness(FakeRuntime())
        second = self.harness(FakeRuntime())
        before = first.server.status()["instance"]
        first.backend.runtime.ready = False
        status, _, payload = first.request("GET", "/status")
        snapshot = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["instance"], before)
        self.assertNotEqual(before["id"], second.server.status()["instance"]["id"])
        self.assertEqual(before["pid"], os.getpid())
        self.assertEqual(before["model"], "test-model")
        self.assertEqual((before["host"], before["port"]), first.server.server_address)
        self.assertGreater(before["started_at"], 0)
        self.assertLessEqual(before["started_at"], time.time())

    def test_critical_memory_pressure_marks_server_unready(self):
        class PressuredRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                # The engine is not ready under critical pressure.
                self.status_event = native_peer.status_event(
                    ready=False, memory_pressure="critical"
                )

        harness = self.harness(PressuredRuntime())
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual(
            (status, json.loads(payload)), (503, {"status": "unavailable"})
        )
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(payload)["ready"])

    def test_vision_capability_is_advertised_by_status_and_models(self):
        for vision, modalities in ((True, ["text", "image", "pdf"]), (False, ["text"])):
            with self.subTest(vision=vision):
                harness = self.harness(
                    FakeRuntime(), vision=vision, served_model_names=("local",)
                )
                status, _, payload = harness.request("GET", "/status")
                self.assertEqual(status, 200)
                snapshot = json.loads(payload)
                self.assertIs(snapshot["vision"], vision)
                self.assertEqual(snapshot["input_modalities"], modalities)
                status, _, payload = harness.request("GET", "/v1/models")
                self.assertEqual(status, 200)
                models = json.loads(payload)["data"]
                self.assertEqual(
                    [model["id"] for model in models], ["test-model", "local"]
                )
                for model in models:
                    self.assertIs(model["vision"], vision)
                    self.assertEqual(model["input_modalities"], modalities)
                    status, _, detail = harness.request(
                        "GET", f"/v1/models/{model['id']}"
                    )
                    self.assertEqual((status, json.loads(detail)), (200, model))


class BuildIdentityRouteTests(HarnessTestCase):
    def test_build_identity_is_served_without_resolving_per_request(self):
        harness = self.harness(FakeRuntime())
        identity = release_identity.BuildIdentity("source", "ebbc670", "src-1")
        harness.server.build_identity = identity
        status, _, payload = harness.request("GET", "/version")
        self.assertEqual((status, json.loads(payload)), (200, identity.as_dict()))
        handler = object.__new__(api.FrontendHandler)
        handler.server = harness.server
        self.assertEqual(handler.version_string(), "Splash/source-ebbc670")
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(json.loads(payload)["build"], identity.as_dict())
        status, _, payload = harness.request("GET", "/metrics")
        self.assertIn(
            'splash_build_info{version="source",commit="ebbc670",build_id="src-1"} 1',
            payload.decode().splitlines(),
        )


class PrometheusFamilyTests(unittest.TestCase):
    def test_an_absent_family_renders_nothing(self):
        absent = prometheus_metrics({})
        self.assertNotIn("splash_state_evictions_by_cause_total", absent)
        self.assertNotIn("splash_cache_checkpoint_publication_failures_total", absent)

    def test_state_evictions_render_by_cause(self):
        lines = prometheus_metrics(
            {"state": {"eviction_causes": {"pressure_pass": 4, "growth": 1}}}
        ).splitlines()
        self.assertIn(
            'splash_state_evictions_by_cause_total{cause="pressure_pass"} 4', lines
        )
        self.assertIn('splash_state_evictions_by_cause_total{cause="growth"} 1', lines)

    def test_draft_acceptance_is_left_out_while_speculation_is_off(self):
        status = {"metrics": {"draft_acceptance_rate": 0.5}}
        self.assertIn("splash_draft_acceptance_ratio 0.5", prometheus_metrics(status))
        status["metrics"]["speculative_decoding"] = False
        self.assertNotIn("splash_draft_acceptance_ratio", prometheus_metrics(status))

    def test_host_memory_renders_only_from_a_valid_host_sample(self):
        governor = {"host_headroom_bytes": 7, "host_measurement_valid": False}
        self.assertNotIn(
            "splash_host_headroom_bytes",
            prometheus_metrics({"memory_governor": governor}),
        )
        governor["host_measurement_valid"] = True
        self.assertIn(
            "splash_host_headroom_bytes 7",
            prometheus_metrics({"memory_governor": governor}).splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
