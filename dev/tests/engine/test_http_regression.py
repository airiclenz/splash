import contextlib
import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.benchmarks import http_regression as benchmark
from dev.tests import smoke_real as smoke


class CharacterTokenizer:
    def encode(self, content, **_kwargs):
        return list(content)

    def decode(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


class HttpRegressionTests(unittest.TestCase):
    def test_any_repository_selects_matching_package_and_api_name(self):
        for selected in (
            "incoai/Qwen3.8-27B-Splash",
            "incoai/Qwen3.6-35B-A3B-Splash",
            "community/custom-splash",
        ):
            with self.subTest(model=selected):
                arguments = smoke.parse_args(["--model", selected])
                self.assertEqual(arguments.model, selected)
                self.assertEqual(
                    arguments.package,
                    smoke.model_artifacts.installed_root(
                        smoke.model_artifacts.MODELS, selected
                    ),
                )
        model = "incoai/Qwen3.8-27B-Splash"
        arguments = smoke.parse_args(["--package", "custom-package", "--model", model])
        self.assertEqual(str(arguments.package), "custom-package")
        self.assertEqual(arguments.model, model)

    def test_real_helpers_require_a_canonical_model(self):
        for parse in (smoke.parse_args, benchmark.parse_args):
            for arguments in ([], ["--model", "qwen3.8-27b"], ["--model", "custom"]):
                with (
                    self.subTest(parse=parse, arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(arguments)

    def test_benchmark_resolves_the_selected_model_package(self):
        model = "incoai/Qwen3.6-35B-A3B-Splash"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "splash"
            binary.touch()
            (root / "splash.metallib").touch()
            package = root / "models" / model
            (package / "tokenizer").mkdir(parents=True)
            (package / "manifest.json").write_text("{}")
            with mock.patch.object(smoke.model_artifacts, "MODELS", root / "models"):
                arguments = benchmark.parse_args(
                    [
                        "--model",
                        model,
                        "--binary",
                        str(binary),
                        "--baseline-binary",
                        str(binary),
                    ]
                )
            self.assertEqual(arguments.model, model)
            self.assertEqual(arguments.package, package)

    def row(self, version, sample=0, latency=10):
        return {
            "version": version,
            "sample": sample,
            "context": 2048,
            "scenario": "cold",
            "prompt_sha256": "prompt",
            "response_sha256": "answer",
            "usage": {"completion_tokens": 1},
            "metrics": {"request_latency": {"ttft_ms": latency}},
        }

    def test_exact_prompt_lengths_unique_prefix_and_reproducibility(self):
        tokenizer = CharacterTokenizer()
        prompts = benchmark.prompts(tokenizer, [256, 2048], 2, "nonce")
        self.assertEqual(prompts, benchmark.prompts(tokenizer, [256, 2048], 2, "nonce"))
        for (_sample, context), prompt in prompts.items():
            self.assertEqual(
                len(
                    tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
                ),
                context,
            )
        self.assertNotEqual(prompts[0, 256][:32], prompts[1, 256][:32])

    def test_median_gate_keeps_raw_timing_separate_from_correctness(self):
        rows = [self.row("baseline"), self.row("candidate", latency=10.1)]
        self.assertTrue(benchmark.summarize(rows, 0.02)[0]["pass"])
        self.assertFalse(benchmark.summarize(rows, 0.005)[0]["pass"])

    def test_missing_duplicate_or_changed_transcript_fails(self):
        baseline, candidate = self.row("baseline"), self.row("candidate")
        for rows in ([baseline], [baseline, baseline, candidate]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows, 0.02)
        for field in ("response_sha256", "prompt_sha256"):
            changed = copy.deepcopy(candidate)
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "transcript differs"):
                benchmark.summarize([baseline, changed], 0.02)

    def test_decode_uses_native_decode_time_per_token(self):
        rows = [self.row("baseline"), self.row("candidate")]
        for row in rows:
            row["scenario"] = "decode"
            row["native_delta"] = {"decode_wall_ms": 100, "decode_output_tokens": 50}
        summary = benchmark.summarize(rows, 0.02)[0]
        self.assertEqual(summary["metric"], "decode_ms_per_token")
        self.assertEqual(summary["baseline_median"], 2)

    def test_round_cases_add_one_long_decode_case_only_with_the_flag(self):
        base = [
            (128, "decode", 64),
            (2048, "cold", 1),
            (2048, "exact", 1),
            (10000, "cold", 1),
            (10000, "exact", 1),
        ]
        arguments = SimpleNamespace(contexts=[2048, 10000], decode_tokens=0)
        self.assertEqual(benchmark.round_cases(arguments), base)
        arguments.decode_tokens = 30720
        self.assertEqual(
            benchmark.round_cases(arguments),
            base + [(benchmark.LONG_DECODE_CONTEXT, "long-decode", 30720)],
        )

    def test_long_decode_prompt_is_exact_per_sample_and_version_stable(self):
        tokenizer = CharacterTokenizer()
        first = benchmark.long_decode_prompt(tokenizer, 0, "nonce", 30720)
        self.assertEqual(
            first, benchmark.long_decode_prompt(tokenizer, 0, "nonce", 30720)
        )
        second = benchmark.long_decode_prompt(tokenizer, 1, "nonce", 30720)
        self.assertNotEqual(first[:32], second[:32])
        for prompt in (first, second):
            self.assertEqual(
                benchmark.chat_template_length(tokenizer, prompt),
                benchmark.LONG_DECODE_CONTEXT,
            )
            self.assertTrue(
                prompt.endswith("Write the integers from 1 to 61440, one per line.")
            )

    def test_long_decode_is_gated_on_native_decode_time_per_token(self):
        for slower, passes in ((1.03, False), (1.01, True)):
            rows = [self.row("baseline"), self.row("candidate", latency=1)]
            for row, decode_wall in zip(rows, (1000, 1000 * slower)):
                row["scenario"] = "long-decode"
                row["context"] = benchmark.LONG_DECODE_CONTEXT
                row["native_delta"] = {
                    "decode_wall_ms": decode_wall,
                    "decode_output_tokens": 500,
                }
            with self.subTest(slower=slower):
                summary = benchmark.summarize(rows, 0.02)[0]
                self.assertEqual(summary["scenario"], "long-decode")
                self.assertEqual(summary["metric"], "decode_ms_per_token")
                self.assertEqual(summary["pass"], passes)

    def measure_long_decode(self, completion_tokens, metrics):
        status = {
            "scheduler": {
                phase: 0
                for phase in (
                    "queued",
                    "waiting_resources",
                    "prefilling",
                    "decoding",
                    "waiting_mask",
                )
            },
            "state": {"active_cells": 0},
            "kv": {"pages_active": 0},
            "metrics": {"decode_wall_ms": 0, "decode_output_tokens": 0},
        }
        after = copy.deepcopy(status)
        after["metrics"].update(decode_wall_ms=900, decode_output_tokens=300)
        response = {
            "choices": [{"message": {"role": "assistant", "content": "1\n2"}}],
            "usage": {
                "prompt_tokens": benchmark.LONG_DECODE_CONTEXT,
                "completion_tokens": completion_tokens,
            },
            "metrics": metrics,
        }
        replies = iter(((200, status), (200, response), (200, after)))
        with (
            mock.patch.object(
                smoke, "request", side_effect=lambda *_a, **_k: next(replies)
            ),
            mock.patch.object(smoke, "validate_status"),
        ):
            return benchmark.measure(
                SimpleNamespace(port=8000),
                "test-model",
                "prompt",
                300,
                "long-decode",
                benchmark.LONG_DECODE_CONTEXT,
                60,
            )

    def test_measure_keeps_the_engine_block_and_requires_the_full_reply(self):
        metrics = {
            "cache": {"matched_tokens": 0},
            "prefill": {"tokens": benchmark.LONG_DECODE_CONTEXT},
        }
        row = self.measure_long_decode(300, copy.deepcopy(metrics))
        self.assertNotIn("engine", row["metrics"])
        self.assertEqual(row["native_delta"]["decode_wall_ms"], 900)
        engine = {"prefill_wall_ms": 12, "decode_wall_ms": 900, "decode_suspensions": 0}
        row = self.measure_long_decode(300, dict(metrics, engine=engine))
        self.assertEqual(row["metrics"]["engine"], engine)
        with self.assertRaisesRegex(smoke.SmokeFailure, "requested output length"):
            self.measure_long_decode(299, copy.deepcopy(metrics))


if __name__ == "__main__":
    unittest.main()
