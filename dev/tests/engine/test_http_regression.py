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

    def measure_long_decode(
        self, completion_tokens, metrics, finish_reason=None, scenario="long-decode"
    ):
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
            "choices": [
                {
                    "message": {"role": "assistant", "content": "1\n2"},
                    "finish_reason": finish_reason,
                }
            ],
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
                scenario,
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

    def test_long_decode_accepts_an_eos_reply_from_three_quarters(self):
        metrics = {
            "cache": {"matched_tokens": 0},
            "prefill": {"tokens": benchmark.LONG_DECODE_CONTEXT},
        }
        self.assertEqual(benchmark.long_decode_floor(300), 225)
        self.assertEqual(benchmark.long_decode_floor(30720), 23040)
        row = self.measure_long_decode(225, copy.deepcopy(metrics), "stop")
        self.assertEqual(row["usage"]["completion_tokens"], 225)
        for completion, finish in ((224, "stop"), (299, "length"), (299, None)):
            with self.subTest(completion=completion, finish=finish):
                with self.assertRaisesRegex(smoke.SmokeFailure, "EOS floor of 225"):
                    self.measure_long_decode(completion, copy.deepcopy(metrics), finish)

    def test_decode_still_requires_the_exact_reply_length(self):
        metrics = {
            "cache": {"matched_tokens": 0},
            "prefill": {"tokens": benchmark.LONG_DECODE_CONTEXT},
        }
        self.measure_long_decode(300, copy.deepcopy(metrics), "length", "decode")
        with self.assertRaisesRegex(smoke.SmokeFailure, "requested output length"):
            self.measure_long_decode(299, copy.deepcopy(metrics), "stop", "decode")

    def test_long_decode_pairs_eos_replies_on_length_and_times_each_own(self):
        rows = []
        for sample, completion in ((0, 900), (1, 600)):
            for version, ms_per_token in (("baseline", 2), ("candidate", 2.01)):
                row = self.row(version, sample=sample)
                row.update(
                    scenario="long-decode",
                    context=benchmark.LONG_DECODE_CONTEXT,
                    usage={"completion_tokens": completion},
                    native_delta={
                        "decode_wall_ms": completion * ms_per_token,
                        "decode_output_tokens": completion,
                    },
                )
                rows.append(row)
        summary = benchmark.summarize(rows, 0.02)[0]
        self.assertEqual(summary["baseline_median"], 2)
        self.assertTrue(summary["pass"])
        changed = copy.deepcopy(rows)
        changed[1]["usage"]["completion_tokens"] = 899
        with self.assertRaisesRegex(ValueError, "transcript differs"):
            benchmark.summarize(changed, 0.02)

    def benchmark_arguments(self, *extra):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "splash"
            binary.touch()
            (root / "splash.metallib").touch()
            package = root / "package"
            (package / "tokenizer").mkdir(parents=True)
            (package / "manifest.json").write_text("{}")
            return benchmark.parse_args(
                [
                    "--model",
                    "incoai/Qwen3.8-27B-Splash",
                    "--package",
                    str(package),
                    "--binary",
                    str(binary),
                    "--baseline-binary",
                    str(binary),
                    *extra,
                ]
            )

    def test_self_check_requires_two_samples_and_a_long_decode_case(self):
        arguments = self.benchmark_arguments(
            "--samples", "2", "--decode-tokens", "20480", "--long-decode-self-check"
        )
        self.assertTrue(arguments.long_decode_self_check)
        self.assertFalse(
            self.benchmark_arguments("--samples", "2").long_decode_self_check
        )
        for extra in (
            ("--samples", "3", "--decode-tokens", "20480"),
            ("--samples", "5", "--decode-tokens", "20480"),
            ("--samples", "2"),
        ):
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.benchmark_arguments(*extra, "--long-decode-self-check")

    def test_self_check_sends_sample_zero_long_decode_prompt_in_every_round(self):
        prepared = {
            (0, 2048): "cold-0",
            (1, 2048): "cold-1",
            (0, "long-decode"): "long-0",
            (1, "long-decode"): "long-1",
        }
        context = benchmark.LONG_DECODE_CONTEXT
        for round_id in range(4):
            for sample in range(int(round_id >= 2), 2, 2):
                with self.subTest(round_id=round_id, sample=sample):
                    self.assertEqual(
                        benchmark.case_prompt(
                            prepared, sample, context, "long-decode", True
                        ),
                        "long-0",
                    )
                    self.assertEqual(
                        benchmark.case_prompt(prepared, sample, 2048, "cold", True),
                        f"cold-{sample}",
                    )
        self.assertEqual(
            benchmark.case_prompt(prepared, 1, context, "long-decode", False), "long-1"
        )

    def test_measure_rows_carry_finish_reason_and_requested_tokens(self):
        metrics = {
            "cache": {"matched_tokens": 0},
            "prefill": {"tokens": benchmark.LONG_DECODE_CONTEXT},
        }

        row = self.measure_long_decode(300, copy.deepcopy(metrics), "length")

        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["output_tokens"], 300)

    def self_check_rows(self, replies, finish_reason="length", completion=300):
        rows = []
        for (version, sample), response in replies.items():
            row = self.row(version, sample=sample)
            row.update(
                scenario="long-decode",
                context=benchmark.LONG_DECODE_CONTEXT,
                response_sha256=response,
                finish_reason=finish_reason,
                output_tokens=300,
                usage={"completion_tokens": completion},
                native_delta={
                    "decode_wall_ms": 2 * completion,
                    "decode_output_tokens": completion,
                },
            )
            rows.append(row)
        return rows

    def self_check(self, rows):
        return benchmark.summarize(
            rows, 0.02, long_decode_self_check=True, decode_tokens=300
        )

    def test_self_check_accepts_builds_that_diverge_only_at_the_length_cap(self):
        rows = self.self_check_rows(
            {
                ("baseline", 0): "off",
                ("candidate", 0): "on",
                ("candidate", 1): "on",
                ("baseline", 1): "off",
            }
        )

        summary = self.self_check(rows)[0]

        self.assertTrue(summary["pass"])
        self.assertEqual(
            benchmark.long_decode_outcome(rows, 300),
            {
                "self_match": {"baseline": True, "candidate": True},
                "builds_diverged": True,
                "length_capped": True,
            },
        )

    def test_self_check_fails_a_build_that_answers_its_prompt_twice_differently(self):
        rows = self.self_check_rows(
            {
                ("baseline", 0): "same",
                ("candidate", 0): "same",
                ("candidate", 1): "other",
                ("baseline", 1): "same",
            }
        )

        with self.assertRaisesRegex(ValueError, "within one build: candidate"):
            self.self_check(rows)

    def test_self_check_fails_a_cross_build_divergence_ended_by_eos(self):
        rows = self.self_check_rows(
            {
                ("baseline", 0): "off",
                ("candidate", 0): "on",
                ("candidate", 1): "on",
                ("baseline", 1): "off",
            }
        )
        rows[0].update(finish_reason="stop", usage={"completion_tokens": 250})
        rows[3].update(finish_reason="stop", usage={"completion_tokens": 250})

        with self.assertRaisesRegex(ValueError, "ended before 300 tokens"):
            self.self_check(rows)

    def test_self_check_passes_matching_eos_replies_above_the_floor(self):
        rows = self.self_check_rows(
            {
                ("baseline", 0): "same",
                ("candidate", 0): "same",
                ("candidate", 1): "same",
                ("baseline", 1): "same",
            },
            finish_reason="stop",
            completion=benchmark.long_decode_floor(300),
        )

        summary = self.self_check(rows)[0]

        self.assertTrue(summary["pass"])
        self.assertFalse(benchmark.long_decode_outcome(rows, 300)["builds_diverged"])

    def test_without_self_check_a_cross_build_long_decode_mismatch_fails(self):
        rows = self.self_check_rows(
            {
                ("baseline", 0): "off",
                ("candidate", 0): "on",
                ("candidate", 1): "on",
                ("baseline", 1): "off",
            }
        )

        with self.assertRaisesRegex(ValueError, "transcript differs"):
            benchmark.summarize(rows, 0.02)


if __name__ == "__main__":
    unittest.main()
