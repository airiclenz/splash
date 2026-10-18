import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import test_server as fixtures
from install import launcher
from server import frontend
from server import server as api

SERVER_ARGS = ["target", "draft", "--tokenizer", "tokenizer", "--model", "owner/repo"]
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class DefaultReasoningTests(unittest.TestCase):
    def harness(self, effort=None, **kwargs):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(), default_reasoning_effort=effort, **kwargs
        )
        self.addCleanup(harness.close)
        return harness

    def assert_effort(self, harness, expected):
        template = harness.tokenizer.templates[-1][1]
        if expected is None:
            self.assertNotIn("enable_thinking", template)
            self.assertNotIn("reasoning_effort", template)
        else:
            self.assertEqual(template["enable_thinking"], expected != "none")
            self.assertEqual(
                template.get("reasoning_effort"),
                None if expected == "none" else expected,
            )

    def test_default_and_every_explicit_effort_across_chat_and_responses(self):
        for default in (None, *EFFORTS):
            harness = self.harness(default)
            for path in ("/v1/chat/completions", "/v1/responses"):
                for effort in (None, *EFFORTS):
                    for stream in (False, True):
                        with self.subTest(
                            default=default, path=path, effort=effort, stream=stream
                        ):
                            if path.endswith("responses"):
                                body = fixtures.ServerTest.responses_body(stream=stream)
                                if effort is not None:
                                    body["reasoning"] = {"effort": effort}
                            else:
                                body = fixtures.ServerTest.body(stream=stream)
                                if effort is not None:
                                    body["reasoning_effort"] = effort
                            status, _, payload = harness.request("POST", path, body)
                            self.assertEqual(status, 200, payload)
                            self.assert_effort(
                                harness, effort if effort is not None else default
                            )
            harness.close()

    def test_null_effort_and_empty_responses_options_use_default(self):
        harness = self.harness("none")
        for path, options in (
            ("/v1/chat/completions", {"reasoning_effort": None}),
            ("/v1/responses", {"reasoning": None}),
            ("/v1/responses", {"reasoning": {}}),
            ("/v1/responses", {"reasoning": {"effort": None}}),
        ):
            body = (
                fixtures.ServerTest.responses_body()
                if path.endswith("responses")
                else fixtures.ServerTest.body()
            )
            status, _, payload = harness.request("POST", path, {**body, **options})
            self.assertEqual(status, 200, payload)
            self.assert_effort(harness, "none")

    def test_invalid_request_effort_is_not_replaced_by_default(self):
        harness = self.harness("none")
        for path in ("/v1/chat/completions", "/v1/responses"):
            for effort in ("", "invalid", False, 1, [], {}):
                body = (
                    fixtures.ServerTest.responses_body()
                    if path.endswith("responses")
                    else fixtures.ServerTest.body()
                )
                body.update(
                    {"reasoning": {"effort": effort}}
                    if path.endswith("responses")
                    else {"reasoning_effort": effort}
                )
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_thinking_and_count_tokens_ignore_openai_default(self):
        for default in ("none", "xhigh"):
            harness = self.harness(default)
            for thinking, expected in (
                (None, "none"),
                ({"type": "disabled"}, "none"),
                ({"type": "adaptive"}, "low"),
                ({"type": "enabled", "budget_tokens": 1024}, "low"),
            ):
                for path in ("/v1/messages", "/v1/messages/count_tokens?beta=true"):
                    body = fixtures.ServerTest.anthropic_body(
                        max_tokens=1100, output_config={"effort": "low"}
                    )
                    if thinking is not None:
                        body["thinking"] = thinking
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assert_effort(harness, expected)

    def test_apply_template_matches_generation(self):
        harness = self.harness("none")
        for effort in (None, "low"):
            body = fixtures.ServerTest.body()
            if effort:
                body["reasoning_effort"] = effort
            status, _, payload = harness.request("POST", "/apply-template", body)
            self.assertEqual(status, 200, payload)
            template = harness.tokenizer.templates[-1][1]
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200, payload)
            self.assertEqual(harness.tokenizer.templates[-1][1], template)
            self.assert_effort(harness, effort or "none")

    def test_unset_preserves_model_template_including_thinking_off(self):
        for default in (False, True):
            tokenizer = fixtures.TemplateTokenizer(
                fixtures.ServerTest.reasoning_template(default=default)
            )
            harness = self.harness(tokenizer=tokenizer)
            job, _, _ = harness.app.prepare(fixtures.ServerTest.body())
            self.assertEqual(job.thinking, default)
            job, _, _ = harness.app.prepare_responses(
                fixtures.ServerTest.responses_body()
            )
            self.assertEqual(job.thinking, default)

    def test_scoring_is_independent_of_generation_default(self):
        jobs = []
        for effort in (None, "none", "xhigh"):
            harness = self.harness(
                effort, tokenizer=fixtures.ServerTest.CharTokenizer(), max_context=8192
            )
            job, _ = harness.app.prepare_judgment(fixtures.ServerTest.judgment_body())
            self.assertFalse(job.thinking)
            jobs.append((job.prompt_tokens, job.score_tokens))
            systemone = harness.app.prepare_systemone(
                {
                    "model": "test-model",
                    "state": {},
                    "questions": {"q": {"type": "noul"}},
                }
            )
            self.assertFalse(systemone[0][2].thinking)
        self.assertEqual(jobs, [jobs[0]] * 3)

    def test_invalid_server_defaults_fail_before_startup(self):
        for value in ("", "off", "None", "HIGH", " low"):
            for parse, args in (
                (api.parse_args, SERVER_ARGS),
                (launcher.parse_args, ["serve", "--model", "owner/repo"]),
            ):
                with (
                    self.subTest(value=value, parse=parse.__module__),
                    mock.patch.dict(
                        os.environ, {"SPLASH_DEFAULT_REASONING_EFFORT": value}
                    ),
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(args)
                    with self.assertRaises(SystemExit):
                        parse([*args, "--default-reasoning-effort", value])
                    self.assertEqual(
                        parse(
                            [*args, "--default-reasoning-effort", "none"]
                        ).default_reasoning_effort,
                        "none",
                    )

    def test_cli_over_environment_and_launcher_forwarding(self):
        self.assertEqual(frontend.REASONING_EFFORTS, launcher.REASONING_EFFORTS)
        for env, explicit, expected in (
            (None, None, None),
            ("none", None, "none"),
            ("low", "xhigh", "xhigh"),
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SPLASH_DEFAULT_REASONING_EFFORT", None)
                if env is not None:
                    os.environ["SPLASH_DEFAULT_REASONING_EFFORT"] = env
                options = ["--default-reasoning-effort", explicit] if explicit else []
                self.assertEqual(
                    api.parse_args([*SERVER_ARGS, *options]).default_reasoning_effort,
                    expected,
                )
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(tmp)),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(launcher, "_ensure_installed"),
                    mock.patch.object(launcher.catalog, "spawn_refresh"),
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(["serve", "--model", "owner/repo", *options])
                    argv = execute.call_args.args[1]
                    self.assertEqual(
                        api.parse_args(argv[3:]).default_reasoning_effort, expected
                    )
                    self.assertEqual(
                        "--default-reasoning-effort" in argv, expected is not None
                    )


QWEN_EFFORTS = ("low", "medium", "xhigh")


class MaxReasoningEffortTests(unittest.TestCase):
    def harness(self, cap, default=None):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(),
            default_reasoning_effort=default,
            max_reasoning_effort=cap,
        )
        self.addCleanup(harness.close)
        return harness

    def template_app(self, cap):
        tokenizer = fixtures.TemplateTokenizer(
            fixtures.ServerTest.reasoning_template(efforts=QWEN_EFFORTS)
        )
        app = frontend.Frontend(
            tokenizer, None, "test-model", 128, 16, 1, 2, max_reasoning_effort=cap
        )
        tokenizer.templates.clear()
        return tokenizer, app

    def rendered_efforts(self, tokenizer):
        return [kwargs.get("reasoning_effort") for _, kwargs in tokenizer.templates]

    def test_request_above_cap_is_clamped_on_every_endpoint(self):
        harness = self.harness("medium")
        anthropic = fixtures.ServerTest.anthropic_body(
            max_tokens=1100, output_config={"effort": "xhigh"}
        )
        anthropic["thinking"] = {"type": "adaptive"}
        for path, body in (
            (
                "/v1/chat/completions",
                fixtures.ServerTest.body(reasoning_effort="xhigh"),
            ),
            (
                "/v1/responses",
                {
                    **fixtures.ServerTest.responses_body(),
                    "reasoning": {"effort": "xhigh"},
                },
            ),
            ("/v1/messages", anthropic),
        ):
            with self.subTest(path=path):
                status, _, payload = harness.request("POST", path, body)

                self.assertEqual(status, 200, payload)
                DefaultReasoningTests.assert_effort(self, harness, "medium")

    def test_request_at_or_below_cap_is_untouched(self):
        harness = self.harness("medium")
        for effort in ("none", "low", "medium"):
            with self.subTest(effort=effort):
                body = fixtures.ServerTest.body(reasoning_effort=effort)

                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", body
                )

                self.assertEqual(status, 200, payload)
                DefaultReasoningTests.assert_effort(self, harness, effort)

    def test_unset_effort_keeps_template_default_under_cap(self):
        harness = self.harness("low")

        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.ServerTest.body()
        )

        self.assertEqual(status, 200, payload)
        DefaultReasoningTests.assert_effort(self, harness, None)

    def test_template_alias_above_cap_renders_once_below_it(self):
        for effort in ("high", "xhigh", "max"):
            with self.subTest(effort=effort):
                tokenizer, app = self.template_app("high")

                job, _, _ = app.prepare(
                    fixtures.ServerTest.body(reasoning_effort=effort)
                )

                self.assertTrue(job.thinking)
                self.assertEqual(self.rendered_efforts(tokenizer), ["medium"])

    def test_alias_retry_never_drops_to_none(self):
        tokenizer, app = self.template_app("minimal")

        with self.assertRaises(frontend.APIError):
            app.prepare(fixtures.ServerTest.body(reasoning_effort="low"))

        self.assertEqual(self.rendered_efforts(tokenizer), ["minimal"])

    def test_alias_within_cap_is_used_unchanged(self):
        tokenizer, app = self.template_app("xhigh")

        app.prepare(fixtures.ServerTest.body(reasoning_effort="high"))

        self.assertEqual(self.rendered_efforts(tokenizer), ["xhigh"])

    def test_frontend_rejects_invalid_cap_and_default_above_cap(self):
        for options in (
            {"max_reasoning_effort": "HIGH"},
            {"max_reasoning_effort": "low", "default_reasoning_effort": "medium"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                frontend.Frontend(None, None, "test-model", 128, 16, 1, 2, **options)

    def test_default_above_cap_fails_before_startup(self):
        for parse, args in (
            (api.parse_args, SERVER_ARGS),
            (launcher.parse_args, ["serve", "--model", "owner/repo"]),
        ):
            for environment, options in (
                ({}, ["--default-reasoning-effort", "xhigh"]),
                ({"SPLASH_DEFAULT_REASONING_EFFORT": "xhigh"}, []),
                (
                    {"SPLASH_MAX_REASONING_EFFORT": "low"},
                    ["--default-reasoning-effort", "medium"],
                ),
            ):
                with (
                    self.subTest(parse=parse.__module__, environment=environment),
                    mock.patch.dict(os.environ, environment),
                    mock.patch("sys.stderr", io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    if "SPLASH_MAX_REASONING_EFFORT" not in environment:
                        options = [*options, "--max-reasoning-effort", "medium"]
                    parse([*args, *options])
            for value in ("", "off", "HIGH"):
                with (
                    self.subTest(parse=parse.__module__, value=value),
                    mock.patch.dict(os.environ, {"SPLASH_MAX_REASONING_EFFORT": value}),
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(args)
                    with self.assertRaises(SystemExit):
                        parse([*args, "--max-reasoning-effort", value])
            parsed = parse(
                [
                    *args,
                    "--default-reasoning-effort",
                    "medium",
                    "--max-reasoning-effort",
                    "medium",
                ]
            )
            self.assertEqual(parsed.max_reasoning_effort, "medium")

    def test_cli_over_environment_and_launcher_forwarding(self):
        for env, explicit, expected in (
            (None, None, None),
            ("low", None, "low"),
            ("low", "high", "high"),
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SPLASH_MAX_REASONING_EFFORT", None)
                os.environ.pop("SPLASH_DEFAULT_REASONING_EFFORT", None)
                if env is not None:
                    os.environ["SPLASH_MAX_REASONING_EFFORT"] = env
                options = ["--max-reasoning-effort", explicit] if explicit else []
                self.assertEqual(
                    api.parse_args([*SERVER_ARGS, *options]).max_reasoning_effort,
                    expected,
                )
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(tmp)),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(launcher, "_ensure_installed"),
                    mock.patch.object(launcher.catalog, "spawn_refresh"),
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(["serve", "--model", "owner/repo", *options])

                    argv = execute.call_args.args[1]
                    os.environ.pop("SPLASH_MAX_REASONING_EFFORT", None)
                    self.assertEqual(
                        api.parse_args(argv[3:]).max_reasoning_effort, expected
                    )


class EffortAliasProbeTests(unittest.TestCase):
    REJECTS_HIGH = ("minimal", "low", "medium", "xhigh", "max")

    @staticmethod
    def template(efforts=None, *, failing_message=None):
        source = fixtures.ServerTest.reasoning_template(efforts=efforts)
        if failing_message is None:
            return source
        return (
            "{% if messages[0].content == "
            + repr(failing_message)
            + " %}{{ raise_exception('bad message') }}{% endif %}"
            + source
        )

    @staticmethod
    def app(tokenizer):
        app = frontend.Frontend(tokenizer, None, "test-model", 128, 16, 1, 2)
        tokenizer.templates.clear()
        return app

    @staticmethod
    def rendered_efforts(tokenizer):
        return [kwargs.get("reasoning_effort") for _, kwargs in tokenizer.templates]

    def test_rejected_effort_renders_its_alias_once(self):
        tokenizer = fixtures.TemplateTokenizer(self.template(self.REJECTS_HIGH))
        app = self.app(tokenizer)

        job, thinking, _ = app.prepare(
            fixtures.ServerTest.body(reasoning_effort="high")
        )

        self.assertTrue(thinking)
        self.assertTrue(job.thinking)
        self.assertEqual(self.rendered_efforts(tokenizer), ["xhigh"])

    def test_probed_render_matches_the_per_request_retry(self):
        for effort in EFFORTS[1:]:
            with self.subTest(effort=effort):
                renders = []
                for probe in (True, False):
                    tokenizer = fixtures.TemplateTokenizer(
                        self.template(self.REJECTS_HIGH)
                    )
                    app = self.app(tokenizer)
                    if not probe:
                        app.effort_aliases = ({}, {})
                    job, _, _ = app.prepare(
                        fixtures.ServerTest.body(reasoning_effort=effort)
                    )
                    renders.append((tokenizer.templates[-1], job.prompt_tokens))

                self.assertEqual(renders[0], renders[1])

    def test_template_accepting_every_effort_records_no_alias(self):
        tokenizer = fixtures.TemplateTokenizer(self.template(EFFORTS[1:]))

        app = self.app(tokenizer)

        self.assertEqual(app.effort_aliases, ({}, {}))

    def test_message_errors_still_raise_after_an_alias(self):
        tokenizer = fixtures.TemplateTokenizer(
            self.template(self.REJECTS_HIGH, failing_message="boom")
        )
        app = self.app(tokenizer)
        for effort, expected in (("high", ["xhigh"]), ("medium", ["medium"])):
            with self.subTest(effort=effort):
                tokenizer.templates.clear()
                body = fixtures.ServerTest.body(
                    reasoning_effort=effort,
                    messages=[{"role": "user", "content": "boom"}],
                )

                with self.assertRaises(frontend.APIError):
                    app.prepare(body)

                self.assertEqual(self.rendered_efforts(tokenizer), expected)

    def test_probe_failures_never_fail_startup(self):
        class RaisingTemplate(fixtures.FakeTokenizer):
            def get_chat_template(self, **kwargs):
                return "template"

            def apply_chat_template(self, messages, **kwargs):
                raise ValueError("template raised")

        class MissingTemplate(fixtures.FakeTokenizer):
            def get_chat_template(self, **kwargs):
                raise ValueError("no chat template")

        for tokenizer in (RaisingTemplate(), MissingTemplate()):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                app = frontend.Frontend(tokenizer, None, "test-model", 128, 16, 1, 2)

                self.assertEqual(app.effort_aliases, ({}, {}))

    def test_aliases_follow_the_template_source_the_request_selects(self):
        tokenizer = fixtures.TemplateTokenizer(
            {
                "default": self.template(self.REJECTS_HIGH),
                "tool_use": self.template(EFFORTS[1:]),
            }
        )
        app = self.app(tokenizer)
        tool = fixtures.ServerTest.rich_weather_tool()
        for tools, expected in (([tool], ["high"]), (None, ["xhigh"])):
            with self.subTest(tools=tools is not None):
                tokenizer.templates.clear()
                body = fixtures.ServerTest.body(reasoning_effort="high")
                if tools is not None:
                    body["tools"] = tools

                app.prepare(body)

                self.assertEqual(self.rendered_efforts(tokenizer), expected)
