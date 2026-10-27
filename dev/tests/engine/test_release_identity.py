import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server import release_identity


def git_run(commit="abc1234\n", changes=""):
    """A fake `subprocess.run` answering rev-parse and status like git."""

    def run(command, **_options):
        output = commit if "rev-parse" in command else changes
        return subprocess.CompletedProcess(command, 0, stdout=output)

    return mock.Mock(side_effect=run)


class ReleaseIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_resolve_packaged_reads_the_release_version_without_git(self):
        (self.root / "release.json").write_text(json.dumps({"version": "1.2.3"}))
        run = mock.Mock(side_effect=AssertionError("git in a release"))

        identity = release_identity.resolve("engine-build", root=self.root, run=run)

        self.assertEqual(
            (identity.as_dict(), identity.server_header),
            (
                {"version": "1.2.3", "commit": "unknown", "build_id": "engine-build"},
                "Splash/1.2.3",
            ),
        )

    def test_resolve_source_checkout_reports_source_and_the_short_commit(self):
        identity = release_identity.resolve(
            "engine-build", root=self.root, run=git_run()
        )

        self.assertEqual(
            (identity.as_dict(), identity.server_header),
            (
                {"version": "source", "commit": "abc1234", "build_id": "engine-build"},
                "Splash/source-abc1234",
            ),
        )

    def test_resolve_packaged_and_source_identities_are_non_empty_and_distinct(self):
        source = release_identity.resolve("b", root=self.root, run=git_run())
        (self.root / "release.json").write_text(json.dumps({"version": "1.2.3"}))
        packaged = release_identity.resolve("b", root=self.root, run=git_run())

        self.assertNotEqual(source.server_header, packaged.server_header)
        self.assertTrue(all(source.as_dict().values()))
        self.assertTrue(all(packaged.as_dict().values()))

    def test_resolve_dirty_tracked_tree_suffixes_the_commit(self):
        run = git_run(changes=" M server/server.py\n")

        identity = release_identity.resolve("b", root=self.root, run=run)

        self.assertEqual(identity.commit, "abc1234-dirty")

    def test_resolve_runs_bounded_git_in_the_program_root_ignoring_untracked(self):
        run = git_run()

        release_identity.resolve("b", root=self.root, run=run)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                ["git", "-C", str(self.root), "rev-parse", "--short", "HEAD"],
                [
                    "git",
                    "-C",
                    str(self.root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
            ],
        )
        for call in run.call_args_list:
            self.assertEqual(
                call.kwargs["timeout"], release_identity.GIT_TIMEOUT_SECONDS
            )

    def test_resolve_git_failure_yields_an_unknown_commit(self):
        failures = (
            FileNotFoundError("git"),
            subprocess.CalledProcessError(128, ["git"]),
            subprocess.TimeoutExpired(["git"], release_identity.GIT_TIMEOUT_SECONDS),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                run = mock.Mock(side_effect=failure)

                with mock.patch("builtins.print"):
                    identity = release_identity.resolve("b", root=self.root, run=run)

                self.assertEqual(
                    (identity.commit, identity.server_header),
                    ("unknown", "Splash/source-unknown"),
                )

    def test_resolve_unreadable_release_json_yields_an_unknown_version(self):
        (self.root / "release.json").write_text("{not json")

        with mock.patch("builtins.print"):
            identity = release_identity.resolve("b", root=self.root)

        self.assertEqual(identity.version, "unknown")

    def test_resolve_empty_build_id_is_unknown(self):
        identity = release_identity.resolve("", root=self.root, run=git_run())

        self.assertEqual(identity.build_id, "unknown")

    def test_native_build_id_reads_the_cache_identity_or_falls_back(self):
        cases = (
            ({"identity": {"cache": {"build_id": "b-1"}}}, "b-1"),
            ({"identity": {"cache": {"build_id": ""}}}, "unknown"),
            ({"identity": {"cache": {}}}, "unknown"),
            ({"identity": None}, "unknown"),
            ({}, "unknown"),
            (mock.Mock(), "unknown"),
            (None, "unknown"),
        )
        for status, expected in cases:
            with self.subTest(status=status):
                self.assertEqual(release_identity.native_build_id(status), expected)


if __name__ == "__main__":
    unittest.main()
