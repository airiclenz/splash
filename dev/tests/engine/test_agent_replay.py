import contextlib
import io
import json
import os
import tempfile
import unittest

from dev.benchmarks import agent_replay as replay

SYSTEM = {"role": "system", "content": "You are a coding agent."}


def user(text):
    """Return a user message with `text`."""
    return {"role": "user", "content": text}


def assistant(text):
    """Return an assistant message with `text`."""
    return {"role": "assistant", "content": text}


def record(received_unix_ms, path, body):
    """Return one recording line object."""
    return {"received_unix_ms": received_unix_ms, "path": path, "body": body}


def chat(received_unix_ms, *messages):
    """Return a recorded chat-completions request sending `messages`."""
    return record(
        received_unix_ms,
        replay.CHAT_PATH,
        {"model": "m", "messages": list(messages)},
    )


def anthropic(received_unix_ms, *messages):
    """Return a recorded Anthropic request sending `messages`."""
    return record(
        received_unix_ms,
        replay.ANTHROPIC_PATH,
        {"model": "m", "system": "agent", "max_tokens": 64, "messages": list(messages)},
    )


class AgentReplayChainsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.recording = os.path.join(directory.name, "requests.jsonl")

    def write_recording(self, records, tail=""):
        """Write `records` as JSON lines, then the raw `tail` text."""
        with open(self.recording, "w", encoding="utf-8") as stream:
            for line in records:
                stream.write(json.dumps(line) + "\n")
            stream.write(tail)

    def run_chains(self, *options):
        """Run `chains` on the recording; return (exit code, stdout, stderr)."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = replay.main(["chains", "--recording", self.recording, *options])
        return code, stdout.getvalue(), stderr.getvalue()

    def chains_json(self):
        """Run `chains --json` and return its chain documents."""
        code, stdout, _ = self.run_chains("--json")
        self.assertEqual(code, 0)
        return json.loads(stdout)["chains"]

    def interleaved_recording(self):
        """Two interleaved conversations, one lone request, one stateful body."""
        long_text = "x" * 400
        return [
            chat(1000, SYSTEM, user("short task")),
            anthropic(1100, user(long_text)),
            chat(1200, SYSTEM, user("short task"), assistant("ok"), user("next")),
            record(1250, replay.RESPONSES_PATH, {"input": "hello", "model": "m"}),
            record(
                1300,
                replay.RESPONSES_PATH,
                {"previous_response_id": "resp_1", "input": [user("more")]},
            ),
            anthropic(1400, user(long_text), assistant("a"), user("b")),
            anthropic(
                1500,
                user(long_text),
                assistant("a"),
                user("b"),
                assistant("c"),
                user("d"),
            ),
        ]

    def test_chains_interleavedConversations_yieldTwoChainsLargestFirst(self):
        self.write_recording(self.interleaved_recording())

        chains = self.chains_json()

        self.assertEqual(
            [(chain["path"], chain["turn_count"]) for chain in chains],
            [(replay.ANTHROPIC_PATH, 3), (replay.CHAT_PATH, 2)],
        )

    def test_chainsJson_listsEveryTurnsReceivedTime(self):
        self.write_recording(self.interleaved_recording())

        chains = self.chains_json()

        self.assertEqual(
            [[turn["received_unix_ms"] for turn in chain["turns"]] for chain in chains],
            [[1100, 1400, 1500], [1000, 1200]],
        )

    def test_chainsJson_reportsFinalSizeAndTimeRange(self):
        self.write_recording(self.interleaved_recording())

        largest = self.chains_json()[0]

        self.assertEqual(
            (
                largest["index"],
                largest["final_characters"],
                largest["final_estimated_tokens"],
                largest["first_received_unix_ms"],
                largest["last_received_unix_ms"],
            ),
            (0, 400 + 3 * len("user") + 2 * len("assistant") + 4, 108, 1100, 1500),
        )

    def test_chains_nonPrefixHistoryEdit_startsANewChain(self):
        self.write_recording(
            [
                chat(1, SYSTEM, user("task")),
                chat(2, SYSTEM, user("task"), assistant("first"), user("go")),
                chat(3, SYSTEM, user("task"), assistant("EDITED"), user("go")),
                chat(
                    4,
                    SYSTEM,
                    user("task"),
                    assistant("EDITED"),
                    user("go"),
                    assistant("y"),
                    user("z"),
                ),
            ]
        )

        chains = self.chains_json()

        received = [
            [turn["received_unix_ms"] for turn in chain["turns"]] for chain in chains
        ]
        self.assertEqual(sorted(received), [[1, 2], [3, 4]])

    def test_chains_requestJoinsLongestProperPrefix(self):
        self.write_recording(
            [
                chat(1, user("a")),
                chat(2, user("a"), assistant("b"), user("c")),
                chat(
                    3, user("a"), assistant("b"), user("c"), assistant("d"), user("e")
                ),
            ]
        )

        chains = self.chains_json()

        self.assertEqual([chain["turn_count"] for chain in chains], [3])

    def test_chains_samePrefixOnAnotherPath_doesNotJoin(self):
        self.write_recording(
            [
                chat(1, user("a")),
                anthropic(2, user("a"), assistant("b"), user("c")),
            ]
        )

        chains = self.chains_json()

        self.assertEqual(chains, [])

    def test_chains_responsesStringInput_chainsWithItsListFollowUp(self):
        self.write_recording(
            [
                record(1, replay.RESPONSES_PATH, {"input": "hi"}),
                record(
                    2,
                    replay.RESPONSES_PATH,
                    {"input": [user("hi"), assistant("hello"), user("bye")]},
                ),
            ]
        )

        chains = self.chains_json()

        self.assertEqual(
            [(chain["path"], chain["turn_count"]) for chain in chains],
            [(replay.RESPONSES_PATH, 2)],
        )

    def test_chains_unterminatedFinalLine_isSkippedAsAWriteInProgress(self):
        self.write_recording(
            [chat(1, user("a")), chat(2, user("a"), assistant("b"), user("c"))],
            tail='{"received_unix_ms": 3, "path": "/v1/chat',
        )

        chains = self.chains_json()

        self.assertEqual([chain["turn_count"] for chain in chains], [2])

    def test_chains_malformedTerminatedLine_failsNamingTheLine(self):
        self.write_recording([chat(1, user("a"))], tail="not json\n")

        code, _, stderr = self.run_chains()

        self.assertEqual((code, "recording line 2" in stderr), (1, True))

    def test_chains_missingRecording_failsWithError(self):
        code, _, stderr = self.run_chains()

        self.assertEqual((code, "cannot read recording" in stderr), (1, True))

    def test_chains_table_printsOneRowPerChainWithFirstAndLastTimes(self):
        self.write_recording(self.interleaved_recording())

        code, stdout, _ = self.run_chains()

        lines = stdout.splitlines()
        self.assertEqual(
            (code, len(lines), lines[1].split()),
            (0, 3, ["0", "3", replay.ANTHROPIC_PATH, "434", "108", "1100", "1500"]),
        )

    def test_chains_noChains_saysSo(self):
        self.write_recording([chat(1, user("a"))])

        code, stdout, _ = self.run_chains()

        self.assertEqual((code, stdout.startswith("no chains")), (0, True))


if __name__ == "__main__":
    unittest.main()
