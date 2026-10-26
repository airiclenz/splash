import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dev.benchmarks import agent_replay as replay
from dev.tests.engine.test_contended_probe import sse_line, token_event

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


# The stub sleeps this long before each turn's first token.
FIRST_TOKEN_DELAY_SECONDS = 0.02
TOOL = {"name": "read_file", "description": "Read a file.", "input_schema": {}}
CHAT_TOOL = {
    "type": "function",
    "function": {"name": "read_file", "description": "Read a file.", "parameters": {}},
}


def turn_script(
    prompt,
    cached,
    completion=10,
    prefill_wall_ms=1000,
    decode_wall_ms=500,
    drafted=8,
    accepted=6,
    admissions=1,
):
    """Return what the stub reports for one turn: usage and counter steps."""
    return {
        "prompt": prompt,
        "cached": cached,
        "completion": completion,
        "prefill_wall_ms": prefill_wall_ms,
        "decode_wall_ms": decode_wall_ms,
        "drafted": drafted,
        "accepted": accepted,
        "admissions": admissions,
    }


class ReplayStubServer:
    """Stream each turn in its path's shape and serve cumulative `/status`.

    Turn scripts are used in order and cycle, so several runs can share one
    stub; each POST advances the counters by its script before streaming.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.posts = []
        self.counters = {
            "prefill_wall_ms": 0,
            "decode_output_tokens": 0,
            "decode_wall_ms": 0,
            "drafted_tokens": 0,
            "accepted_draft_tokens": 0,
            "reused_tokens": 0,
            "submitted": 0,
        }
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_arguments):
                pass

            def do_GET(self):
                raw = json.dumps(stub.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                script = stub.scripts[len(stub.posts) % len(stub.scripts)]
                stub.posts.append((self.path, body))
                stub.advance(script)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                STREAMS[self.path](self, script)

            def data(self, payload, event=None):
                prefix = f"event: {event}\n".encode() if event else b""
                self.wfile.write(prefix + sse_line(payload) + b"\n")
                self.wfile.flush()

            def events(self, *pairs):
                for event, payload in pairs:
                    self.data({"type": event, **payload}, event)

        def chat_stream(handler, script):
            handler.data(token_event("role", "assistant"))
            handler.wfile.write(b": splash-keepalive\n\n")
            time.sleep(FIRST_TOKEN_DELAY_SECONDS)
            handler.data(token_event())
            usage = {
                "prompt_tokens": script["prompt"],
                "completion_tokens": script["completion"],
                "prompt_tokens_details": {"cached_tokens": script["cached"]},
            }
            handler.data({"choices": [], "usage": usage})
            handler.data("[DONE]")

        def anthropic_stream(handler, script):
            usage = {
                "input_tokens": script["prompt"] - script["cached"],
                "cache_read_input_tokens": script["cached"],
                "output_tokens": 0,
            }
            handler.events(
                ("message_start", {"message": {"usage": usage}}),
                ("ping", {}),
                ("content_block_start", {"index": 0, "content_block": {}}),
            )
            time.sleep(FIRST_TOKEN_DELAY_SECONDS)
            delta = {"type": "text_delta", "text": "1"}
            handler.events(
                ("content_block_delta", {"index": 0, "delta": delta}),
                ("content_block_stop", {"index": 0}),
                ("message_delta", {"usage": {"output_tokens": script["completion"]}}),
                ("message_stop", {}),
            )

        def responses_stream(handler, script):
            usage = {
                "input_tokens": script["prompt"],
                "input_tokens_details": {"cached_tokens": script["cached"]},
                "output_tokens": script["completion"],
            }
            handler.events(("response.created", {"response": {}}))
            time.sleep(FIRST_TOKEN_DELAY_SECONDS)
            handler.events(
                ("response.output_text.delta", {"delta": "1"}),
                ("response.completed", {"response": {"usage": usage}}),
            )

        STREAMS = {
            replay.CHAT_PATH: chat_stream,
            replay.ANTHROPIC_PATH: anthropic_stream,
            replay.RESPONSES_PATH: responses_stream,
        }
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def advance(self, script):
        self.counters["prefill_wall_ms"] += script["prefill_wall_ms"]
        self.counters["decode_output_tokens"] += script["completion"]
        self.counters["decode_wall_ms"] += script["decode_wall_ms"]
        self.counters["drafted_tokens"] += script["drafted"]
        self.counters["accepted_draft_tokens"] += script["accepted"]
        self.counters["reused_tokens"] += script["cached"]
        self.counters["submitted"] += script["admissions"]

    def snapshot(self):
        counters = self.counters
        return {
            "transport": {"status_stale": False},
            "metrics": {
                key: counters[key]
                for key in (
                    "prefill_wall_ms",
                    "decode_output_tokens",
                    "decode_wall_ms",
                    "drafted_tokens",
                    "accepted_draft_tokens",
                )
            },
            "cache": {"reused_tokens": counters["reused_tokens"]},
            "requests": {"submitted": counters["submitted"]},
        }

    def bodies(self):
        return [body for _, body in self.posts]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exception):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def growing(path, extra, *texts):
    """Return a chain's recorded requests: each adds a reply and a user turn."""
    history = [user(texts[0])]
    requests = []
    for index, text in enumerate(texts):
        if index:
            history = [*history, assistant(f"reply {index}"), user(text)]
        requests.append(
            record(1000 + index, path, {"model": "m", **extra, "messages": history})
        )
    return requests


def first_line(text):
    return text.partition("\n")[0]


class AgentReplayRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.recording = os.path.join(directory.name, "requests.jsonl")

    def write_recording(self, records):
        with open(self.recording, "w", encoding="utf-8") as stream:
            for line in records:
                stream.write(json.dumps(line) + "\n")

    def run_replay(self, stub, label="head", *options):
        """Run `run` against `stub`; return (exit code, run document, stderr)."""
        output = os.path.join(self.directory, f"{label}-{len(stub.posts)}.json")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = replay.main(
                [
                    "run",
                    "--recording",
                    self.recording,
                    "--chain",
                    "0",
                    "--base-url",
                    stub.base_url,
                    "--label",
                    label,
                    "--output",
                    output,
                    *options,
                ]
            )
        document = None
        if code == 0:
            with open(output, encoding="utf-8") as stream:
                document = json.load(stream)
        return code, document, stderr.getvalue(), output

    def replay_against(self, scripts, *options):
        """Replay chain 0 once; return (run document, bodies sent)."""
        with ReplayStubServer(scripts) as stub:
            code, document, stderr, _ = self.run_replay(stub, "head", *options)
        self.assertEqual(code, 0, stderr)
        return document, stub.bodies()

    def test_run_chatChain_recordsEachTurnFromUsageAndStatusDeltas(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c", "e"))
        scripts = [
            turn_script(2000, 0),
            turn_script(4000, 2000, prefill_wall_ms=500, drafted=0, accepted=0),
            turn_script(6000, 3990, completion=20, decode_wall_ms=400),
        ]

        document, _ = self.replay_against(scripts)

        self.assertEqual(
            [
                (
                    turn["prompt_tokens"],
                    turn["cached_tokens"],
                    turn["completion_tokens"],
                    turn["prefill_wall_ms"],
                    turn["tail_prefill_tokens_per_second"],
                    turn["decode_tokens_per_second"],
                    turn["draft_acceptance"],
                    turn["prefix_lost"],
                )
                for turn in document["turns"]
            ],
            [
                (2000, 0, 10, 1000, 2000.0, 20.0, 0.75, None),
                (4000, 2000, 10, 500, 4000.0, 20.0, None, False),
                (6000, 3990, 20, 1000, 2010.0, 50.0, 0.75, False),
            ],
        )

    def test_run_everyBody_isGreedyAndStreaming(self):
        self.write_recording(growing(replay.CHAT_PATH, {"temperature": 0.7}, "a", "c"))

        _, bodies = self.replay_against([turn_script(100, 0)])

        self.assertEqual(
            [
                (body["temperature"], body["stream"], body["stream_options"])
                for body in bodies
            ],
            [(0, True, {"include_usage": True})] * 2,
        )

    def test_run_twoRuns_shareANonceWithinARunAndDifferAcrossRuns(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c", "e"))

        with ReplayStubServer([turn_script(100, 0)]) as stub:
            self.run_replay(stub)
            self.run_replay(stub)

        nonces = [first_line(body["messages"][0]["content"]) for body in stub.bodies()]
        self.assertEqual(
            (len(set(nonces[:3])), len(set(nonces[3:])), nonces[0] != nonces[3]),
            (1, 1, True),
        )

    def test_run_nonce_leadsTheRecordedFirstMessage(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))

        document, bodies = self.replay_against([turn_script(100, 0)])

        self.assertEqual(
            [body["messages"][0]["content"] for body in bodies],
            [f"{document['nonce']}\na"] * 2,
        )

    def test_run_toolBearingChatChain_putsTheNonceInTheFirstToolDescription(self):
        extra = {"tools": [CHAT_TOOL]}
        self.write_recording(growing(replay.CHAT_PATH, extra, "a", "c"))

        document, bodies = self.replay_against([turn_script(100, 0)])

        self.assertEqual(
            [
                (body["tools"][0]["function"]["description"], body["messages"][0])
                for body in bodies
            ],
            [(f"{document['nonce']}\nRead a file.", user("a"))] * 2,
        )

    def test_run_toolBearingAnthropicChain_putsTheNonceInTheFirstToolDescription(self):
        extra = {"system": "agent", "max_tokens": 64, "tools": [TOOL]}
        self.write_recording(growing(replay.ANTHROPIC_PATH, extra, "a", "c"))

        document, bodies = self.replay_against([turn_script(100, 0)])

        self.assertEqual(
            [(body["tools"][0]["description"], body["system"]) for body in bodies],
            [(f"{document['nonce']}\nRead a file.", "agent")] * 2,
        )

    def test_placeNonce_anthropicWithoutTools_leadsTheSystemText(self):
        bodies = [
            {"system": "agent", "messages": [user("a")]},
            {"system": [{"type": "text", "text": "agent"}], "messages": [user("a")]},
        ]

        for body in bodies:
            replay.place_nonce(replay.ANTHROPIC_PATH, body, "nonce")

        self.assertEqual(
            [bodies[0]["system"], bodies[1]["system"][0]["text"]],
            ["nonce\nagent", "nonce\nagent"],
        )

    def test_run_anthropicTurn_takesPromptAndCachedFromMessageStart(self):
        extra = {"system": "agent", "max_tokens": 64}
        self.write_recording(growing(replay.ANTHROPIC_PATH, extra, "a", "c"))
        scripts = [turn_script(3000, 0), turn_script(5000, 2000, completion=7)]

        document, _ = self.replay_against(scripts)

        second = document["turns"][1]
        self.assertEqual(
            (
                second["prompt_tokens"],
                second["cached_tokens"],
                second["completion_tokens"],
                second["prefix_lost"],
            ),
            (5000, 2000, 7, False),
        )

    def test_run_chatAndAnthropicTurns_getAFirstTokenTime(self):
        delay_ms = FIRST_TOKEN_DELAY_SECONDS * 1000
        for path, extra in (
            (replay.CHAT_PATH, {}),
            (replay.ANTHROPIC_PATH, {"max_tokens": 64}),
        ):
            with self.subTest(path=path):
                self.write_recording(growing(path, extra, "a", "c"))

                document, _ = self.replay_against([turn_script(100, 0)])

                first = document["turns"][0]
                self.assertTrue(delay_ms <= first["ttft_ms"] <= first["http_wall_ms"])

    def test_run_lowCachedFollowUp_isFlaggedPrefixLost(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c", "e"))
        scripts = [turn_script(8000, 0), turn_script(9000, 6975), turn_script(9500, 0)]

        document, _ = self.replay_against(scripts)

        self.assertEqual(
            [turn["prefix_lost"] for turn in document["turns"]], [None, True, True]
        )

    def test_run_maxTokens_capsTheFieldEachPathReads(self):
        responses = [
            record(1, replay.RESPONSES_PATH, {"input": [user("a")]}),
            record(2, replay.RESPONSES_PATH, {"input": [user("a"), assistant("b")]}),
        ]
        cases = (
            (growing(replay.CHAT_PATH, {"max_tokens": 4096}, "a", "c"), 2048),
            (growing(replay.CHAT_PATH, {"max_completion_tokens": 100}, "a", "c"), 2048),
            (growing(replay.ANTHROPIC_PATH, {"max_tokens": 64}, "a", "c"), 32),
            (responses, 16),
        )
        sent = []
        for recording, cap in cases:
            self.write_recording(recording)

            _, bodies = self.replay_against(
                [turn_script(100, 0)], "--max-tokens", str(cap)
            )

            limits = replay.MAX_TOKEN_FIELDS[recording[0]["path"]]
            sent.append({name: bodies[0].get(name) for name in limits})
        self.assertEqual(
            sent,
            [
                {"max_completion_tokens": 2048, "max_tokens": 2048},
                {"max_completion_tokens": 100, "max_tokens": None},
                {"max_tokens": 32},
                {"max_output_tokens": 16},
            ],
        )

    def test_run_responsesChain_isMeasuredFromItsStream(self):
        path = replay.RESPONSES_PATH
        self.write_recording(
            [
                record(1, path, {"model": "m", "input": "hi"}),
                record(2, path, {"model": "m", "input": [user("hi"), assistant("o")]}),
            ]
        )
        scripts = [turn_script(300, 0), turn_script(500, 280, completion=4)]

        document, bodies = self.replay_against(scripts)

        second = document["turns"][1]
        self.assertEqual(
            (
                bodies[0]["input"],
                second["prompt_tokens"],
                second["cached_tokens"],
                second["completion_tokens"],
                second["ttft_ms"] is not None,
            ),
            (f"{document['nonce']}\nhi", 500, 280, 4, True),
        )

    def test_run_otherAdmissionMidTurn_aborts(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))
        scripts = [turn_script(100, 0), turn_script(200, 100, admissions=2)]

        with ReplayStubServer(scripts) as stub:
            code, _, stderr, output = self.run_replay(stub)

        self.assertEqual(
            (code, "2 admissions" in stderr, os.path.exists(output)), (1, True, False)
        )

    def test_run_cachedFirstTurn_isFlaggedNotCold(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))

        with ReplayStubServer([turn_script(100, 64)]) as stub:
            code, document, stderr, _ = self.run_replay(stub)

        self.assertEqual(
            (code, document["first_turn_cold"], "not cold" in stderr), (0, False, True)
        )

    def test_run_chainOutOfRange_fails(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))

        with ReplayStubServer([turn_script(100, 0)]) as stub:
            code, _, stderr, _ = self.run_replay(stub, "head", "--chain", "1")

        self.assertEqual((code, "out of range" in stderr, stub.posts), (1, True, []))

    def test_summarize_twoLabels_printsBothWithPerTurnAndTotalRows(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))
        outputs = []
        with ReplayStubServer([turn_script(2000, 0), turn_script(3000, 2000)]) as stub:
            for label in ("cutover", "head", "head", "cutover"):
                outputs.append(self.run_replay(stub, label)[3])
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            code = replay.main(["summarize", *outputs])

        blocks = stdout.getvalue().split("\n\n")
        self.assertEqual(
            (code, [block.splitlines()[0] for block in blocks]),
            (
                0,
                [
                    "cutover: 2 run(s), max tokens None",
                    "head: 2 run(s), max tokens None",
                ],
            ),
        )
        self.assertEqual(
            [line.split()[0] for line in blocks[1].splitlines()[2:]],
            ["1", "2", "total"],
        )

    def test_summarize_totalRow_sumsTheChainAndCountsLostPrefixes(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))
        with ReplayStubServer([turn_script(2000, 0), turn_script(3000, 0)]) as stub:
            output = self.run_replay(stub)[3]
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            replay.main(["summarize", output])

        total = stdout.getvalue().splitlines()[-1].split()
        self.assertEqual(
            (total[0], total[1], total[3], total[-1]), ("total", "5000", "20", "1/1")
        )

    def test_summarize_runsOfDifferentChains_fails(self):
        self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c"))
        with ReplayStubServer([turn_script(100, 0)]) as stub:
            first = self.run_replay(stub)[3]
            self.write_recording(growing(replay.CHAT_PATH, {}, "a", "c", "e"))
            second = self.run_replay(stub)[3]
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            code = replay.main(["summarize", first, second])

        self.assertEqual((code, "different chains" in stderr.getvalue()), (1, True))


if __name__ == "__main__":
    unittest.main()
