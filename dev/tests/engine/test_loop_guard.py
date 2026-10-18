import unittest
from unittest import mock

from dev.tests.engine.test_native_backend import FakeRuntime, make_job, success_result
from dev.tests.test_server import _byte_backend
from server import backend as backend_api
from server import output as model_output
from server import protocol as wire

REPEATED = "I should check the file again."
THINK_END_ID = 3
REPEATED_ID = 1
CONTENT_ID = 4


class LoopTokenizer:
    """Byte-level tokenizer with one repeated reasoning sentence per token."""

    def __init__(self):
        self.fragments = {
            REPEATED_ID: REPEATED + "\n",
            2: "Let me think.\n",
            THINK_END_ID: "</think>",
            CONTENT_ID: "The answer stays the same here.\n",
        }
        self.backend_tokenizer = _byte_backend(self.fragments)

    def decode(self, token_ids, **_kwargs):
        return "".join(self.fragments[token] for token in token_ids)

    @staticmethod
    def convert_tokens_to_ids(token):
        return THINK_END_ID if token == "</think>" else None


def feed_repeated(guard, sentence, times, separator=" "):
    tripped_at = None
    for index in range(times):
        if guard.feed(sentence + separator) and tripped_at is None:
            tripped_at = index + 1
    return tripped_at


class ReasoningLoopGuardTests(unittest.TestCase):
    def test_trips_at_exactly_the_limit(self):
        guard = model_output.ReasoningLoopGuard(24)

        tripped_at = feed_repeated(guard, REPEATED, 30)

        self.assertEqual(tripped_at, 24)
        self.assertTrue(guard.tripped)

    def test_does_not_trip_before_the_limit(self):
        guard = model_output.ReasoningLoopGuard(24)

        tripped_at = feed_repeated(guard, REPEATED, 23)

        self.assertIsNone(tripped_at)
        self.assertFalse(guard.tripped)

    def test_sentences_split_across_chunks_are_counted(self):
        guard = model_output.ReasoningLoopGuard(3)
        text = (REPEATED + " ") * 3
        results = []

        for start in range(0, len(text), 2):
            results.append(guard.feed(text[start : start + 2]))

        self.assertTrue(results[-1])
        self.assertFalse(any(results[:-1]))

    def test_case_spacing_and_punctuation_are_normalized(self):
        guard = model_output.ReasoningLoopGuard(3)

        guard.feed("I should  check the FILE again.\n")
        guard.feed("i should check the file, again!\n")
        tripped = guard.feed("I SHOULD check   the file again?\n")

        self.assertTrue(tripped)

    def test_zero_limit_disables_the_guard(self):
        guard = model_output.ReasoningLoopGuard(0)

        tripped_at = feed_repeated(guard, REPEATED, 100)

        self.assertIsNone(tripped_at)
        self.assertEqual(guard.pending_chars, 0)

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            model_output.ReasoningLoopGuard(-1)

    def test_benign_repetition_does_not_trip(self):
        cases = {
            "blank lines": ("", "\n"),
            "whitespace lines": ("   ", "\n"),
            "short sentences": ("Let me think.", " "),
        }
        for name, (sentence, separator) in cases.items():
            with self.subTest(name):
                guard = model_output.ReasoningLoopGuard(24)

                tripped_at = feed_repeated(guard, sentence, 60, separator)

                self.assertIsNone(tripped_at)

    def test_file_paths_decimals_and_e_g_stay_inside_their_sentence(self):
        guard = model_output.ReasoningLoopGuard(24)
        tripped_at = None

        for index in range(60):
            sentence = (
                f"We could edit backend.py, e.g. with ratio 0.5 and step {index}. "
            )
            if guard.feed(sentence) and tripped_at is None:
                tripped_at = index + 1

        self.assertIsNone(tripped_at)

    def test_overlong_unterminated_sentence_is_dropped(self):
        guard = model_output.ReasoningLoopGuard(2)
        long_sentence = "word " * 300

        for start in range(0, len(long_sentence), 7):
            guard.feed(long_sentence[start : start + 7])
            self.assertLessEqual(
                guard.pending_chars, model_output.REASONING_LOOP_MAX_SENTENCE_CHARS
            )
        guard.feed(".\n")
        guard.feed(long_sentence + ".\n")

        self.assertFalse(guard.tripped)
        self.assertEqual(guard.pending_chars, 0)

    def test_buffer_stays_under_one_sentence_over_thirty_thousand_sentences(self):
        guard = model_output.ReasoningLoopGuard(24)
        largest_pending = 0

        for index in range(30_000):
            sentence = f"Step number {index} looks fine to me. "
            for start in range(0, len(sentence), 5):
                guard.feed(sentence[start : start + 5])
                largest_pending = max(largest_pending, guard.pending_chars)

        self.assertFalse(guard.tripped)
        self.assertLess(largest_pending, len("Step number 29999 looks fine to me. "))


class CallbackStreamerLoopGuardTests(unittest.TestCase):
    @staticmethod
    def streamer(limit=24, on_stop=None):
        chunks = []
        streamer = backend_api.CallbackStreamer(
            LoopTokenizer(), chunks.append, (), on_stop, limit
        )
        return streamer, chunks

    def test_repeated_reasoning_trips_and_stops(self):
        stops = []
        streamer, _chunks = self.streamer(on_stop=lambda: stops.append(True))

        streamer.put_tokens([REPEATED_ID] * 23)
        before = (streamer.reasoning_loop_tripped, list(stops))
        streamer.put_tokens([REPEATED_ID] * 7)

        self.assertEqual(before, (False, []))
        self.assertTrue(streamer.reasoning_loop_tripped)
        self.assertEqual(stops, [True])
        self.assertEqual(len(streamer.token_ids), 24)

    def test_content_after_think_end_is_never_inspected(self):
        streamer, chunks = self.streamer()

        streamer.put_tokens([2, THINK_END_ID, *([REPEATED_ID] * 40)])
        streamer.end()

        self.assertFalse(streamer.reasoning_loop_tripped)
        self.assertIsNone(streamer.reasoning_loop_guard)
        self.assertEqual("".join(chunks).count(REPEATED), 40)

    def test_zero_limit_builds_no_guard(self):
        streamer, _chunks = self.streamer(limit=0)

        streamer.put_tokens([REPEATED_ID] * 40)

        self.assertIsNone(streamer.reasoning_loop_guard)
        self.assertFalse(streamer.reasoning_loop_tripped)


class NativeBackendLoopGuardTests(unittest.TestCase):
    def make_backend(self, limit=backend_api.DEFAULT_REASONING_LOOP_LIMIT):
        runtime = FakeRuntime()
        backend = backend_api.NativeBackend(
            runtime, LoopTokenizer(), reasoning_loop_limit=limit
        )
        self.addCleanup(backend.close)
        return backend, runtime

    @staticmethod
    def terminal(job, timeout=1.0):
        while True:
            event = job.events.get(timeout=timeout)
            if event[0] in ("done", "error"):
                return event

    def test_repeated_reasoning_finishes_with_length_and_one_warning(self):
        backend, runtime = self.make_backend()
        job = make_job()
        job.thinking = True
        self.assertTrue(backend.submit(job))
        call = runtime.calls[0]

        with mock.patch.object(backend_api, "print_status") as print_status:
            call.emit(wire.TokensEvent(call.request_id, 0, (REPEATED_ID,) * 23))
            cancelled_early = call.cancel_requested
            call.emit(wire.TokensEvent(call.request_id, 23, (REPEATED_ID,)))
            cancelled = call.cancel_requested
            generated = (REPEATED_ID,) * 26
            call.complete(
                result=success_result(
                    call, reason=wire.FinishReason.CANCELLED, tokens=generated
                )
            )
            kind, result = self.terminal(job)

        self.assertFalse(cancelled_early)
        self.assertTrue(cancelled)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "length")
        self.assertEqual(result.completion_tokens, 24)
        self.assertEqual(job.reasoning_tokens, 24)
        warnings = [
            entry.args[0]
            for entry in print_status.call_args_list
            if "reasoning loop" in entry.args[0]
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn(f"request {job.request_id}", warnings[0])

    def test_non_thinking_job_is_not_guarded(self):
        backend, runtime = self.make_backend()
        job = make_job()
        self.assertTrue(backend.submit(job))
        call = runtime.calls[0]

        call.emit(wire.TokensEvent(call.request_id, 0, (REPEATED_ID,) * 30))
        call.complete(result=success_result(call, tokens=(REPEATED_ID,) * 30))
        kind, result = self.terminal(job)

        self.assertFalse(call.cancel_requested)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "stop")

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            backend_api.NativeBackend(FakeRuntime(), LoopTokenizer(), None, -1)


if __name__ == "__main__":
    unittest.main()
