"""Compare retained servers with matched HTTP requests in ABBA order.

Run as ``python -m dev.benchmarks.http_regression --baseline-binary PATH``.
Reuses the real HTTP test lifecycle; never contacts an existing server.
With ``--burst N`` it measures bursts of N concurrent requests instead, and
with ``--follow-up`` each conversation's next turn after its burst.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from dev.benchmarks import abba, weights
from dev.tests import smoke_real as smoke


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prompts(tokenizer, contexts: list[int], samples: int, nonce: str) -> dict:
    result = {}
    filler = tokenizer.encode(
        " The archive contains routine project notes and implementation details",
        add_special_tokens=False,
    )
    for sample in range(samples):
        for context in contexts:
            prefix = digest([nonce, sample, context]) + "\n"
            suffix = "\nWrite the integers from 1 to 200, one per line."
            rows = context
            for _ in range(8):
                content = (
                    prefix
                    + tokenizer.decode((filler * (rows // len(filler) + 1))[:rows])
                    + suffix
                )
                count = len(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                if count == context:
                    break
                rows -= count - context
            else:
                raise ValueError(f"cannot construct an exact {context}-token prompt")
            result[sample, context] = content
    return result


# ------------------------------------------------------------------------------
# The long-decode case's exact chat-template prompt length; its rows are keyed
# on this context with scenario "long-decode", apart from every --contexts row.
LONG_DECODE_CONTEXT = 256

# The scenarios gated on native decode time per token instead of TTFT.
DECODE_SCENARIOS = ("decode", "long-decode")


# ------------------------------------------------------------------------------
def long_decode_floor(output_tokens: int) -> int:
    """Return the shortest EOS-ended long-decode reply that still counts.

    The model ends its count at a nonce-dependent point (seen at 27.5k-29.6k
    tokens) and nothing forces N tokens, so a reply stopped by EOS at three
    quarters of N or more is accepted; without `--long-decode-self-check`,
    `summarize` already pairs baseline and candidate on reply text and length
    and gates each on its own decode time per token, so both versions decode
    the same tokens.
    """
    return (3 * output_tokens + 3) // 4


# ------------------------------------------------------------------------------
def chat_template_length(tokenizer, content: str) -> int:
    """Return the token count of `content` as one user turn ready for a reply."""
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


# ------------------------------------------------------------------------------
def long_decode_prompt(
    tokenizer, sample: int, nonce: str, output_tokens: int, context=LONG_DECODE_CONTEXT
) -> str:
    """Return an exact `context`-token prompt asking for more than `output_tokens`.

    Built from `live_probe.cold_prompt` (lead nonce, filler, over-long count
    instruction) with its seed derived from the run nonce and `sample`, so both
    versions get the same prompt and each sample a different one; the filler is
    then cut to the token so the chat-template length is exactly `context`.
    """
    # Imported here: live_probe imports this module at load time.
    from dev.benchmarks.live_probe import cold_prompt

    seed = digest([nonce, sample, "long-decode"])
    lead, filler, instruction = cold_prompt(2 * context, output_tokens, seed).split(
        "\n"
    )
    tokens = tokenizer.encode(filler, add_special_tokens=False)
    rows = context
    for _ in range(8):
        content = f"{lead}\n{tokenizer.decode(tokens[:rows])}\n{instruction}"
        count = chat_template_length(tokenizer, content)
        if count == context:
            return content
        rows -= count - context
    raise ValueError(f"cannot construct an exact {context}-token long-decode prompt")


# ------------------------------------------------------------------------------
def round_cases(args) -> list[tuple[int, str, int]]:
    """Return one sample's `(context, scenario, output_tokens)` cases in order.

    `--decode-tokens N` appends one long-decode case asking for N tokens (an
    EOS-ended reply counts from `long_decode_floor(N)`); without it the list is
    the base decode/cold/exact set.
    """
    cases = [(128, "decode", 64)] + [
        (context, scenario, 1)
        for context in args.contexts
        for scenario in ("cold", "exact")
    ]
    if args.decode_tokens:
        cases.append((LONG_DECODE_CONTEXT, "long-decode", args.decode_tokens))
    return cases


# ------------------------------------------------------------------------------
def case_prompt(
    prepared: dict, sample: int, context: int, scenario: str, self_check: bool
) -> str:
    """Return the prompt one round case sends.

    A long-decode case takes its sample's long-decode prompt, or sample 0's
    under `--long-decode-self-check`, so all four rounds decode one prompt and
    each build answers it twice; every other case takes its sample's exact
    `context`-token prompt.
    """
    if scenario != "long-decode":
        return prepared[sample, context]
    return prepared[0 if self_check else sample, "long-decode"]


# ------------------------------------------------------------------------------
def transcript(row: dict) -> tuple:
    """Return the prompt, reply sha256 and reply token count a row is paired on."""
    return (
        row["prompt_sha256"],
        row["response_sha256"],
        row["usage"]["completion_tokens"],
    )


# ------------------------------------------------------------------------------
def long_decode_outcome(records: list[dict], decode_tokens: int) -> dict:
    """Describe the long-decode rows of a `--long-decode-self-check` run.

    Returns `self_match`, per version, true when its replies (at least two)
    share one transcript; `builds_diverged`, true when the versions' transcript
    sets differ; and `length_capped`, true when every reply stopped on length
    at exactly `decode_tokens`, so none ended by EOS. Pure; raises nothing.
    """
    transcripts = defaultdict(list)
    is_length_capped = True
    for row in records:
        if row["scenario"] != "long-decode":
            continue
        transcripts[row["version"]].append(transcript(row))
        is_length_capped = is_length_capped and (
            row.get("finish_reason") == "length"
            and row["usage"]["completion_tokens"] == decode_tokens
        )
    return {
        "self_match": {
            version: len(replies) >= 2 and len(set(replies)) == 1
            for version, replies in sorted(transcripts.items())
        },
        "builds_diverged": set(transcripts["baseline"])
        != set(transcripts["candidate"]),
        "length_capped": is_length_capped,
    }


# ------------------------------------------------------------------------------
def require_long_decode_reproducible(records: list[dict], decode_tokens: int):
    """Raise ValueError unless the long-decode rows pass the self-check rule.

    Each version must answer its repeated long-decode prompt identically; the
    versions may disagree only when every reply ran to exactly
    `decode_tokens` on length, since checkpoints regroup verify rows and can
    flip a greedy near-tie, while an EOS-ended reply must match across builds.
    """
    outcome = long_decode_outcome(records, decode_tokens)
    unstable = [
        version for version, match in outcome["self_match"].items() if not match
    ]
    if unstable:
        raise ValueError(
            f"long-decode transcript differs within one build: {', '.join(unstable)}"
        )
    if outcome["builds_diverged"] and not outcome["length_capped"]:
        raise ValueError(
            "baseline/candidate long-decode transcript differs and a reply "
            f"ended before {decode_tokens} tokens"
        )


# ------------------------------------------------------------------------------
# The cumulative native counters `measure` diffs around one request; `summarize`
# reads the decode pair from the resulting `native_delta`.
NATIVE_DELTA_PATHS = tuple(
    ("metrics", key)
    for key in (
        "prefill_wall_ms",
        "decode_wall_ms",
        "decode_cycle_ms",
        "prefill_input_tokens",
        "decode_output_tokens",
        "drafted_tokens",
        "accepted_draft_tokens",
    )
)


# ------------------------------------------------------------------------------
def status_value(snapshot, path) -> float:
    """Return the number at a nested key path of a `/status` document, or 0.

    A missing key, a non-object on the way, or a non-numeric leaf reads as 0 so
    a counter an older server does not export diffs to 0 instead of raising.
    """
    value = snapshot
    for key in path:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value


# ------------------------------------------------------------------------------
def status_delta(before, after, paths) -> dict:
    """Diff cumulative `/status` counters between two snapshots.

    `paths` are tuples of nested keys, e.g. ``("metrics", "prefill_wall_ms")``;
    the result maps each path tuple to ``after - before``, reading an absent
    counter as 0 on either side. Pure: neither snapshot is modified.
    """
    return {
        path: status_value(after, path) - status_value(before, path) for path in paths
    }


def measure(server, model, content, output_tokens, scenario, context, timeout):
    _, before = smoke.request(server.port, "GET", "/status")
    started = time.monotonic()
    code, response = smoke.request(
        server.port,
        "POST",
        "/v1/chat/completions",
        smoke.chat_body(model, content, max_completion_tokens=output_tokens),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - started) * 1000
    smoke.require(code == 200, f"{scenario}: HTTP {code}: {response!r}")
    _, after = smoke.request(server.port, "GET", "/status")
    smoke.validate_status(after)
    metrics, usage = response["metrics"], response["usage"]
    smoke.require(usage["prompt_tokens"] == context, "prompt token count changed")
    reused = metrics["cache"]["matched_tokens"]
    prefill = metrics["prefill"]["tokens"]
    if scenario == "exact":
        smoke.require(0 < reused < context and prefill < context, "exact hit missing")
    else:
        smoke.require(reused == 0 and prefill == context, "cold prompt reused cache")
    completion = usage["completion_tokens"]
    if scenario == "decode":
        smoke.require(
            completion == output_tokens,
            "decode sample ended before the requested output length",
        )
    elif scenario == "long-decode":
        smoke.require(
            completion == output_tokens
            or (
                response["choices"][0].get("finish_reason") == "stop"
                and completion >= long_decode_floor(output_tokens)
            ),
            "long-decode sample ended before the requested output length "
            f"and below its EOS floor of {long_decode_floor(output_tokens)}",
        )
    for phase in (
        "queued",
        "waiting_resources",
        "prefilling",
        "decoding",
        "waiting_mask",
    ):
        smoke.require(after["scheduler"][phase] == 0, f"request left {phase} work")
    smoke.require(
        after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
        "active resources leaked",
    )
    delta = {
        path[-1]: value
        for path, value in status_delta(before, after, NATIVE_DELTA_PATHS).items()
    }
    return {
        "scenario": scenario,
        "context": context,
        "prompt_sha256": digest(content),
        "response_sha256": digest(response["choices"][0]["message"]),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "output_tokens": output_tokens,
        "usage": usage,
        "metrics": metrics,
        "native_delta": delta,
        "http_wall_ms": elapsed,
    }


# A burst's requests are short, so their replay points come due together.
BURST_OUTPUT_TOKENS = 16
FOLLOW_UP = "Continue with the next ten integers."


def measure_burst(server, model, contents, follow_up, timeout):
    """N chat requests of equal length at once, then, with follow_up, each
    conversation's next turn one after another: the publication failures of
    the burst's replay points, its decode rate, and whether each next turn
    resumed at its conversation's replay point."""
    _, before = smoke.request(server.port, "GET", "/status")
    responses = [None] * len(contents)

    def send(index):
        responses[index] = smoke.request(
            server.port,
            "POST",
            "/v1/chat/completions",
            smoke.chat_body(
                model, contents[index], max_completion_tokens=BURST_OUTPUT_TOKENS
            ),
            timeout=timeout,
        )

    threads = [threading.Thread(target=send, args=(i,)) for i in range(len(contents))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for code, response in responses:
        smoke.require(code == 200, f"burst: HTTP {code}: {response!r}")
    _, after = smoke.request(server.port, "GET", "/status")
    smoke.validate_status(after)
    decode_ms = after["metrics"]["decode_wall_ms"] - before["metrics"]["decode_wall_ms"]
    decoded = (
        after["metrics"]["decode_output_tokens"]
        - before["metrics"]["decode_output_tokens"]
    )
    failures = "replay_state_publication_failures"
    row = {
        "scenario": "burst",
        "replay_state_publication_failures": after["cache"][failures]
        - before["cache"][failures],
        "decode_ms_per_token": decode_ms / max(1, decoded),
        "maximum_context_tokens": after["maximum_context_tokens"],
        "follow_ups": [],
    }
    if not follow_up:
        return row
    page_tokens = after["memory_plan"]["budget"]["kv_page_tokens"]
    for content, (_, response) in zip(contents, responses):
        answer = response["choices"][0]["message"]["content"]
        body = smoke.chat_body(model, FOLLOW_UP, max_completion_tokens=1)
        body["messages"] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": FOLLOW_UP},
        ]
        code, turn = smoke.request(
            server.port, "POST", "/v1/chat/completions", body, timeout=timeout
        )
        smoke.require(code == 200, f"follow-up: HTTP {code}: {turn!r}")
        # The replay point lies in the first turn's last page or the page
        # before its generation prompt; any other match is a checkpoint at
        # least a page below it.
        matched = turn["metrics"]["cache"]["matched_tokens"]
        resumed = matched >= response["usage"]["prompt_tokens"] - 2 * page_tokens
        row["follow_ups"].append(
            {
                "ttft_ms": turn["metrics"]["request_latency"]["ttft_ms"],
                "matched_tokens": matched,
                "resumed": resumed,
            }
        )
    return row


ROUNDS = ("baseline", "candidate", "candidate", "baseline")


def summarize(
    records: list[dict],
    *,
    long_decode_self_check: bool = False,
    decode_tokens: int = 0,
) -> list[dict]:
    """Per context and scenario, the ABBA verdict (abba.compare) of the
    median latency of each of the four rounds; the matched transcripts of
    both versions must be identical, except the long-decode rows of a
    `--long-decode-self-check` run, which require_long_decode_reproducible
    judges as a whole."""
    groups = defaultdict(lambda: defaultdict(list))
    outputs = {}
    for row in records:
        key = row["sample"], row["context"], row["scenario"]
        output = transcript(row)
        paired = outputs.setdefault(key, {})
        if row["version"] in paired:
            raise ValueError(f"duplicate performance sample: {key}")
        is_self_checked = long_decode_self_check and row["scenario"] == "long-decode"
        if paired and not is_self_checked and next(iter(paired.values())) != output:
            raise ValueError(f"baseline/candidate transcript differs: {key}")
        paired[row["version"]] = output
        # Decode is judged by the engine's cycle, so host work between
        # commands counts.
        if row["scenario"] in DECODE_SCENARIOS:
            latency = row["native_delta"]["decode_cycle_ms"] / max(
                1, row["native_delta"]["decode_output_tokens"]
            )
        else:
            latency = row["metrics"]["request_latency"]["ttft_ms"]
        groups[row["context"], row["scenario"]][row["round"]].append(latency)
    results = []
    if any(set(pair) != {"baseline", "candidate"} for pair in outputs.values()):
        raise ValueError("unpaired request samples")
    if long_decode_self_check:
        require_long_decode_reproducible(records, decode_tokens)
    for (context, scenario), rounds in sorted(groups.items()):
        if sorted(rounds) != list(range(len(ROUNDS))):
            raise ValueError(f"a round has no samples: {context} {scenario}")
        comparison = abba.compare_samples(rounds[index] for index in range(len(ROUNDS)))
        results.append(
            {
                "context": context,
                "scenario": scenario,
                "metric": (
                    "decode_cycle_ms_per_token"
                    if scenario in DECODE_SCENARIOS
                    else "ttft_ms"
                ),
                "samples_per_round": [
                    len(rounds[index]) for index in range(len(ROUNDS))
                ],
                "baseline_median": statistics.median(rounds[0] + rounds[3]),
                "candidate_median": statistics.median(rounds[1] + rounds[2]),
                **comparison,
            }
        )
    return results


def summarize_bursts(records: list[dict]) -> list[dict]:
    """Per context, each version's replay-point publication failures per
    burst, the next turns that resumed at their replay point and the
    advertised context, and the ABBA verdicts of the bursts' decode time per
    token and of the next turns' time to first token, when they were sent.
    The candidate is kept when it publishes every replay point where the
    baseline lost some, a conclusive verdict finds its decode at most 2%
    slower, and next turns were sent and start at least 15% sooner with no
    round overlapping the baseline's."""
    groups = defaultdict(list)
    for row in records:
        groups[row["context"]].append(row)
    results = []
    for context, rows in sorted(groups.items()):
        versions = {}
        for version in ("baseline", "candidate"):
            bursts = [row for row in rows if row["version"] == version]
            follow_ups = [turn for row in bursts for turn in row["follow_ups"]]
            versions[version] = {
                "bursts": len(bursts),
                "failures_per_burst": statistics.mean(
                    row["replay_state_publication_failures"] for row in bursts
                ),
                "resumed_follow_ups": sum(turn["resumed"] for turn in follow_ups),
                "follow_ups": len(follow_ups),
                "maximum_context_tokens": min(
                    row["maximum_context_tokens"] for row in bursts
                ),
            }

        def per_round(sample):
            return [
                [
                    value
                    for row in rows
                    if row["round"] == index
                    for value in sample(row)
                ]
                for index in range(len(ROUNDS))
            ]

        decode = abba.compare_samples(
            per_round(lambda row: [row["decode_ms_per_token"]])
        )
        follow_ups = per_round(
            lambda row: [turn["ttft_ms"] for turn in row["follow_ups"]]
        )
        ttft = abba.compare_samples(follow_ups) if all(follow_ups) else None
        baseline, candidate = versions["baseline"], versions["candidate"]
        keep = (
            baseline["failures_per_burst"] > 0
            and candidate["failures_per_burst"] == 0
            and decode["verdict"] != abba.INCONCLUSIVE
            and decode["regression"] <= 0.02
            and ttft is not None
            and ttft["regression"] <= -0.15
            and max(ttft["rounds"][1:3]) < min(ttft["rounds"][0], ttft["rounds"][3])
        )
        results.append(
            {
                "context": context,
                "scenario": "burst",
                **versions,
                "follow_up_ttft_ms": ttft,
                "decode_ms_per_token": decode,
                "keep": keep,
            }
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--contexts", default="2048,10000")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--burst",
        type=int,
        default=0,
        help="measure bursts of this many concurrent requests per context instead",
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help="after each burst, send each conversation's next turn",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=0,
        help="add a long-decode case per sample asking for this many tokens; "
        "a reply ended by EOS counts from 3/4 of it when both versions match "
        "(without --long-decode-self-check), gated on native decode cycle time "
        "per token (0 = off)",
    )
    parser.add_argument(
        "--long-decode-self-check",
        action="store_true",
        help="send sample 0's long-decode prompt in all four rounds and require "
        "each version's two replies to match; the versions may differ only when "
        "every reply ran to exactly --decode-tokens (needs --samples 2 and "
        "--decode-tokens)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/release/http-regression.json")
    )
    args = smoke.resolve_server_arguments(parser.parse_args(argv))
    args.contexts = [int(value) for value in args.contexts.split(",")]
    if args.samples < 2 or not args.contexts or min(args.contexts) < 256:
        parser.error("at least two samples and contexts >= 256 are required")
    if args.burst < 0 or args.burst == 1 or (args.follow_up and not args.burst):
        parser.error("--burst needs two or more requests, and --follow-up a burst")
    if args.request_timeout <= 0:
        parser.error("the request timeout must be positive")
    if args.decode_tokens < 0:
        parser.error("--decode-tokens must be nonnegative")
    if args.long_decode_self_check and (args.samples != 2 or not args.decode_tokens):
        parser.error(
            "--long-decode-self-check requires --samples 2 and --decode-tokens"
        )
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("contexts must be unique")
    # A build that speaks this server's wire version loads the weights into
    # memory; its weight-digests reads the images it loads.
    for binary in (args.baseline_binary, args.binary):
        for path in (binary, binary.parent / "splash.metallib"):
            if not path.is_file():
                parser.error(f"missing retained executable/library: {path}")
        if not weights.loads_in_memory(binary.resolve().parent):
            parser.error(f"{binary.parent} has no {weights.WEIGHT_DIGESTS}")
    if (
        smoke.model_artifacts.installation_kind(args.model_root)
        != smoke.model_artifacts.ASSEMBLY
    ):
        parser.error(f"missing installed model: {args.model_root}")
    return args


def check_identity(status: dict, version: str, rounds: list[dict]):
    """Every round serves the same model and KV format, and a build's rounds
    load the same executable and model layout. The builds' weights are
    compared by their bytes after the rounds (weights.compare)."""
    identity = status["identity"]
    for previous in rounds:
        expected = previous["identity"]
        smoke.require(
            smoke.kv_identity(identity) == smoke.kv_identity(expected),
            "KV identity changed",
        )
        if previous["version"] == version:
            smoke.require(
                identity["cache"]["build_id"] == expected["cache"]["build_id"],
                "executable source changed between rounds",
            )
            smoke.require(
                identity["cache"]["loaded_model_layout_sha256"]
                == expected["cache"]["loaded_model_layout_sha256"],
                f"the {version} loaded another model layout",
            )


def main(argv=None):
    args = parse_args(argv)
    smoke.hold_model_root(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root / "tokenizer", local_files_only=True
    )
    nonce = uuid.uuid4().hex
    # A burst's requests each have a prefix of their own.
    prepared = prompts(
        tokenizer, [128, *args.contexts], args.samples * max(1, args.burst), nonce
    )
    if args.decode_tokens:
        for sample in range(args.samples):
            prepared[sample, "long-decode"] = long_decode_prompt(
                tokenizer, sample, nonce, args.decode_tokens
            )
    binaries = {"baseline": args.baseline_binary, "candidate": args.binary}
    builds = {version: binary.resolve().parent for version, binary in binaries.items()}
    document = {
        "schema_version": 1,
        "timing": "HTTP/native wall; not GPU time",
        "model_root": str(args.model_root.resolve()),
        "rounds": [],
        "samples": [],
        "correctness_pass": False,
        "performance_pass": False,
    }
    try:
        for round_id, version in enumerate(ROUNDS):
            run_args = argparse.Namespace(**vars(args))
            run_args.binary = binaries[version]
            server = smoke.RealServer(run_args)
            try:
                status = server.wait_ready(args.startup_timeout)
                smoke.validate_status(status, args.kv_format)
                check_identity(status, version, document["rounds"])
                document["rounds"].append(
                    {
                        "version": version,
                        "binary": str(run_args.binary.resolve()),
                        "identity": status["identity"],
                    }
                )
                for sample in range(int(round_id >= 2), args.samples, 2):
                    if args.burst:
                        for context in args.contexts:
                            contents = [
                                prepared[sample * args.burst + index, context]
                                for index in range(args.burst)
                            ]
                            row = measure_burst(
                                server,
                                args.model,
                                contents,
                                args.follow_up,
                                args.request_timeout,
                            )
                            row.update(
                                version=version,
                                round=round_id,
                                sample=sample,
                                context=context,
                            )
                            document["samples"].append(row)
                            print(
                                f"{version} sample={sample} burst context={context}",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    cold_outputs = {}
                    for context, scenario, output_tokens in round_cases(args):
                        prompt = case_prompt(
                            prepared,
                            sample,
                            context,
                            scenario,
                            args.long_decode_self_check,
                        )
                        row = measure(
                            server,
                            args.model,
                            prompt,
                            output_tokens,
                            scenario,
                            context,
                            args.request_timeout,
                        )
                        if scenario == "cold":
                            cold_outputs[context] = row["response_sha256"]
                        elif scenario == "exact":
                            smoke.require(
                                row["response_sha256"] == cold_outputs[context],
                                "exact hit changed the cold transcript",
                            )
                        row.update(version=version, round=round_id, sample=sample)
                        document["samples"].append(row)
                        print(
                            f"{version} sample={sample} {scenario} context={context}",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception:
                document["server_tail"] = server.tail()
                raise
            finally:
                server.close()
        if args.burst:
            document["comparison"] = summarize_bursts(document["samples"])
        else:
            if args.long_decode_self_check:
                document["long_decode_self_check"] = long_decode_outcome(
                    document["samples"], args.decode_tokens
                )
            document["comparison"] = summarize(
                document["samples"],
                long_decode_self_check=args.long_decode_self_check,
                decode_tokens=args.decode_tokens,
            )
        # The model root RealServer gives both builds.
        model_root = args.model_root.resolve()
        document["weights"] = weights.compare(
            weights.digests(builds["baseline"], model_root),
            weights.digests(builds["candidate"], model_root),
        )
        smoke.require(
            document["weights"]["pass"],
            f"weight bytes differ: {document['weights']['failures']}",
        )
        document["correctness_pass"] = True
        document["performance_pass"] = all(
            row["keep"] if args.burst else row["pass"] for row in document["comparison"]
        )
    except Exception as error:
        document["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(document["comparison"], indent=2))
    return 0 if document["performance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
