# Validate

```sh
make check
make install test-real test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit
```

`make check` needs no model weights, but it runs the Metal tests, so it needs a
Mac that Splash supports: it runs the source checks, `check-native-cpu`,
`check-native-metal`, `check-native-build`, the Python checks and `check-app`,
the desktop app's tests (`swift test --package-path app`). `make
check-native-cpu` builds production and runs native CPU tests without a GPU;
`make check-native-metal` requires a supported Metal device and runs the kernel
tests under shader validation, and the Linear pipeline resource check without
it, then `test-sanitizers-metal`. They include loading small synthetic MLX,
GGUF and vision sources and the GGUF kernels on synthetic tensors. `make
check-native-build` builds every native test, benchmark and tool and runs
none. Hosted CI runs the source checks, the CPU checks, `check-native-build`
and `check-app` in one job, `test-sanitizers` in another beside it, and `make
check-python` on Python 3.12, 3.13 and 3.14. The Metal suite and the
real-model runs are in the `release-hardware` job, on self-hosted M3 Max
through M5 Max runners, started by hand (`workflow_dispatch` with
`release_gate`, `expect_output_change`, `expect_image_change`). `make
test-sanitizers` runs the CPU tests that start threads, the control plane's
(`engine`, `native-engine-loop`, the cache, protocol and transport tests) among
them, under AddressSanitizer with UndefinedBehavior and under ThreadSanitizer.
`make test-sanitizers-metal` runs the Metal tests of the backend
(`metal-backend`'s `behaviour` tests: its host-step runner shares each
command's steps with the thread that encodes it) and of the
components whose host steps share their state with the engine thread and the
SSD readers (the hot block cache of spilled KV, the history stream and the
expert store) under ThreadSanitizer, built with it in `build/metal-tsan`;
only orders that lie inside Metal are suppressed
(`dev/tests/engine/metal_tsan_suppressions.txt`). `metal-backend`'s
`latency` tests, which bound the runner's times on the wall clock (the GPU's
resume after a step within 10 or 50 ms, the runner's pickup within 25 ms,
its stop within 0.5 s) that ThreadSanitizer's instrumentation lengthens, run
in `test-engine-metal` alone. `make architecture-check`, one of the source checks,
also fails when a workflow under `.github/workflows` names a Hugging Face token
(`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or a `secrets.HF_*` secret): CI installs
public models alone and never receives one.

The native tests are in `dev/tests/engine` and the desktop app's in
`app/Tests`; each Python test is in the directory of what it tests:
`dev/tests/server`, `dev/tests/install` (launcher, client configuration and
model installation) or `dev/tests/tools` (`dev/tools`, `dev/benchmarks`, the
Makefiles and the real-model harnesses).

The real-model targets run against an installation in this checkout's
`install/models`: run `make install` with the same options first, or in the
same command, as above. They take `MODEL` exactly as `splash serve --model`
does, and `REVISION`, `DRAFT_MODEL` and `LANGUAGE_ONLY=1` as its `--revision`,
`--draft-model` and `--language-only`:

| Target | Runs |
| --- | --- |
| `verify-models` | the installer's restarts without the Hub, `verify --full`, and the record of the weight images (`dev/tools/installer_restarts.py`, `weight-digests` with `WEIGHT_DIGESTS_ARGS` added, such as `WEIGHT_DIGESTS_ARGS="--expert-memory 7G --ple-memory 2G"` to stream the routed experts and the PLE table, [Release check](#release-check)) |
| `test-real` | vision parity with the family's fixture in `dev/tests/fixtures/vision-parity/` when the installation serves vision, and the native model runtime oracle, each of its three runs with `ORACLE_ARGS` added (`ORACLE_ARGS="--kv-format bf16"`, `"--expert-memory 16G"` or `"--max-context 1010000"`); for a block-sparse target their spill checks take the default prompts, two staging buffers, and 32K tokens with the default window, sizes that a `--spill-tokens` in `ORACLE_ARGS` replaces in all three (on a 24 GB Mac, `ORACLE_ARGS="--expert-memory 6656M --ple-memory 237M"`; below) |
| `test-http-real` | the HTTP frontend on an isolated server (`dev/tests/smoke_real.py`), whose greedy comparisons (a penalized greedy request repeats once its prompt is cached, `min_p` 1 answers as greedy) run where `/status` reports `verify_rows` `all`, which on any MoE target takes `HTTP_SMOKE_ARGS="--verify-rows all"`; where `/status` reports `reasoning_switch` `false` (Flint), it gives each request whose answer it checks room for the model's reasoning and leaves out the thinking budget's check; with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`, instead of that smoke, the [persistent cache](server.md#persistent-cache) across a restart in a fresh cache directory: a clean stop, a restart that takes the restore point back, and a next turn that restores the prompt from disk; `release-check` runs both |
| `test-agent-real` | the five official clients through `splash serve` (`dev/tests/agent_real.py`), in `AGENT_SCENARIO` `complete` (the default) or `smoke` |
| `test-release-real` | the HTTP smoke and all five clients on one `splash serve` |
| `test-performance-real` | the native decode, partial-prefix and short-prompt benchmark, or with `BASELINE` its ABBA comparison with that build, in which both builds' `weight-digests` take `WEIGHT_DIGESTS_ARGS` (`dev/benchmarks/backend_regression.py`) |
| `test-pressure-real` | the installation on this checkout's own server under `PRESSURE_ARGS` (default `--max-memory 18G`), which must make it stream its routed experts, with the [serving benchmark](#serving-benchmark)'s three churn clients for `PRESSURE_MINUTES` (default 5), checked on its own by [`check.py run`](#serving-benchmark) on G4, G8, G9, Store, Churn (its load window and every client served) and First token; results in `build/release/<model>/pressure` |
| `release-check` | one model on this Mac ([Release check](#release-check)) |

`test-agent-real` and `test-release-real` first run the tests
`dev/tests/install/test_clients.py` has of the installed OpenCode, Codex and Pi
among the selected clients, which need no model.

`test-agent-real` runs Hermes in a profile of its own in the developer's Hermes
root, `splash-test-<id>`, which moves into the run's folder under
`build/release` when Hermes finishes.

Each phase's record keeps what its requests reused of the cache (`reuse`).
Every request after a phase's first resends the conversation, so a phase that
completed two or more requests and reused no cached prompt token fails. Two
phases are exempt: the cancellation phase (`cancel`), and the reference wave
that compacts the conversation, whose summary request and the request after it
share only the system prompt and tools. Replay points of unfinished requests
evicted during a phase only print a warning.

`benchmark-backend` and `benchmark-decode-profile` take `MODEL` the same way.
The models they are run with, one per family and source format:

| Family | MLX | GGUF |
| --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` |

Both source formats load into the block layout of the GGUF projection and MoE
kernels, an MLX target in its MLX formats, a GGUF target in its GGUF ones.

On a 24 GB Mac, `test-agent-real` stops a client's workflow at macOS's warning
memory pressure, which the smaller GGUF variants such a Mac uses can reach under
an agent's load; `SPLASH_TEST_PRESSURE_STOP=4` stops only at critical pressure,
to observe how the engine sheds its cache. The runtime oracle in `test-real` has
no production memory guard: the weights must fit in what macOS has available
above its reserve. When they do not, it measures again every second for up to
30 s before it stops, as a start waits for memory macOS refuses it: macOS can
return the memory of a process that has just exited a moment later (on a 36 GB
M5 Max, a server engine's memory came back within a second of its exit).
Once it has loaded, its checks can neither wait for memory nor yield it, so the
governor a start builds admits all the runtime then allocates, its arenas and
the lanes, KV and cached states of the requests the oracle runs together, as
memory a request in service needs: only the memory plan's budget and macOS's
critical pressure refuse it, and the run stops at critical pressure, naming
what it asked for. Production grants that only to its start, a request it
serves alone and the KV of the requests it serves; a start beside a resident
request, and the cache, are held to the spare reserve, the 1 GiB margin above
it and macOS's warning. A Neural Engine split leaves the plan room for what the
checks hold at once (every lane's state, six cached states and the oracle's KV
pages), as a start's split leaves `--max-context`, so the oracle's split may
take fewer of the Neural Engine's units than a server's does. With only desktop
applications open, a 24 GB Mac runs it for those variants. A refusal of the
budget fails its check with its cause, the bytes asked for and the governor's
limit and charge. A block-sparse target's spill check counts what it allocates
at most before its first prompt, and where the budget leaves less room, stops
there, naming both, an `--expert-memory` that fits where that store still
loads, and a `--spill-tokens` that fits for the oracle's command run by hand:
`ORACLE_ARGS` reaches all three runs of `test-real`, and a `--spill-tokens`
there replaces the first two runs' default prompts and the third's 32K.
Beside the weights, Qwen3.8-Flash-Next's default checks hold up to 4.3 GB and
its 32K check about 5.9 GB. On the 24 GB M6 (11-02) a check's need as the
oracle names it (up to 1866 MiB by default, 3382 MiB at 32K) and the expert
store together fit in about 10440 MiB on a text installation and 9580 MiB on
one that serves vision, so all three runs take
`ORACLE_ARGS="--expert-memory 6656M --ple-memory 237M"`, or
`"--expert-memory 5632M --ple-memory 237M"` where the installation serves
vision, whose default checks alone fit only up to about `7700M` of experts.
Unlike `splash serve`, which
streams the PLE table where memory cannot hold it, the oracle loads it whole
unless `--ple-memory` streams it, and its weights then need 37555 MiB, which
it refuses at once on the M6 (11-03); `237M` is the row cache `splash serve`
gives the table there, with which the 32K check passed.

The engine's `--kv-spill always`, which `splash serve` never passes, keeps
the KV of every request of a block-sparse target on the SSD cache
([KV on the SSD](server.md#kv-on-the-ssd)), so that a run of `build/splash
serve-native` with it compares spilled requests' outputs with those of a run
without it, which must match bit for bit.

Compare performance on the same idle Mac with the same model and workload.
Keep generated reports, profiles, local paths and experiment notes out of the
source tree and commits.

## Release check

A release is checked once per source identity, and then on each Apple GPU
family (an Apple9 M3 and an Apple10 M5) against a retained baseline build,
`BASELINE`: a checkout whose `build/` holds `splash`, `splash.metallib`,
`engine-tests/backend-benchmark` and `engine-tests/weight-digests`.
`release-check` fails without it. The baseline must load the model: it is the previous release's
build when that loads the model. From a clean checkout:

```sh
make check test-sanitizers                      # once, model-free, on one Mac
make check-native-metal                         # on the other Mac
make install release-check MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit> BASELINE=../splash-baseline
make install release-check MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M LANGUAGE_ONLY=1 REVISION=<commit> BASELINE=...
make install release-check MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> BASELINE=...
make install release-check MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> BASELINE=...
make install test-pressure-real MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> PRESSURE_ARGS="--max-memory 18G"
make install verify-models MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
make test-agent-real MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> AGENT_SCENARIO=smoke AGENT_CLIENTS=...
```

Pin each upstream model to one commit, the same on both Macs. Without a Hub
token for a private draft repository, set `DRAFT_MODEL` to a local copy of the
draft's checkpoint ([Drafts](models.md#drafts)). The Metal suite depends on the GPU
family, so it runs once on each Mac: within `make check` on the first, alone on
the other. Per model, `release-check`:

- runs the runtime oracle and, when the installation serves vision, vision
  parity (`test-real`);
- restarts the installer offline, with the Hub unreachable
  (`HF_ENDPOINT=http://127.0.0.1:9`) and with an empty `HF_HUB_CACHE`: each
  restart must start the same assembly within 10 seconds, name the Hub's
  reason on one line when it asked the Hub ([Revisions](server.md#revisions)), and
  download nothing. It then hashes the
  sources and records in `weights.json` the component, size and SHA-256 of
  every weight image the installation loads (`verify-models`);
- runs the HTTP smoke, which for a text-only installation checks the 400s
  instead of images (`test-http-real`), on the default verify rows: on a MoE
  target it leaves out the greedy comparisons, which run only with
  `HTTP_SMOKE_ARGS="--verify-rows all"`, and says so;
- stops and restarts a server with a [persistent cache](server.md#persistent-cache),
  and requires the conversation's next turn to restore its prompt from disk
  (`test-http-real` with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`);
- compares this build with `BASELINE`, which must have another build
  identity, in ABBA order (`test-performance-real`), every round after the
  first to report a [Neural Engine share](internals.md#neural-engine-prefill) running that
  share: output tokens and acceptance must be
  identical (`EXPECT_OUTPUT_CHANGE=1` allows changed outputs with acceptance
  within 0.02; the benchmark verifies every row (`--verify-rows all`), so a
  MoE target's outputs repeat and its times measure the backend, not the
  verify budget of [Drafts](models.md#drafts)), and so must the bytes of the weight images
  both load (`dev/benchmarks/weights.py`); decode and prefill GPU time, with the cold
  prefills of short prompts (480 to 2016 tokens, one chunk each), and their wall
  times (decode milliseconds per output token, the cold 14K prefill's wall
  time and the short prefills' TTFT), which hold the host work between
  commands, may regress by at most the larger of 2% and
  twice the run's own ABBA spread, a spread above 5% fails as inconclusive,
  and this build's
  memory plan must hold, without the Neural Engine split, at least the
  context the baseline's holds without it. `ANE_FFN_SHARE=SHARE` runs
  every round at that share instead (0: the GPU alone).

Where the Mac's memory cannot hold a model's weight images, or its largest
Metal buffer cannot hold one of them, as for Qwen3.8-Flash-Next,
`WEIGHT_DIGESTS_ARGS` streams its routed experts and PLE table in the loads
whose images are recorded and compared ([streamed experts](server.md#streamed-experts),
[PLE table streaming](server.md#ple-table-streaming)): it gives `weight-digests` the
`--expert-memory SIZE` and `--ple-memory SIZE` that `ORACLE_ARGS` gives the
runtime oracle, each SIZE a whole number of bytes or of `K`, `M` or `G`, such
as `7G`, in `verify-models` and in both builds of the comparison with
`BASELINE`. The layers' images then
leave out their routed experts, and a PLE table that streams has no image
(`ple.bin`): the engine reads those bytes from the source files, which
`verify --full` hashes, and the digests do not cover them. Any size a store
takes records the same images. Where the images and stores of a load exceed
the engine's Metal memory budget, `weight-digests` refuses it before loading
any image and names the options. An expert store is SIZE and stays so, all
of it its floor, in a tool (`dev/tools/LoadOptions.hpp`) as in `splash
serve`: an HTTP smoke streams the experts with `--expert-memory` alone in
`HTTP_SMOKE_ARGS`.

Results go to `build/release/<owner>--<repo>[--VARIANT]/`. The weight images
do not depend on the GPU, so each model's `weights.json`, recorded with the
same `WEIGHT_DIGESTS_ARGS`, must be identical on the two Macs. The unpinned
`verify-models`, run after the pinned ones while the default branch still
names the pinned commit, resolves the branch online, and its unreachable-Hub
restart must fall back with the Hub's reason. The agent
clients depend on neither the model's format nor the GPU: run the smoke
scenario once per Mac, with the five clients split between the Macs, and
`AGENT_SCENARIO=complete` for one model when the client integration changed.
Expect about 1.5 hours on an M5 Pro and 2.5 hours on an M3 Max, most of it in
the three 27B comparisons. A laptop can cap its GPU power during a long
comparison and so make it inconclusive; rerun `make test-performance-real` for
that model alone once the Mac has cooled.

`.venv/bin/python dev/tools/kernel_identity.py BASELINE/build build` compares
the AIR of each kernel the decode path may run in the two builds'
`splash.metallib` (`--match` for others) and fails if any differs. The helpers
a kernel's module defines are compared in the order they appear there, not by
name: C++ names them for their template values, which a kernel template that
gains a value renames without changing their code. So are the numbers of its
lambdas' anonymous classes and its classes' names in the alias metadata, which
the same change renames, and the numbers LLVM gives classes whose names clash,
which count every such class compiled before the kernel.

Two pairs of constants in `runtime/engine/Protocol.hpp` and `server/protocol.py`
version what the server and the engine exchange. When the native wire layout
changed since the last release, bump `kProtocolVersion` and `PROTOCOL_VERSION`
together; each side refuses frames of another version. When a field of the
status document the engine writes was removed or changed meaning, bump
`kStatusSchemaVersion` and `STATUS_SCHEMA_VERSION` together; the server refuses
a status document of another schema. An added field keeps the version, as
`weights` did in 1.2.1 and `ane_ffn` in 1.3.0. Builds between releases share a
version while its layout changes.

## Local benchmarks

From a source checkout with the model installed, use the native benchmark for
prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit BASELINE=/path/to/baseline
```

The first characterizes this build: the decode widths B1-B4, a 14,096-token
partial-prefix request and the cold prefills of short prompts, three samples
each, in
`build/release/<owner>--<repo>[--VARIANT]/backend-benchmark.json`;
`make benchmark-backend MODEL=...` adds the 2K to 128K contexts the memory
plan holds. The second compares this build with a retained checkout's in ABBA
order, as the release check does ([Release check](#release-check)), and writes
`backend-regression.json` there. Neither is a comparison with another engine
or a test of agent task quality.

A change of arithmetic that moves outputs (`EXPECT_OUTPUT_CHANGE=1`) is
measured by its next-token distributions instead. The benchmark's `logits`
scenario dumps the full-vocabulary logits after 16 prefixes of each of the six
transcript requests' 256-token greedy answers, the score-only requests cold and
`--logits-width` at once so that they prefill beside each other, and
`dev/benchmarks/logit_compare.py` compares a dump with one whose prefixes it
scored (`--logits-from`): top-1 agreement, mean and p99 KL and the next-token
NLL change, failing past `--min-top1`, `--max-mean-kl` and `--max-p99-kl`. A
dump is three files, `sequences.txt`, `records.txt` and `logits.f32` (fp32
rows, little-endian as numpy's tofile writes them), so that another
engine's logits after the same prefixes compare the same way:

```sh
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/base
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/b4 --logits-from /tmp/base --logits-width 4
.venv/bin/python -m dev.benchmarks.logit_compare /tmp/base /tmp/b4
```

The second line also measures this build's noise floor: the same prefixes
prefilled four at a time.

A change of the weight images' layout (`EXPECT_IMAGE_CHANGE=1`) makes
`backend_regression` report the images it changed instead of failing on them.
`decode-profile --dispatch-list FILE` writes the profiled B1 cycles'
dispatches in order with their GPU time.

To compare the model's numerics with another engine on the same weights,
`make teacher-logits MODEL=... TEACHER_LOGITS_ARGS='TOKENS OUTPUT --window END:ROWS ...'`
writes the installed model's teacher-forced next-token logits over windows of
a file of uint32 token ids (`dev/benchmarks/teacher_logits.mm`): for each
window, the fp32 logits of the prefill's final position and of ROWS decode
positions after it, each computed by a verify row forced to the true next
token, in `OUTPUT-END.f32`, and prints how well they predict the true ids. It
also takes `--kv-format`, `--state-format`, `--expert-memory` and
`--ple-memory`, and, with `--captures yes`, writes the output rows of the
target layers the model's draft reads (its `target_layer_ids`) at the
positions a window's prefill computed, in one command, to
`OUTPUT-END.captures.f32` (fp32, a row per position of each layer's hidden
values in the draft's layer order), which a reference's residuals after those
layers compare with (`dev/tools/flint_reference.py` writes Flint's,
`residuals.npy`): a draft of the layers to compare, such as a test draft whose
`fc` reads them, makes them its captures.

`benchmark-backend` lists the contexts the memory plan cannot hold in its
report. Its cache checks reuse each context's cached prefix, so they need that
memory free: when other programs leave too little, the engine evicts cached
prefixes and the checks fail, naming what each lookup found.

## Serving benchmark

`dev/benchmarks/serving` measures a server under concurrent requests and
compares builds of different trees, such as a release candidate with the last
release, on one Mac. Each run starts the tree's own server, so every tree is
measured through the same OpenAI API whatever its native wire and status
schema, and the client times what a user sees.

| Scenario | Requests |
| --- | --- |
| `churn` | three clients, each sending 1500-token answers to prompts that grow every round (about 16K tokens, then more; `CHURN_BASE`/`CHURN_STEP` words), every fourth round resending its conversation, the first requests in a fixed arrival order (`--order 2,0,1` or `0,1,2`, 0.6 s apart), for `--minutes`; `/status` every 2 s |
| `ttft` | for each N of `--n`, on a fresh server: N conversations cache an ~8K-token prefix one at a time, then send it with distinct ~300-token suffixes at the same instant, `--rounds` times, each round recording its `/status` counters' changes and the largest `memory_governor.claims` its samples showed; `/status` every 0.5 s |
| `serve1` | one request at a time on an idle server, cold prompts of about `--sizes` tokens (300, 2K and 14K): time to first token and decode milliseconds per token; the `/status` each request waited idle for, and one after the last |
| `warm` | per cached depth of `--sizes`, a fresh cold prefix, then 2 discarded and 8 measured ~300-token suffixes after it: the server's prompt time of each (`--trial` picks the texts, the same for every tree); `/status` as `serve1` |

Every request timestamps each content chunk of its stream; its gaps are the
intervals between consecutive chunks after the first. Each request may take
`--request-seconds` from its send (by default twice the longest request
measured on any Mac). A request the client gets no whole answer to is
unanswered, the server's failure: one the server keeps streaming past that
time, though its keepalives hold the connection open, or one whose connection
fails or closes before the stream's `data: [DONE]`. It ends `serve1` and
`warm`, ends `ttft` as one of its warm-ups and fails its round otherwise; in
`churn` its client records it, with its chunk times and why, and stops.

```sh
.venv/bin/python -m dev.benchmarks.serving.run --tree TREE --machine m6 \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision <commit> \
  --draft-model DRAFT --scenario churn --order 2,0,1 --minutes 15 --label LABEL --out RESULTS
```

`run.py` installs the model with the tree's own installer, offline, or takes
`--model-root`; refuses a port that already answers; checks that `/status`
names the server it started (`instance.pid`) and the build of the tree's
sources (`identity.cache.build_id` against the tree's own
`dev/tools/build_identity.py check` with its Makefile's constants); runs the
scenario; and stops the server with SIGINT, or, when it outlasts that, SIGTERM
after 60 s and SIGKILL after 30 s more. Beside the client's
`LABEL.events.jsonl` and `LABEL.status.jsonl` it writes `LABEL.out`, whose last
line is `DONE` only once everything completed, `LABEL.meta.json` (binary
SHA-256, build id, model, spawn-to-ready seconds and whether the binary had
started on that Mac before, host memory and pressure, process snapshots at
the start, middle and end of the scenario while the server serves, the pids
the run started, each server's stop, the signals it took and its exit code,
and for a run that did not complete why and whose failure
that was: the server's, which exited, was not ready in time, failed a
request or, once ready, did not let the scenario end, a client's, which
raised an exception of its own, or `run.py`'s refusal to measure) and on the
M5 Pro
`LABEL.idle.jsonl`, its idle time and sleep assertions every 10 s. It refuses any output directory that resolves,
through whatever symlinks, into `~/.cache/huggingface` or an `install/models`
directory.

`check.py` judges runs:

- `run RESULTS LABEL [--minutes N] [--streamed] [--build BUILD_ID]` checks one
  run on its own and exits 1 on any failure; `test-pressure-real` runs it on
  its churn run.
- `compare A1 B1 B2 A2`, each run as `RESULTS/LABEL`, compares two builds on
  four runs in ABBA order on one Mac: each run must pass its own check on a
  quiet machine, the two A runs and the two B runs must each be one build,
  and every per-request metric the four share (churn's TTFT, completion latency and
  decode milliseconds per token, ttft's TTFT per concurrency, serve1's TTFT
  and decode per size, warm's prompt time per depth) must pass the release check's speed rule
  (`dev/benchmarks/abba.py`).

A run measured nothing and is to be repeated when it did not end with `DONE`
for a reason other than the server's, served a build other than `--build`,
or, with `--streamed`, did not start with its routed experts streamed. Only a
timed comparison needs a quiet machine: `compare` also fails a run that saw a
`serious` or `critical` thermal sample, found the GPU still busy after its wait
for idle (`--gpu-idle-percent`), on the M5 Pro was touched within a 10 s
sampling period or saw a sleep assertion held by a process it did not start
other than `powerd`, or had a process it did not start among the top five by
CPU at 24 % or more. `run`, whose gates judge freezes and correctness, which
these do not excuse, notes them.

| Gate | Rule |
| --- | --- |
| G4 | the client gaps between token events (chunks less than half the shortest decode command apart are one event's; an unanswered request counts its open gap, up to when the client stopped waiting): the largest at most 10 s, the p99 at most 5 s; every gap past 2.5 s is listed |
| G8 | no engine restart, failed request (an HTTP error but `context_length_exceeded`, an error event in a stream the server had begun, a request of a failed TTFT round, or one `/status` counts in `requests.failed`), unanswered request, Metal failure or loop tick of 5 s or more, no exit, late start or failed request that ended the run, and every server stopped by SIGINT alone with exit 0, the engine it started gone within SIGINT's grace too |
| G9 | the memory accounts sum to `charged_bytes` in every sample, no claim waits 30 s, nothing is owed at the end |
| Store | a regrown expert unit taken back within 60 s of its regrow is counted; a second such take back within 60 s of the one before is an oscillation and fails |
| Churn | the run lasted its load window (`--minutes`) and every client completed a request |
| First token | no answered request (G8 judges unanswered ones) waited for its first token more than twice the prefill time, at the run's own rate (the tokens its answered requests prefilled over the time from the first send to the last answer), of the work the scheduler may put ahead of it: the requests in flight at its send, 24,576 rows of later arrivals (three prefill commands of at most 8,192 rows, the most that may pass over it in a row) and its own prompt, each less its cached tokens; a starvation guard, not a speed check |

G8 applies to every scenario, the others to churn runs. The check also counts
the client gaps past the decode gap target of 500 ms by cause, each taking the
first of these that the `/status` samples from 2 s before it to 2 s after it
show: a claim waiting for memory, the experts' size changing, a compaction, a
replan, a history-stream wait, a committed prefill command; else it is
unattributed.

Copy a tree to another Mac by content (`rsync --checksum`): `git archive`
leaves `dev/tests` and `dev/benchmarks` out (`.gitattributes`), and a copy
that keeps the files' old times leaves make running stale binaries. For a
gate, delete the test binary before building it and record the commit, an
empty `git status --porcelain`, the binary's SHA-256 and its test count.

## HTTP regression check

checkout, after
`make build/engine-tests/weight-digests`:

```sh
.venv/bin/python -m dev.benchmarks.http_regression \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M \
  --baseline-binary /path/to/baseline/build/splash \
  --contexts 2048,10000 --samples 5
```

A baseline built inside a checkout (`<checkout>/build/splash`) runs that
checkout's Python server, so builds on either side of a protocol change each
speak their own wire format; a worktree is enough. A bare retained `splash`
outside a checkout runs this checkout's server, which works only while the
baseline speaks the current wire version and status schema.

It takes any installed model and, for an upstream one, holds its assembly for
the whole run, so every round serves the same model. For a MoE target, pass
`--verify-rows all`: the check requires the two builds' transcripts to match,
and by default the verify budget chooses a MoE target's rows from cycle wall
times, so its greedy output can differ between runs (see [drafts](models.md#drafts)).
Both rounds pass the option to their engine, so a MoE comparison needs a
baseline built from a checkout that takes it; an older baseline cannot run one.
It starts isolated servers in ABBA order, compares matched cold, exact-prefix
and decode requests by the release check's speed rule and weight bytes (decode
by `metrics.decode_cycle_ms` per output token, so host work between commands
counts), and saves
`build/release/http-regression.json`. With `--burst N` it sends N requests of
each context at once instead, 16 output tokens each, and reports each build's
replay points lost to the burst (`replay_state_publication_failures`), its
decode rate and advertised context. `--follow-up`, which needs `--burst`, then
sends each conversation's next turn and compares their time to first token and
how many resumed at their replay point; a burst run passes only with it. It does
not contact your running server. Use the same power mode and charger, stop other
GPU workloads, and report chip/GPU cores, memory, Splash version, model
revision, actual input/output token counts, and cache hits with results. Keep
cold prefill, cached TTFT and sustained decode separate; a UI token rate alone
does not measure end-to-end agent performance.

`--decode-tokens N` adds per sample one long-decode request: a short prompt that
asks for more than N tokens with `max_completion_tokens = N`, gated like the
decode request on native decode time per token, so `--samples S` makes 2·S
replies of up to N tokens. The model may end the count by EOS first (seen at
27.5k-29.6k tokens), so such a reply counts when it reaches at least 3/4 of N
and, without `--long-decode-self-check`, the baseline and candidate replies for
that sample match in text and length; each is timed over its own length. Its rows keep the response's `metrics` (including `metrics.engine`
when the server reports it) plus `finish_reason` and the requested `output_tokens`;
without the flag the report is unchanged.

`--long-decode-self-check` (with `--samples 2` and `--decode-tokens N`) sends
sample 0's long-decode prompt in all four rounds, so each build answers it
twice. Correctness then requires each build's two replies to be identical (reply
sha256 and token count); the builds may differ from each other only when every
long-decode reply ran to exactly N tokens on length, and an EOS-ended reply must
still match across builds and reach the 3/4 floor. The decode-time gate is
unchanged, and the report's `long_decode_self_check` object records each build's
`self_match`, `builds_diverged` and `length_capped`. Use it when comparing
builds that group verify rows differently: greedy speculative decoding is
deterministic per configuration, but a different verify row grouping, like a
different batch width, changes the GPU's reduction grouping and can flip a
greedy near-tie between two almost equal logits, after which the texts diverge.

## Live server probes

To measure the server you are actually running (for example on the target
machine), probe it over HTTP:

```sh
.venv/bin/python -m dev.benchmarks.live_probe --base-url http://127.0.0.1:8000 \
  --prompt-tokens 8192 --max-tokens 256
```

It sends one cold prompt (a random nonce leads, so nothing is cached) and then
its exact repeat, and prints per request prefill and decode tok/s, draft
acceptance, average prefill chunk rows, cache matched tokens, resource
suspensions, replay tokens and denied memory reservations, each diffed from
`GET /status` snapshots taken just before and after the request, then the
memory governor's host available, inactive, wired and compressor bytes, read
from the after-snapshot alone, and the request's HTTP wall time. It never
starts a server. `--api-key` defaults to `SPLASH_API_KEY`, `--model` to the
first model `/v1/models` lists, and `--json` prints the records as JSON. The
counters are server-wide, so probe an otherwise idle server; a rate whose
native wall time did not move prints as `unavailable`. Requests ask for
`reasoning_effort: none`; a chat template that cannot turn thinking off is
probed with its default effort instead, with a note on stderr. Each record
also carries the reply's `finish_reason` and `content`, printed below the table
(or as record fields with `--json`). `--prompt-file PATH` sends that file's
text as the user message in one cold request with no repeat, ignoring
`--prompt-tokens`; use it to replay a fixed prompt, such as a long-context
reproduction.

`--stop-sequence [S]` instead streams one greedy counting reply ("Count from 1
to 60, separated by commas.") with `stop: [S]` (default `, 20,`; S must occur
past the start of `1, 2, ..., 60`) and adds two rows: `stop to done ms`, the
arrival of the finish chunk minus that of the last content delta (it includes
decoding the stop text the server held back), and `cycle ms`, the request's
`GET /requests` `engine.decode_wall_ms` over its `draft_cycles`, or over
`engine.decode_steps` without the draft. That record lands after native
completion, which can follow the client's done, so `/requests` is polled for up
to 5 s; a record still missing leaves `cycle ms` `unavailable` with a note on
stderr. The stop record's `/status`-diffed rows are `unavailable`. The probe
exits non-zero unless the finish reason is `stop`, `usage.completion_tokens` is
below `--max-tokens` and the content ends just before the stop text: an EOS
finish reports `stop` too. It cannot be combined with `--prompt-file`.

To see what a long cold prefill costs a request that is already decoding, run
the two-lane contended probe against the same idle server, once per
`--contended-prefill-rows` floor you compare:

```sh
.venv/bin/python -m dev.benchmarks.contended_probe --base-url http://127.0.0.1:8000 \
  --prompt-tokens 16384 --decode-tokens 4096
```

It streams one long decode and, once its first token arrives, a cold prompt
beside it. Over the cold request's prefill (from sending it to its first token)
it prints the decode's client-side inter-chunk gap p50/p95/max, the cold TTFT,
and the native prefill tok/s and average prefill chunk rows diffed from
`/status`. A decode that ends before the cold prefill does is flagged on stderr;
raise `--decode-tokens` then. `--api-key`, `--model` and `--json` work as for
`live_probe`.

To replay real agent traffic, record it with `splash serve --record-requests
<file>` (keep the file under `build/release/`: it holds real request content),
then list its conversations and replay one against an idle server:

```sh
.venv/bin/python -m dev.benchmarks.agent_replay chains --recording build/release/agent.jsonl
.venv/bin/python -m dev.benchmarks.agent_replay run --recording build/release/agent.jsonl \
  --chain 0 --base-url http://127.0.0.1:8000 --label head --output head-1.json \
  --max-tokens 2048
.venv/bin/python -m dev.benchmarks.agent_replay summarize head-1.json head-2.json
```

`chains` lists each chain (a run of requests on one path, each extending the
previous history; a leading system message rewritten after at least two equal
conversation messages, or a suffix cut from the previous request's last
message, does not break it; a request under another system message that a
later request skips, continuing the history before it under the unchanged
system message, is a side request and starts its own chain with any
follow-ups to it) with its turns,
final prompt
size (messages, system field and tools) and estimated tokens, largest first.
`run` streams the chain's turns serially at temperature 0, every other
field as recorded and `--max-tokens` capping the completion limit; `--turns N`
replays only turns 1..N. The recorded next turn is sent whatever the model
replied. A per-run nonce line leads each
turn's rendered prompt, so turn 1 is cold and runs never share cache. Per turn
it writes prompt, cached and completion tokens, client TTFT, prefill wall and
uncached-tail prefill tok/s, decode tok/s, drafted and accepted tokens, and
`prefix_lost` (cached tokens more than 1024 below the previous turn's prompt).
It fails when `/status` counts another admission during a turn. `summarize`
prints per label the per-turn medians and whole-chain totals, refusing runs of
different chains or of different turn counts.
