# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a registered `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint. With `--logits-out` it
also writes the full next-token distributions of the scored positions in the llama.cpp
`--kl-divergence-base` format, for KL-divergence comparisons against another artifact or a
llama.cpp reference (see [Logits dumps and KL divergence](#logits-dumps-and-kl-divergence)).

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

## Logits dumps and KL divergence

Perplexity alone hides where two models disagree. A weight-format change is judged by the
KL divergence between the next-token distributions of a reference model and the candidate, the
agreement of their top-1 tokens, and their perplexities, all on identical token ids.

### Protocol and dump format

`--logits-out FILE` switches the window plan to the llama.cpp `llama-perplexity` KL-divergence
protocol so that NInfer and llama.cpp dumps are position-for-position comparable. It requires
`--text` (one token stream) and a `--stride` of `context/2` (the default for the default context):

- the stream is cut into `floor(tokens/context)` non-overlapping windows of exactly `context`
  tokens; the tail after the last full window is not evaluated;
- each window starts from empty State and KV and scores only its last `context-1-context/2`
  targets, so every scored token sees between `context/2+1` and `context-1` tokens of history;
- with `--context 4096` that is 2,047 scored positions per window;
- `--chunks N` keeps the first N windows, like `llama-perplexity --chunks N`.

The file layout is the one written by `llama-perplexity --kl-divergence-base`, so one reader handles
both engines and `llama-perplexity --kl-divergence` can also use an NInfer dump as its base
(little-endian):

```text
char[8] "_logits_" | u32 context | i32 vocab | i32 chunks | i32 tokens[chunks*context]
per scored position: f32 scale | f32 min_log_prob | u16 q[vocab] | u16 pad (odd vocab only)
log p(i) = min_log_prob + scale*q[i]
```

Every position stores its full log-softmax with a per-position affine 16-bit code over the 16 nats
below its most likely token (a step of 0.00024 nat); less likely tokens encode as the window floor.
NInfer writes the output-head row width (248,320 for Qwen3.8, the GGUF vocabulary size) and
normalizes over the 248,077-token domain; padding rows encode as the floor and carry no mass. The
dump holds the evaluated token ids, so a comparison verifies tokenization equality. A dump is
written as `FILE.partial` and renamed only when every position is present; `report.json` records it
under `logits_dump`. The partial file is created before scoring, so a missing or unwritable
directory fails immediately, and a `FILE.partial` left by an interrupted run is refused rather than
overwritten. For 32 windows of 4,096 tokens (65,504 positions) a dump is 32.5 GB.

`--logits-reference REF` reads the token block of an existing dump and aborts before scoring if the
artifact tokenizer produced different ids, a different chunk count, or a different context. With
`--chunks N` the reference may hold more windows: the run must match its first N windows, the
prefix that `kld.py compare --chunks N` compares. It also rejects a logits row width that differs
from the reference vocabulary.

Because of the 16-nat window, a target less likely than `max_prob*e^-16` is counted at the floor, so
perplexity recomputed from dumps slightly under-counts very surprising tokens (llama.cpp's
`PPL(base)` has the same bias). Each run also reports the unclamped perplexity of the same positions:
`overall.perplexity` in NInfer's `report.json`, `Final estimate: PPL` in the llama.cpp log.

### Comparison and gate tools

`tools/kld/kld.py` (Python 3.10+, NumPy) reads dumps from either engine:

- `inspect FILE...` validates the size implied by the header and prints the shape and token hash;
- `compare --reference REF --candidate NAME=FILE ...` checks that context, chunks, vocabulary and
  every token id match, then reports per candidate the mean (with standard error), median, p90,
  p95, p99, p99.9 and max KLD, the top-1 agreement, the reference and candidate perplexities and
  their delta, and the target-probability difference. `--segments` adds the same statistics per
  corpus domain, `--exact-ppl NAME=SOURCE` attaches the unclamped perplexity of a run (it must
  cover the compared windows: `--chunks` needs the perplexity of the same prefix), `--json`
  writes the machine-readable result;
- `gate --results JSON... --candidate NAME --prod NAME --ceiling NAME [--previous NAME]` applies the
  quantization gate and exits 0 on pass, 1 on failure, 2 on unusable input.

KLD per position is llama.cpp's definition, `sum_i p_ref(i) (log p_ref(i) - log p_cand(i))` over
tokens with `log p_ref(i) > -16`. llama.cpp evaluates the candidate from unquantized logits, here
both sides are dumps; on Qwen3.5-9B with a Q8_0 KV cache against F16 KV (4,094 positions) the reader
matched `llama-perplexity --kl-divergence` within 0.1% on median, p99 and p99.9 KLD, 1.2% (0.00015 nat)
on the mean, which is dominated by rare positions where the candidate falls below its 16-nat window,
and 0.03 point on top-1 agreement (quantization ties), with identical reference perplexity and
target-probability RMS.

Gate limits, per conversion stage (`--previous` is the preceding stage, the production artifact by
default); each limit has a command-line override:

| Check | Limit |
|---|---|
| mean KLD increase over the previous stage | at most +0.005 nat |
| mean KLD increase over production | at most +0.015 nat |
| mean KLD | below the `--ceiling` run (llama.cpp Q4_K_M) |
| p99 KLD | at most 2x production |
| top-1 agreement drop from the previous stage | at most 1 percentage point |
| perplexity increase over the previous stage | at most +1% (unclamped when both runs provide it) |

The gate refuses results computed against different references or corpora.

### Corpus

`eval/corpora/kld-v1/manifest.json` defines `ninfer-kld-v1`: the four `--quick` streams of
`ninfer-ppl-1m-v1` plus an original French text and an original chat/reasoning transcript, cut to
token budgets that fill 32 windows of 4,096 tokens (131,072 evaluated tokens, 65,504 scored positions).
The budgets are interleaved in four rounds, so `--chunks 8` on every run (16,376 scored positions)
is a proportional sample of all six domains and a prefix of the full comparison.
`tools/kld/build_corpus.py` builds it with the reference GGUF tokenizer through llama.cpp's
`llama-tokenize`:

```bash
python3 -m tools.kld.build_corpus \
  --llama-tokenize /path/to/llama.cpp/build/bin/llama-tokenize \
  --model /path/to/Qwen3.8-27B-Q8_0.gguf \
  --out out/kld/corpus-v1
```

It writes `corpus.txt` (no trailing newline, no special-token markers), `tokens.bin` (int32 ids)
and `segments.json` (token range of each source). Two llama.cpp defaults would silently change the
token stream and must be overridden: `-f` text goes through escape processing (`\n` in the code
stream would become a newline) unless `--no-escape` is given, and a trailing newline is stripped.
Control-token strings such as `<|im_start|>` are parsed by the NInfer tokenizer but not by
`llama-perplexity`, which is why the corpus contains none. The Qwen3.8 GGUF adds no BOS token.

### Runbook

Commands assume a llama.cpp build with `llama-perplexity` and `llama-tokenize`, a NInfer build with
this `ninfer-perplexity`, and the corpus above. `$LLAMA` is the llama.cpp `bin` directory
(also exported as `LD_LIBRARY_PATH` for shared builds), `$NINFER` the NInfer build directory and
`$K` the working directory (about 33 GB per dump).

1. Reference dump, llama.cpp Q8_0 (or BF16) of the same checkpoint:

   ```bash
   $LLAMA/llama-perplexity -m Qwen3.8-27B-Q8_0.gguf -f $K/corpus-v1/corpus.txt --no-escape \
     -c 4096 -b 2048 -ngl 99 -fa on --kl-divergence-base $K/dumps/q8_0.kld 2>&1 | tee $K/logs/q8_0.log
   ```

   A BF16 GGUF of the 27B model (55 GB) does not fit one 32 GB GPU; with two visible GPUs add
   `-sm layer` and it runs at about 290 tok/s on two V100s, versus about 26 tok/s on a 36-core
   AVX-512 CPU (`-ngl 0`). CPU and GPU BF16 dumps of the same corpus differ by a mean KLD of
   about 4e-5 nat.

2. Ceiling dump, llama.cpp Q4_K_M of the same checkpoint, identical options:

   ```bash
   $LLAMA/llama-perplexity -m Qwen3.8-27B-Q4_K_M.gguf -f $K/corpus-v1/corpus.txt --no-escape \
     -c 4096 -b 2048 -ngl 99 -fa on --kl-divergence-base $K/dumps/q4_k_m.kld 2>&1 | tee $K/logs/q4_k_m.log
   ```

3. NInfer dumps, production artifact first, then every candidate artifact with the same KV dtype:

   ```bash
   $NINFER/apps/ninfer-perplexity qwen3_8_27b.ninfer --text $K/corpus-v1/corpus.txt \
     --kv-dtype int8 --logits-out $K/dumps/ninfer-prod.kld \
     --logits-reference $K/dumps/q8_0.kld --output $K/reports/ninfer-prod
   ```

4. Compare every run against the reference in one pass over the reference dump, then gate a
   candidate:

   ```bash
   python3 -m tools.kld.kld compare --reference $K/dumps/q8_0.kld \
     --segments $K/corpus-v1/segments.json \
     --candidate q4_k_m=$K/dumps/q4_k_m.kld --candidate prod=$K/dumps/ninfer-prod.kld \
     --candidate stage1=$K/dumps/ninfer-stage1.kld \
     --exact-ppl reference=$K/logs/q8_0.log --exact-ppl q4_k_m=$K/logs/q4_k_m.log \
     --exact-ppl prod=$K/reports/ninfer-prod/report.json \
     --exact-ppl stage1=$K/reports/ninfer-stage1/report.json \
     --json $K/results/compare-stage1.json

   python3 -m tools.kld.kld gate --results $K/results/compare-stage1.json \
     --prod prod --ceiling q4_k_m --candidate stage1 --json $K/results/gate-stage1.json
   ```

   For a later stage, add its dump to a compare (a previous compare JSON against the same reference
   can be passed to `gate --results` alongside) and pass `--previous stage1 --candidate stage2`.
   A comparison of three 32.5 GB candidates takes a few minutes with 16 worker processes; it is
   bounded by reading the dumps.

### Weight-only comparison without a GPU

`tools/kld/artifact_to_hf.py` isolates the weight-format effect of an artifact from NInfer kernels
and KV formats. It decodes every `text/` parameter of a Qwen3.8-27B artifact from its stored words
(NVFP4, row-scaled FP8, grouped integer, direct) into the Hugging Face layout of the BF16 source
checkpoint, which supplies the remaining tensors, the config and the tokenizer. llama.cpp then
converts and scores that checkpoint like any other model, on CPU if needed:

```bash
python3 -m tools.kld.artifact_to_hf --artifact candidate.ninfer \
  --source /path/to/Qwen3.8-27B-BF16 --out $K/candidate-hf
python3 /path/to/llama.cpp/convert_hf_to_gguf.py $K/candidate-hf --outtype f16 \
  --outfile $K/candidate-F16.gguf
$LLAMA/llama-perplexity -m $K/candidate-F16.gguf -f $K/corpus-v1/corpus.txt --no-escape \
  -c 4096 --chunks 8 -ngl 0 --kl-divergence-base $K/dumps/candidate.kld
```

The reference is then the BF16 GGUF of the source checkpoint scored the same way. F16 storage keeps
FP8 row-scaled values nearly exact and rounds NVFP4 values to 11 significant bits, far below their
own quantization error. `artifact_to_hf.json` records the relative RMS difference of every decoded
tensor to the source.

The comparison is only meaningful with identical corpus, context and KV settings on the NInfer
side; the llama.cpp runs use their default F16 KV cache. Delete candidate dumps once their compare
JSON exists; keep the reference and production dumps for later stages.
