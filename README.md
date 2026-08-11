# Needle 2 for .NET

<img src="assets/banner.png" alt="Needle" style="border-radius: 30px; width: 100%;">

A pure C# implementation of [Needle 2](https://github.com/cactus-compute/needle),
Cactus Compute's 45M-parameter tool-calling model. It loads the released
`needle2.cact` blob — weights and tokenizer in one 14 MB file — and runs it with
no native dependency and no ML framework: the whole model is `float[]` behind
`System.Numerics.Tensors` and `Vector<float>`.

The upstream Python package is vendored under [`.reference/`](.reference) and is
the specification this port is checked against. Nothing here is a
transcription-from-memory: every stage is compared numerically against the
unmodified reference on the released weights (see [Parity](#parity)).

## What it is

Needle 2 is a *Simple Attention Network*, not a plain transformer:

| Piece | What replaces the usual thing |
|---|---|
| MLP | **Hadamard MLP** — two fixed Walsh transforms around a SiLU with three learned diagonals, and no weight matrix at all. Runs here as the in-place butterfly, `n log n`. |
| Memory | **Engram key-value memory** — hashed n-gram tables read at two layers, gated into the residual by a learned similarity. |
| Residual | **Multi-lane hyper-connections** — four parallel residual lanes, mixed each layer by an input-dependent, Sinkhorn-normalised doubly-stochastic routing matrix. |
| Attention | GQA with QK-norm, RoPE and a sigmoid output gate, over a 256-token sliding window with the tool block pinned as a KV sink. |
| Heads | Tied output projection, plus probe-pooled **contrastive** (tool retrieval) and **confidence** heads. |

See the paper for the design and ablations: [arXiv:2607.18363](https://arxiv.org/abs/2607.18363).

## Quickstart

```sh
# the released blob: weights + tokenizer, 14 MB
curl -L -o needle2.cact \
  https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact

dotnet run --project src/Needle.Cli -c Release -- \
  call --weights needle2.cact \
       --tools '[{"name":"set_lights","description":"Turn a room'"'"'s lights on or off and set brightness","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer"}},"required":["room","on"]}}]' \
       --query 'dim the living room to 30'
```

```json
{
  "type": "call",
  "success": true,
  "function_calls": [
    { "name": "set_lights", "arguments": { "room": "living room", "brightness": 30 } }
  ],
  "reasoning": "room 'living room' from query; brightness 30 from '30'",
  "confidence": 0.9243
}
```

From code:

```csharp
using Needle.Inference;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Weights;

var blob = CactLayout.Load("needle2.cact");
var model = new Needle2Model(Needle2Weights.FromFlat(blob.Config, blob.Parameters));
var tokenizer = CactTokenizer.FromCact("needle2.cact");

var agent = new NeedleAgent(model, tokenizer, toolsJson);
var response = agent.Complete("dim the living room to 30");

foreach (var call in response.FunctionCalls)
    Console.WriteLine($"{call.Name} {call.Arguments}");
```

`NeedleAgent` keeps one toolset per session; later turns are bare queries
against the same tools, and `Reset()` rewinds the conversation while keeping
them loaded. Feed an executed tool's result straight back into `Complete()` to
continue the loop.

Tool names and argument keys are constrained at decode time by a grammar
compiled from your schemas, so the model cannot invent a field you did not
declare — on the example above, unconstrained decoding emits an `action`
argument that no schema mentions, and the grammar turns it into the declared
`on`. The derivation is generated unconstrained and stays legible. A request no
declared tool can serve comes back as the empty call `[]`, with the confidence
score dropping accordingly.

## CLI

```
needle call       --weights <file.cact> --query <text> [--tools <json|@file>] [--system <text>]
needle run        --weights <dir|file|.cact> --tokens 2,100,200 [--max-new 32]
needle info       --weights <dir|file|.cact> [--dense]
needle bench      --weights <dir|file|.cact> [--prompt 128] [--decode 64] [--dense]
                  [--profile] [--profile-alloc]
needle finetune   --weights <file.cact> --data <file.jsonl> --out <adapter.safetensors>
                  [--epochs 3] [--lr 1e-4] [--rank 16] [--alpha 32] [--max-len 128]
                  [--batch-size 4] [--eval]
needle parity     --fixtures <dir> [--tolerance 0.01] [--verbose]
needle cact-check --cact <file> --expected <dir> [--tolerance 1e-4] [--verbose]
```

Weights may be a `.cact` blob, a directory holding `weights.safetensors` +
`config.json`, or a bare `.safetensors` file with `--config` alongside it.

## Parity

Correctness is established by replaying the reference, not by inspection. Two
scripts dump what upstream actually produces; two CLI commands replay it here.

```sh
pip install "jax[cpu]" flax numpy huggingface_hub

# every stage of the model on the released 45M checkpoint
python3 scripts/parity/dump_reference.py --out fixtures/parity
dotnet run --project src/Needle.Cli -c Release -- parity --fixtures fixtures/parity

# the .cact blob: quantised weights and the embedded tokenizer
python3 scripts/parity/dump_cact.py --out fixtures/cact
dotnet run --project src/Needle.Cli -c Release -- cact-check --cact needle2.cact --expected fixtures/cact
```

`dump_reference.py` records the embeddings, both engram sites' keys and values,
**every layer's residual snapshot**, the final hidden state, the logits, the MTP
logits, the contrastive embedding, the confidence logit, and a KV-cached greedy
continuation. Comparing stage by stage rather than only at the logits means a
disagreement localises to the layer that introduced it — which is how the one
real porting bug in this work was found (a residual measured from the
post-engram-injection input instead of the pre-injection one, invisible until
layer 2).

Current results on `Cactus-Compute/needle2`:

| Check | Result |
|---|---|
| Every stage, 3 prompts (8 / 10 / 64 tokens) | worst relative error **4.7e-5**, cosine > 0.99999 |
| Next-token argmax, all positions | **82/82** identical |
| KV-cached greedy continuation | token stream **identical** to the reference decoder |
| `.cact` dequantisation, all 404 tensors | worst relative error **2.1e-5** vs `read_export` |
| Tokenizer, 18 cases (markers, byte fallback, CJK, emoji) | **18/18** encode and decode identically |

Everything left is float32 rounding: the reference runs the same arithmetic in a
different order.

Fixtures are generated rather than committed (the float32 checkpoint alone is
180 MB). `dotnet test` runs 141 tests; the six that need fixtures skip cleanly
when they are absent, and pick them up from `fixtures/` or `$NEEDLE_FIXTURES`.

## Layout

```
src/Needle/Math/          NdArray, SIMD GEMM, attention kernels, Walsh-Hadamard
src/Needle/Model/         the model: attention, engram, MHC stack, heads
src/Needle/Weights/       .cact reader, Cactus-Quant, safetensors, config
src/Needle/Tokenizer/     SentencePiece BPE from the blob, chat template
src/Needle/Inference/     KV-cached session, agent API, constrained decoding
src/Needle/Training/      autodiff tape, tape-based model, LoRA, AdamW, the loop
src/Needle/Diagnostics/   the parity harnesses and the stage profiler
scripts/parity/           the Python side of those harnesses
.reference/               upstream, vendored verbatim
```

## Quantized weights

A `.cact` blob runs on its packed Cactus-Quant codes by default — the weights are
never expanded. The trick is the rotation the codec already applies: a group
reconstructs as `w = (codebook[idx] * norm) @ H` with `H` symmetric and
orthonormal, so

```
x · (q @ H) = (x @ H) · q
```

Transforming the *activation* once per group replaces transforming every weight
row, and what is left is a dot product against codebook indices. A codebook is
small enough to live in one vector — four entries at two bits, sixteen at four —
so a lane permute turns sixteen packed codes into sixteen weights with no memory
traffic beyond the codes themselves. Without AVX-512 the same loops fall back to
a byte-indexed table and a scalar tail.

```sh
needle bench --weights needle2.cact                    # packed
needle bench --weights needle2.cact --dense            # expanded to float32
needle bench --weights needle2.cact --profile          # per-stage timings
needle bench --weights needle2.cact --profile-alloc    # per-stage allocation
```

Throughput depends sharply on whether the machine has AVX-512, because that is
what the codebook permute needs. `needle bench` prints which vector widths the
run actually got. On a 4-core 2.8 GHz Xeon VM, 128-token prompt, 300 decode
steps, best of several interleaved runs:

|              | packed, AVX-512 | packed, AVX2 | float32, AVX2 |
|---|---|---|---|
| weights      | **13.6 MB** (2.50 bits/parameter) | 13.6 MB | 173.1 MB |
| prefill      | 356 tok/s | 207 tok/s | 395 tok/s |
| decode       | **127 tok/s** | 67 tok/s | 54 tok/s |
| allocated    | 21 KB/token | 21 KB/token | 37 KB/token |

The two columns are the same binary on the same VM before and after it lost
AVX-512 mid-session, so read them as two machines rather than as one comparison:
absolute figures move with the host. What holds in both is the ordering — packed
decodes faster than float32 either way, so the packed path is both the smaller
and the quicker one, and 13.6 MB of weights plus a 4.7 MB scratch arena and a
bounded KV cache is what makes the model fit the memory budget it was designed
for.

Without AVX-512 the packed kernels fall back to a byte-indexed table for two bits
and a scalar loop for four, which costs roughly half the decode rate. An AVX2
path for both is the obvious next step (see [todo.md](todo.md)).

Float32 decode is bound by streaming weights: 173 MB per token at roughly
10 GB/s. Splitting one output row across the four cores makes it *slower*
(45 tok/s against 60) — a decode step runs 135 of these matmuls and the dispatch
costs more than the work — so decode stays serial and only prefill parallelises.
That is also why packed overtakes it: at two bits a token reads 13.6 MB instead
of 173 MB, so the bandwidth wall moves out of the way.

Per-token garbage is down from ~295 KB to ~21 KB: layer temporaries come from a
pooled bump arena that rewinds between layers, the engram sites stream their
convolution history in a ring buffer instead of recomputing a twelve-token
window each step, and the logits row and attention plan are reused across steps.
Decode triggers no collections at all.

`--profile-alloc` says where the remaining 21 KB goes, and it is almost all
bookkeeping rather than data. About 16 KB is `NdArray` *view* objects: the arena
pools the float storage but still hands back a 40-byte wrapper per tensor, and a
layer takes fifteen of them (four attention projections, three lane gates, two
block norms, and one each for the context, output, MLP, lane read, lane reduce
and residual). Twenty-seven layers make roughly 400 wrappers a token. The other
4 KB is two arrays that genuinely are new each step — the embedding row and the
final hidden state, 2 KB of float32 apiece — plus about 1 KB of engram plumbing
and the attention plan. Nothing scales with the window or the sequence.

All numbers are measured after a warm-up pass over the real shapes — tiered JIT
needs it, and timing a cold run measures the interpreter.

## Where the time goes

`needle bench --profile` attaches a stage timer to the decode loop. It answers a
question a sampling profiler cannot: the same GEMM kernel serves the projections,
the gates and the output head, so knowing which *function* is hot does not say
which part of the model is. Per token, packed:

| stage | µs | share |
|---|---|---|
| Q/K/V and output-gate projections | 2710 | 34% |
| attention scores, softmax, weighted sum | 1564 | 20% |
| tied output projection over the vocabulary | 1215 | 15% |
| attention output projection | 918 | 12% |
| hyper-connection gate projections | 854 | 11% |
| Sinkhorn routing | 179 | 2% |
| engram | 171 | 2% |
| everything else | 240 | 3% |

Profiling this way, plus `dotnet-trace` for the function-level view, turned up
five things worth fixing. Interleaved against the previous build, packed decode
went 60 → 133 tok/s with AVX-512 and 49 → 67 tok/s without:

- **The packed inner loops were scalar or narrow.** Two-bit codes were decoded
  through a 4 KB byte→`Vector4` table, one dependent load per four weights; four
  bits had no vector path at all. Both are lane permutes: a four-bit codebook has
  exactly sixteen entries, which is the width of an AVX-512 permute, and sixteen
  two-bit codes fit one 32-bit word, so a single variable shift spreads them
  across all sixteen lanes. That is most of the 2.2×.
- **The activation rotation was repeated.** Packed weights need their input
  rotated by the codec's Walsh transform, and the rotation depends on the
  activation, not the matrix — but query, key, value and the output gate all
  project the same block input, and all three hyper-connection gates project the
  same lane vector. Now rotated once per group.
- **The Walsh butterfly called a helper per block.** A 128-wide group runs seven
  passes, four of which work in strides of 1 to 8 floats — 120 of the 127 block
  combinations. The call cost far more than the two adds inside it.
- **Attention made 110,000 library calls per token.** Eight heads against a
  256-position window, 27 layers over, each a 64-float dot or scaled add: about
  23 ns a call for 3 ns of arithmetic. `Math/AttentionKernels.cs` fuses the
  window into one call per head.
- **A single-token GEMM read its accumulator back once per reduction row.** The
  four-row blocking that makes prefill fast does nothing when there is one output
  row; blocking along the reduction axis instead cuts accumulator traffic to a
  quarter. Both widths are written out explicitly, because on AVX-512 hardware
  .NET keeps `Vector<T>` at 256 bits while `TensorPrimitives` uses 512 — the
  portable form would have lost to the library call it replaced.

Measuring took as much care as fixing. Run-to-run spread on a shared VM is
several percent, enough to swamp a real 5% win, so changes were judged by
building both versions and interleaving them best-of-N rather than by comparing
successive runs. Even that only protects the ratio: partway through this work the
VM stopped reporting AVX-512, which halved every absolute figure and cut the
speedup from 2.2× to 1.4× — hence the two columns above, and hence `bench`
printing the vector widths it actually got.

## Training

LoRA fine-tuning runs here too, on the same JSONL format the reference's
`finetune.py` consumes: `{"query", "tools", "answers", "reasoning"}` per line.

```sh
needle finetune --weights needle2.cact --data train.jsonl --out lora.safetensors \
                --epochs 3 --rank 8 --lr 1e-3 --batch-size 4
```

```
4 training examples; LoRA rank 8 on 135 projections (1.00M trainable of 43.4M)
warmed the training path in 12.0s
training-set loss before: 1.9585
  epoch 1 step 1: loss 1.9585, grad norm 0.419, lr 0.001, 6.1s
  epoch 10 step 10: loss 1.1734, grad norm 0.307, lr 0.000127, 7.9s
training-set loss after:  1.1614
```

Only the five attention projections adapt, matching the reference's
`LORA_TARGETS`; norms, the Hadamard diagonals, the engram tables and the
hyper-connection routing stay frozen. `B` starts at zero, so step zero *is* the
base model, and `LoraSet.Merge` folds `W + (alpha/rank)·A·B` back into an
ordinary weight set when you are done.

**Training is a separate execution flow.** Inference runs allocation-free over a
pooled arena and records nothing; making it carry a tape would slow down the path
that matters. So `Training/` re-expresses the same arithmetic through a
reverse-mode autodiff tape — two implementations of one model, each shaped for
what it has to do. That is a place for them to drift apart, so the agreement is
asserted rather than assumed:

| Check | How |
|---|---|
| The two forwards agree | `TrainingCheck.ForwardMatchesInference` — same tokens, no adapters, < 1e-4 relative |
| Every hand-written backward | central finite differences, per op: Sigmoid, SiLU, RmsUnit, Walsh, Softmax, Sinkhorn, ZcRmsNorm, MatMul, CrossEntropy |
| The whole model's gradient | `TrainingCheck.CheckGradients`, probing the largest-gradient adapter coordinates |
| The optimiser descends what it reports | fine-tune on the released weights, evaluate a fixed set before and after |

A gradient check has a subtlety worth naming: a central difference divides by
`2·epsilon`, so a loss carrying float32 rounding of about `|loss|·2⁻²³` cannot
resolve anything below `|loss|·2⁻²³/epsilon`. `GradientProbe` carries that noise
floor explicitly and the check asserts at least one probe sits well above it, so
it can neither fail on rounding nor pass vacuously.

Costs, for 128-token examples on the 45M model: a step is a few seconds per
example on four cores, and the peak working set is ~2.8 GB — about 1 GB of live
tape (every intermediate is retained for the backward pass) plus segments the
collector has not returned yet. `DOTNET_GCConserveMemory=9` holds it near 1 GB
with no measurable slowdown. Training also expands the packed weights to float32,
since it multiplies by them from both sides.

The run warms the whole training path — forward, backward, clip, optimiser step —
before the first timed step, using a scratch optimiser at learning rate zero so
every parameter comes back bit-identical. Without it step one reports 10.4s
against a 6.1s steady state, which is a JIT measurement, not a step.

Not implemented: full fine-tuning (only LoRA), and the reference's
OpenRouter-backed synthetic data generation.

## Relationship to upstream

`.reference/` tracks `cactus-compute/needle`. The Python package there is the
real thing — `pip install cactus-needle` — and ships a compiled engine; this
repository is an independent .NET implementation of the same model, useful when
you want the model inside a .NET process without a native dependency.

Needle 2 is built by the Cactus Compute team:

```bibtex
@misc{needle2_2026,
  title        = {Needle 2: A 45M-Parameter Foundation Tool-Calling Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```
