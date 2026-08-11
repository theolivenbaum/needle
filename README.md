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
180 MB). `dotnet test` runs 135 tests; the six that need fixtures skip cleanly
when they are absent, and pick them up from `fixtures/` or `$NEEDLE_FIXTURES`.

## Layout

```
src/Needle/Math/          NdArray, SIMD GEMM, norms, Walsh-Hadamard, Sinkhorn
src/Needle/Model/         the model: attention, engram, MHC stack, heads
src/Needle/Weights/       .cact reader, Cactus-Quant, safetensors, config
src/Needle/Tokenizer/     SentencePiece BPE from the blob, chat template
src/Needle/Inference/     KV-cached session, agent API, constrained decoding
src/Needle/Training/      autodiff tape, tape-based model, LoRA, AdamW, the loop
src/Needle/Diagnostics/   the parity harnesses
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
row, and what is left is a dot product against 2-bit codebook indices — for
which a 256-entry table maps each byte to the four values it encodes, one vector
multiply-add per four weights.

```sh
needle bench --weights needle2.cact          # packed
needle bench --weights needle2.cact --dense  # expanded to float32
```

|              | packed | float32 |
|---|---|---|
| weights      | **13.6 MB** (2.50 bits/parameter) | 173.1 MB |
| prefill      | 159 tok/s | 302 tok/s |
| decode       | 53 tok/s  | 48 tok/s  |
| allocated    | 33 KB/token | 37 KB/token |

Both produce identical token streams. The footprint is the point: 13.6 MB of
weights plus a 4.7 MB scratch arena and a bounded KV cache is what makes the
model fit the memory budget it was designed for. On a 4-core 2.8 GHz Xeon VM the
speeds are close enough that either is usable; prefill favours float32 because
dense GEMM vectorises eight lanes wide against the codec's four.

Per-token garbage is down from ~295 KB to ~33 KB: layer temporaries come from a
pooled bump arena that rewinds between layers, the engram sites stream their
convolution history in a ring buffer instead of recomputing a twelve-token
window each step, and the logits row and attention plan are reused across steps.

Numbers above were measured after a warm-up pass over the real shapes — tiered
JIT needs it, and timing a cold run measures the interpreter.

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
