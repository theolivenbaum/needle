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
needle info       --weights <dir|file|.cact>
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
180 MB). `dotnet test` runs 114 tests; the five that replay fixtures skip
cleanly when they are absent, and pick them up from `fixtures/` or
`$NEEDLE_FIXTURES`.

## Layout

```
src/Needle/Math/          NdArray, SIMD GEMM, norms, Walsh-Hadamard, Sinkhorn
src/Needle/Model/         the model: attention, engram, MHC stack, heads
src/Needle/Weights/       .cact reader, Cactus-Quant, safetensors, config
src/Needle/Tokenizer/     SentencePiece BPE from the blob, chat template
src/Needle/Inference/     KV-cached session, agent API, constrained decoding
src/Needle/Diagnostics/   the parity harnesses
scripts/parity/           the Python side of those harnesses
.reference/               upstream, vendored verbatim
```

## Performance

Single-threaded decode is memory-bound: 43M float32 parameters is ~170 MB of
weight traffic per token, so throughput tracks memory bandwidth rather than
FLOPs. On a 4-core 2.8 GHz Xeon VM that is roughly 40 tokens/s decode, with
prefill parallelised across cores. Keeping the weights quantised and
dequantising inside the matmul is the obvious next step and is not implemented.

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
