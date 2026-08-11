# Port status: Needle 2 → .NET 10

What the C# implementation under `src/` covers relative to the vendored
reference in `.reference/needle`, and what it deliberately does not.

Upstream synced at `cactus-compute/needle@8b01d9f` (2026-08-10), the Needle 2
release. That release replaced the encoder-decoder tool-caller entirely, so the
previous port was rewritten rather than extended: no TorchSharp, no
`Microsoft.ML.Tokenizers`, one NuGet dependency (`System.Numerics.Tensors`).

## Implemented and verified

| Reference | C# | Verified by |
|---|---|---|
| `model/architecture.py` — attention, Hadamard MLP, engram, MHC stack, heads | `Model/Needle2Model.cs`, `Model/StackState.cs`, `Model/EngramHash.cs`, `Model/RoPE.cs`, `Model/SequenceMask.cs`, `Model/AttentionPlan.cs` | per-layer parity on the released weights |
| `model/decode.py` — KV cache, sliding window, engram window | `Inference/NeedleSession.cs` | token-identical greedy continuation |
| `model/quantize.py` — Cactus-Quant codec | `Weights/CactusQuant.cs` | all 404 tensors vs `read_export` |
| `model/export.py` — `.cact` container and geometry recovery | `Weights/CactFile.cs`, `Weights/CactLayout.cs` | as above, plus a loaded-and-run assertion |
| `model/export.py` — `RefTokenizer` | `Tokenizer/CactTokenizer.cs` | 18/18 encode+decode cases |
| `model/tokenizer.py` — reserved IDs and chat markers | `Tokenizer/ChatMarkers.cs` | ID assignment asserted against the blob |
| `model/finetune.py` — `render_example`, JSONL contract | `Tokenizer/ChatMarkers.cs` (`ChatTemplate`), `Training/JsonlDataset.cs` | unit tests |
| `needle/__init__.py` — the `Needle` session API | `Inference/NeedleAgent.cs` | end-to-end calls from `needle2.cact` |
| `architecture.py` — `kv_budget_window` | `Model/SequenceMask.cs` (`KvBudget`) | unit tests |
| `model/quantize.py` — inference on packed weights | `Weights/QuantizedMatrix.cs`, `Weights/CactWeights.cs`, `Model/WeightViews.cs` | identical token streams to the float32 path |
| n/a — safetensors interchange | `Weights/Safetensors.cs` | round-trips the reference dumps |
| grammar-constrained decoding (upstream moved this into its compiled engine) | `Inference/ConstrainedDecoding.cs`, wired into `NeedleAgent` | unit tests; observed to correct an out-of-schema argument on a real call |
| `model/finetune.py` — `init_lora`, `LORA_TARGETS`, the AdamW + warmup-cosine loop | `Training/Autodiff/`, `Training/TrainableModel.cs`, `Training/LoraAdapter.cs`, `Training/AdamW.cs`, `Training/Finetuner.cs` | forward parity against the inference model, finite-difference gradient checks, a fine-tune on the released weights |

`dotnet test` → 135 tests. Six need the reference fixtures and skip cleanly when
they are absent.

## Parity results

Against `Cactus-Compute/needle2` (45M parameters, the released `.cact` and the
float16 checkpoint behind it):

- every stage, three prompts: worst relative error **4.7e-5**, cosine > 0.99999
- next-token argmax: **82/82** positions identical
- KV-cached greedy decode: token stream identical to `decode.py`
- `.cact` dequantisation: worst relative error **2.1e-5** across 404 tensors
- tokenizer: **18/18** cases encode and decode identically

Regenerate with `scripts/parity/dump_reference.py` and
`scripts/parity/dump_cact.py`, then `needle parity` / `needle cact-check`.

One real defect was found this way: the stack was measuring each layer's
residual against the *post*-engram-injection block input, where the reference
measures it against the pre-injection read. Invisible for layers 0–1, a 12%
error from layer 2 on. Nothing but a stage-by-stage comparison would have caught
it — end-to-end argmax still mostly agreed.

## Not implemented

Inference-side gaps, in rough order of usefulness:

- **The packed path is not faster, only smaller.** 13.6 MB against 173 MB, at
  roughly the same decode rate and about half the prefill rate. The 2-bit inner
  loop moves four weights per vector operation where dense float32 moves eight or
  sixteen; closing that needs real intrinsics (masked accumulation per codebook
  entry, or a wider byte table) rather than portable `Vector4`.

- **The grammar only constrains names and argument keys**, not argument *values*.
  Upstream's engine also compiles `Field` constraints — ranges, patterns,
  lengths, enums, item counts — into the decode grammar, so a value that
  violates the schema is unreachable. Here those still have to be validated
  after the fact.
- **`.cact` writing.** Reading is complete. Writing needs the 2/3/4-bit
  Lloyd-Max codebooks, which the format carries in the header but which are
  generated in Python from NumPy's legacy RandomState — reproducible only by
  reimplementing MT19937 and Box-Muller exactly. A writer that reuses codebooks
  read from an existing blob would be straightforward.
- **Tool retrieval is not automatic.** `NeedleAgent.RetrieveTools` exposes the
  contrastive head, but the agent does not yet drop to the top five tools and
  rebuild the grammar when more than five are declared.
- **Full fine-tuning.** Only LoRA on the five attention projections trains; the
  autodiff layer under it is general, but nothing else is registered as a
  parameter and the tape would have to retain the base activations to make it
  worthwhile.
- **Adapters do not export to `.cact`.** `LoraSet.Merge` folds them back into a
  float32 weight set, which safetensors can carry, but writing a `.cact` needs
  the codebook generation described above.

## Intentionally out of scope

Python-ecosystem infrastructure with no .NET counterpart:

- `agent/fetch.py` — downloads the compiled inference engine
- `playground/server.py` — the browser playground
- `model/finetune.py` — the OpenRouter-backed synthetic data generation, which
  is an API client rather than model code (the training loop itself is ported)
- `cli.py` — the `needle` Python CLI (this repo has its own)
