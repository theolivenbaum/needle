# Port Status: Python → .NET 10

Snapshot of what the C# port (under `src/Needle`, `src/Needle.Cli`,
`tests/Needle.Tests`) covers, relative to the Python reference under
`.reference/needle` (mirrored in `needle/`).

## Ported and tested

| Python module                       | C# location                                                |
|-------------------------------------|------------------------------------------------------------|
| `model/architecture.py`             | `Model/NeedleModel.cs`, `Model/RoPE.cs`, `Model/TransformerConfig.cs`, `Model/MaskUtils.cs` |
| `model/run.py` (generate, retrieve) | `Inference/Runner.cs`                                      |
| `model/constrained.py`              | `Inference/ConstrainedDecoding.cs`                         |
| `dataset/tokenizer.py`              | `Tokenizer/NeedleTokenizer.cs`, `Tokenizer/INeedleTokenizer.cs` |
| `model/run.py` (tool normalize)     | `Inference/ToolNormalizer.cs`                              |
| `training/train.py` (loss core)     | `Training/LossFunctions.cs`, `Training/Trainer.cs`         |
| `training/optim.py`                 | `Training/MuonOptimizer.cs`, `Training/LRSchedule.cs`      |
| `model/quantize.py`                 | `Model/Quantize.cs` ✨                                      |
| `model/architecture.py` (matryoshka)| `Model/NeedleModel.cs::ForwardMasked` + FFN mask plumbing ✨|
| `model/architecture.py` (contrastive twin)| `Model/NeedleModel.cs::ForwardContrastive` ✨        |
| `model/export.py`                   | `Weights/SubmodelExport.cs` ✨                              |
| `training/eval.py` (tool-call F1)   | `Inference/ToolCallMetrics.cs` ✨                           |
| `training/eval.py` (perplexity)     | `Training/PerplexityEval.cs` ✨                             |
| `training/finetune.py` (JSONL flow) | `Training/JsonlDataset.cs`, `Training/JsonlFinetuner.cs` ✨ |
| `cli.py` (run/eval/finetune/export) | `src/Needle.Cli/CliEntry.cs` ✨                             |
| n/a (own binary format)             | `Weights/WeightLoader.cs` (.ndlw + safetensors)            |

✨ added on branch `claude/investigate-port-gaps-Z1hjA`.

115 xUnit tests pass across all of the above (`dotnet test`).

## Intentionally NOT ported

These are infrastructure or Python-ecosystem dependent and outside the scope
of an inference / local-finetune .NET runtime:

- `dataset/generate.py` — Gemini-based synthetic data generation
- `ui/server.py` — Gradio-style web UI
- `utils/distributed.py` — JAX-pmap multi-host data sharding
- `utils/gcs.py` — Google Cloud Storage uploader
- `utils/tpu.py` — `gcloud` TPU VM lifecycle management
- `dataset/dataset.py` — HuggingFace download + arrow caching + multi-process
  packing (the .NET side ships a simpler one-example-per-row JSONL loader
  in `Training/JsonlDataset.cs`)
- `dataset/tokenize.py` — bulk corpus pre-tokenisation on TPU pods
- `training/pretrain.py` — pretrain on PleIAs/SYNTH (JAX/Flax specific)
- `training/train.py` distributed code paths (host slicing, pmap, multihost
  all-gather) — the .NET trainer is single-host
- `model/run.py` pickle (`.pkl`) checkpoint loader — upstream weights are
  published in safetensors format; convert externally if you only have a
  `.pkl`

## Known smaller gaps still open

- **Contrastive loss in trainer.** The Python `_train_step` mixes a CLIP
  contrastive loss every 1000 steps (`_contrastive_loss_fn`). The .NET
  `Trainer.TrainStep` is text-loss only. The building blocks
  (`SimpleAttentionNetwork.ForwardContrastive`,
  `LossFunctions.ClipContrastiveLoss`) are in place — wiring them into the
  train loop is the next obvious step if contrastive retrieval matters.
- **Multi-example sequence packing.** `BatchBuilder` puts one example per
  row and pads to the longest row; the Python `pack_sequences` packs
  multiple examples into one fixed-width row separated by segment IDs.
  Single-example packing is correct (the segment-aware mask functions
  collapse to plain padding masks) just less compute-efficient.
- **RoPE caching.** `SimpleAttentionNetwork` recomputes RoPE on every
  forward; Python caches per (model, max_gen_len).
