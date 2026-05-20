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
| `training/eval.py` (throughput, repetition, generation-quality, WER, retrieval Recall@k / MRR) | `Inference/GenerationBenchmarks.cs` ✱ |
| `training/finetune.py` (JSONL flow) | `Training/JsonlDataset.cs`, `Training/JsonlFinetuner.cs` ✨ |
| `cli.py` (run/eval/finetune/export) | `src/Needle.Cli/CliEntry.cs` ✨                             |
| n/a (own binary format)             | `Weights/WeightLoader.cs` (.ndlw + safetensors)            |
| `training/train.py` (CLIP step)     | `Training/Trainer.cs::TrainStepWithContrastive` ✦          |
| `dataset/dataset.py` (`pack_sequences`) | `Training/JsonlDataset.cs::BatchBuilder.PackBatch` ✦   |
| `model/architecture.py` (RoPE cache)| `Model/NeedleModel.cs::GetRope` (per-device cache) ✦       |

✨ added on branch `claude/investigate-port-gaps-Z1hjA`.
✦ added on branch `claude/test-implement-missing-Xip11`.
✱ added on branch `claude/test-implement-missing-G5KET`.

138 xUnit tests pass across all of the above (`dotnet test`).

End-to-end parity against Python on the published checkpoint
(`Cactus-Compute/needle`) is exercised by the harness in
`scripts/compare/`.  Greedy generation matches Python token-for-token
on both spec test queries after fixing two bugs surfaced by that run
(decode-once in `InferenceRunner.Generate`, and layer-counting in
`CliEntry.CountLayers`).

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

(All previously listed smaller gaps are now closed — see the ✦ and ✱ rows
in the table above.  Contrastive loss is wired into `Trainer` via
`TrainStepWithContrastive`; multi-example bin packing is implemented in
`BatchBuilder.PackBatch`/`IteratePacked`; RoPE tables are cached
per-device on `SimpleAttentionNetwork.GetRope` and grow lazily up to
`TransformerConfig.MaxSeqLen`; throughput, bigram-repetition,
generation-quality, WER, and retrieval Recall@k/MRR benchmarks live in
`Inference/GenerationBenchmarks.cs`.)
