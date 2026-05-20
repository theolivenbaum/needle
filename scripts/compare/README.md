# Cross-runtime parity harness (Python ↔ .NET)

This directory contains a harness for diffing the Python `needle` package
against the .NET (C#) port end-to-end.  It runs the same inputs through
both runtimes and compares the produced outputs:

- **Tokenization** — exact match of encoded ID lists for a list of strings.
- **Greedy generation** — exact match of generated token IDs.
- **Retrieval embeddings** — element-wise diff of contrastive embeddings
  (within tolerance).

The harness is build/test-clean from this repo, but **running it end-to-end
requires both a Python environment with the `needle` package installed and
a matched pair of model files** (a `.pkl` for Python and a `.safetensors`
for C#).  See "Getting weights" below.

## Files

| File                              | Purpose                                                          |
|-----------------------------------|------------------------------------------------------------------|
| `spec.json`                       | Test cases (queries, tools, retrieval texts).                    |
| `dump_python.py`                  | Runs the spec through Python and writes `python.json`.           |
| `compare.py`                      | Diffs the two output JSON files.                                 |
| `convert_pkl_to_safetensors.py`   | Best-effort `.pkl` → `.safetensors` converter (Flax → TorchSharp). |

The C# side uses the `dump-compare` subcommand on `needle` (see
`src/Needle.Cli/CliEntry.cs`).

## End-to-end workflow

1. **Get weights and tokenizer.**  Download `needle.pkl` and
   `tokenizer.model` from <https://huggingface.co/Cactus-Compute/needle>.

2. **Convert weights for the .NET side.**

   ```bash
   python scripts/compare/convert_pkl_to_safetensors.py \
       --input  needle.pkl \
       --output needle.safetensors
   ```

   The converter handles Flax→TorchSharp layout differences (key
   renaming, `kernel`→`weight` transpose, `nn.scan` unstacking).

3. **Dump from each runtime.**

   ```bash
   # Python side
   python scripts/compare/dump_python.py \
       --checkpoint needle.pkl \
       --tokenizer  tokenizer.model \
       --spec       scripts/compare/spec.json \
       --out        outputs/python.json

   # C# side
   dotnet run --project src/Needle.Cli -- dump-compare \
       --checkpoint needle.safetensors \
       --tokenizer  tokenizer.model \
       --spec       scripts/compare/spec.json \
       --out        outputs/csharp.json
   ```

4. **Diff.**

   ```bash
   python scripts/compare/compare.py \
       --python outputs/python.json \
       --csharp outputs/csharp.json \
       --emb-tol 1e-3
   ```

   Exit code 0 = all checks pass under tolerance.

## Tokenizer-only smoke test (no weights needed)

If you only have the SentencePiece model and want a quick parity check on
encoding, run the spec with the `tokenize` section only (or strip the
other sections out) — `dump_python.py` and `needle dump-compare` will skip
model loading when neither `generate` nor `encode_retrieval` is present.

## What "match" means

| Check                | Tolerance                                                                  |
|----------------------|----------------------------------------------------------------------------|
| Tokenization         | Exact integer-ID equality.                                                 |
| Generated tokens     | Reported on a per-case basis; pass `--token-strict` to fail on any drift.  |
| Retrieval embeddings | Element-wise max absolute error ≤ `--emb-tol` (default `1e-3`).            |

Greedy argmax decoding is sensitive to small logit differences near the
top-1 boundary, so a stray drift early in a sequence can fan out to many
divergent tokens later.  The comparator reports the first divergence
position and the prefixes either side — useful for triage, but not a
clean pass/fail unless you opt in with `--token-strict`.

## Findings from the first end-to-end run

Running this harness against the published `Cactus-Compute/needle`
checkpoint (`d=512`, `12/8` enc/dec layers, contrastive_dim 128) surfaced
two real bugs in the .NET port plus one tokenizer edge case:

1. **`InferenceRunner.Generate` decoded the buffer only once.**  The
   autoregressive loop reused the same logits tensor across every
   position, so predictions ignored anything appended after the initial
   pass.  `GenerateBatch` re-decoded each step and was fine.  Fixed in
   `src/Needle/Inference/Runner.cs`; regression test
   `NeedleModelTests.Decode_LogitsDependOnPriorDecoderTokens` covers the
   underlying property.

2. **`CliEntry.CountLayers` didn't recognise the TorchSharp ModuleList
   naming.**  It looked for `encoder.layer_<n>` but `named_parameters()`
   emits `encoder._layers.<n>`, so a freshly converted safetensors
   checkpoint loaded as a single-layer model.  Fixed in
   `src/Needle.Cli/CliEntry.cs`.

3. **Special-token tokenization drift around `<tool_call>` / `<tools>`.**
   The Microsoft.ML.Tokenizers wrapper splits the input on each
   special-token literal and SP-encodes each segment independently,
   producing a fresh ▁ "dummy prefix" per segment.  Python's
   SentencePiece adds the dummy prefix once at the start of the entire
   input and treats user-defined symbols as in-stream tokens.  Plus the
   underlying library has a normalisation quirk that returns `[]` for
   several short / whitespace-only inputs.  Fixed by replacing the
   library's specialTokens preprocessing with a manual segmenter in
   `NeedleTokenizer.Encode` / `Decode` that uses a fixed sentinel prefix
   (`"ab\n"`, plus an extra space for the dummy-prefix slot) to coax the
   underlying SP encoder into producing the right pieces.  Locked in by
   `TokenizerParityTests` (runs against the embedded model — no env var
   needed).

4. **Attention mask used `float.NegativeInfinity` instead of `finfo.min`.**
   A fully-masked query row (e.g. a padded slot in a packed batch)
   then produced a NaN softmax row that poisoned the encoder output
   via matmul.  Greedy single-example generation never hit this
   because no padding was present, but every training step on packed
   batches would have.  Fixed in `MultiHeadAttention.Call` to use the
   dtype's most negative *finite* value, matching Python's
   `jnp.finfo(dtype).min`.  Regression test
   `Forward_PackedBatchWithPadding_NoNaNLogits`.

After all four fixes, the harness reports zero mismatches on the spec
JSON: tokenize 4/4 exact, tool-name normalization 4/4 exact, four
generation cases (incl. constrained and tool-name normalized) all
exact, batched generation exact, retrieval embeddings within
tolerance, training-step loss within bf16 vs fp32 drift, INT4 fake-
quantization exact to float32 precision.

The tokenizer parity tests are network-free: the published SentencePiece
model (`needle.model`, ~125 KB) is shipped as an embedded resource in
`Needle.dll` under the logical name `Needle.Resources.needle.model`, and
`NeedleTokenizer.LoadDefault()` reads it directly out of the assembly.
The internal `NeedleTokenizerDownloader` is retained as an opt-in
maintenance utility for refreshing `src/Needle/Resources/needle.model`
from HuggingFace, but no runtime code path depends on it.

## Known limitations

- **bfloat16 vs float32.**  The Python reference runs in bfloat16 by
  default; the .NET port runs in float32.  Expect modest numeric drift in
  retrieval embeddings (typically `1e-3`–`1e-2`).  Use a tighter tolerance
  only after converting to float32 in both sides.
- **No deterministic dropout.**  Both `dump_python.py` and the C# runner
  use eval-mode forward passes, so dropout is disabled on both sides.
- **The converter is best-effort.**  It rejects unknown keys but cannot
  verify that the resulting C# config (heads, kv-heads, layer counts,
  d_ff) matches what the Python checkpoint expects.  If the .NET side
  raises a shape mismatch, double-check the inferred config in
  `CliEntry.LoadModel`.
