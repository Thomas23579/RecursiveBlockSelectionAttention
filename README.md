# Recursive Block-Selection Attention (RBA)

RBA is a drop-in sparse attention mechanism with $\mathcal{O}(N \log N)$ complexity in compute and memory transfer. This repository provides a monkey patch for Google's Gemma-4 models.

## Requirements

```
torch==2.8.0
transformers==5.5.4
accelerate==1.13.0
opencompass==0.5.2   # benchmarks only
```

## Installation

### RBA (`logattention`)

Work in an isolated conda environment (or similar): the monkey patch can modify the installed `transformers` and `opencompass` packages irreversibly.

* Ensure PyTorch 2.8.0 is installed.
* Install the package: `pip install -e .`

### opencompass benchmarks

* Install opencompass: `pip install opencompass==0.5.2`
* opencompass calls a deprecated `transformers` method and may raise:
  `AttributeError: GemmaTokenizer has no attribute batch_encode_plus. Did you mean: '_encode_plus'?`
  Patch it.

  Linux:

  ```bash
  sed -i 's/self\.tokenizer\.batch_encode_plus(/self.tokenizer(/' \
    /usr/local/lib/python3.12/dist-packages/opencompass/models/huggingface_above_v4_33.py
  ```

  Windows (PowerShell, in the activated conda env):

  ```powershell
  $f = python -c "import opencompass, os; print(os.path.join(os.path.dirname(opencompass.__file__), 'models', 'huggingface_above_v4_33.py'))"
  $c = (Get-Content $f -Raw) -replace 'self\.tokenizer\.batch_encode_plus\(', 'self.tokenizer('
  [System.IO.File]::WriteAllText($f, $c)
  ```

  This replaces the removed `batch_encode_plus(...)` call with the equivalent `self.tokenizer(...)` used by newer versions.

## Using RBA with Gemma-4

There are two ways to enable RBA:

**1. Replace the modeling file** in the installed `transformers` package. Required for the opencompass benchmarks. This mutates the package, so work in a conda environment.

```powershell
# RBA
python unsafe_replace_gemma4_model.py
# or dense attention for long context (the default implementation OOMs at 32k)
python unsafe_replace_gemma4_model_standardAttentionLongContext.py
```

**2. Patch the model at runtime:**

```python
from logattention import patch_global_attention
...
patch_global_attention(model)
```

A minimal end-to-end example is provided in `testModel.py`.

### Hyperparameters

Set the branching factor $B$, candidate budget $C$, and scale count $H$ in `logattention/gemma4_attention.py`:

* $B$ — keys combined per scale.
* $C$ — active (composite) keys per scale. Must be a multiple of $B$; otherwise `B * (C // B) != C` raises an error.
* $H$ — number of scales.

The supported context length is $C \cdot B^{\,H-1}$. The default $B=8$, $C=256$, $H=4$ therefore covers a 128k context ($256 \cdot 8^{3} = 131072$ tokens).

Set `model.config.text_config.block_size` to control how many sequence items are processed at once. Larger values create larger temporary tensors and can OOM the VRAM; 512 is a reasonable choice.

## Running the opencompass benchmark

### Prepare

```bash
python unsafe_register_gemma4_opencompass_config.py
python unsafe_replace_opencompass_needlebench.py
```

### Run

Linux:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  opencompass --datasets needlebench_v2_32k \
    --models hf_gemma4_31B_it \
    --summarizer needlebench/needlebench_v2_32k_summarizer \
    --work-dir ./outputs/gemma4_31B_needlebench_32k
```

Windows:
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
opencompass --datasets needlebench_v2_32k --models hf_gemma4_31B_it --summarizer needlebench/needlebench_v2_32k_summarizer --work-dir ./outputs/gemma4_31B_needlebench_32k
```


Results are written to:

* Summary: `outputs/gemma4_31B_needlebench_32k/<date-time>/summary/summary_<date-time>.txt`
* Generated responses: `outputs/gemma4_31B_needlebench_32k/<date-time>/results/<model>/*`

### Expected results

For the NeedleBench scores and the latency/memory comparison against dense attention, see the paper (cited below).

## Troubleshooting

* **OOM with default attention at long context.** Gemma-4's `head_dim=512` exceeds the fused FlashAttention/SDPA kernel limit, so the default dense path allocates large unfused tensors. Use RBA, or for a dense long-context baseline use `unsafe_replace_gemma4_model_standardAttentionLongContext.py`.
* **OOM in general.** Lower `model.config.text_config.block_size` and make sure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set.
* **`B * (C // B) != C` error.** $C$ must be a multiple of $B$.
* **`AttributeError: GemmaTokenizer has no attribute batch_encode_plus`.** Apply the opencompass `sed` patch from the installation section.
* `opencompass` might run into an error while evaluating the benchmark results. However, the summary is still created - no fix necessary.

## Additional information

### Benchmark Results for Reference
**Table 1 — Multi-Needle Reasoning (2 hops).** Accuracy (%) over 10 cases per configuration. RBA matches or exceeds dense attention everywhere, and the gap widens as N grows.

| Method | 1k/0 | 1k/50 | 1k/100 | 4k/0 | 4k/50 | 4k/100 | 16k/0 | 16k/50 | 16k/100 | 32k/0 | 32k/50 | 32k/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RBA | 100 | 100 | 100 | 90 | 100 | 100 | 80 | 50 | 100 | 70 | 50 | 100 |
| Dense Attention | 80 | 80 | 80 | 40 | 50 | 70 | 10 | 0 | 50 | 0 | 0 | 40 |

**Table 2 — Multi-Needle Reasoning (4 hops).** Accuracy (%) over 10 cases per configuration. RBA matches or exceeds dense attention everywhere, but both degrade sharply with N — the hardest task.

| Method | 1k/0 | 1k/50 | 1k/100 | 4k/0 | 4k/50 | 4k/100 | 16k/0 | 16k/50 | 16k/100 | 32k/0 | 32k/50 | 32k/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RBA | 80 | 80 | 80 | 30 | 20 | 70 | 10 | 10 | 40 | 0 | 0 | 70 |
| Dense Attention | 30 | 30 | 30 | 10 | 20 | 40 | 0 | 0 | 20 | 0 | 0 | 20 |

**Table 3 — Single-Needle Retrieval.** Accuracy (%) over 10 cases per configuration. RBA retrieves near-perfectly throughout; dense attention collapses at mid-depth (50%) for N ≥ 16k.

| Method | 1k/0 | 1k/50 | 1k/100 | 4k/0 | 4k/50 | 4k/100 | 16k/0 | 16k/50 | 16k/100 | 32k/0 | 32k/50 | 32k/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RBA | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 100 |
| Dense Attention | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 0 | 90 | 90 | 0 | 80 |

**Table 4 — Multi-Needle Retrieval.** Per-needle success (%) over 25 cases per N; each case holds all 3 needles, columns give the scored needle's depth. Dense attention fails on mid-depth needles for N ≥ 16k where RBA stays accurate; both decline mildly at N = 32k.

| Method | 1k/0 | 1k/50 | 1k/100 | 4k/0 | 4k/50 | 4k/100 | 16k/0 | 16k/50 | 16k/100 | 32k/0 | 32k/50 | 32k/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RBA | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 80 | 100 | 100 | 68 | 88 |
| Dense Attention | 100 | 100 | 100 | 100 | 88 | 96 | 96 | 12 | 100 | 96 | 12 | 88 |

### Hardware

Gemma-4 31B-it needs a high-memory GPU. In bf16 the weights alone occupy ~62 GB, so an H200 is recommended; the KV cache and benchmark context add further memory on top.

### Model weights

Gemma-4 is gated on the Hugging Face Hub. Accept the license on the model page, then authenticate before running:

```bash
huggingface-cli login
```

### License

See the `LICENSE` file. Use of the Gemma-4 weights is additionally governed by Google's Gemma Terms of Use.

## Citation

If you use RBA, please cite:

```bibtex
@misc{zeiringer2026rba,
  title         = {{Training-Free $O(N \log N)$ Sparse Attention via Recursive Block Selection}},
  author        = {Zeiringer, Thomas},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```