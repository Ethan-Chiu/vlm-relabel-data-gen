# data-gen

Large-scale image dataset generation and VLM annotation pipeline for robot manipulation training.

## Project Structure

```
data-gen/
├── config.toml                  # Runtime config (edit this to change behaviour)
├── .env                         # Secrets — API keys (gitignored, copy from .env.example)
├── .env.example                 # Template for .env
├── configs/                     # Alternative config files for different backends
│   ├── qwen_local.toml          # Qwen3-VL-8B via local transformers
│   └── qwen_vllm.toml           # Qwen3-VL-8B via vLLM server
│
├── src/datagen/
│   ├── config.py                # Pydantic-settings Config (TOML + env vars)
│   ├── download.py              # Download images → data/raw/, write metadata.parquet
│   ├── storage.py               # Parquet read/write helpers
│   ├── prompts.py               # Prompt templates (TYPE_A, TYPE_B, TYPE_C, VERIFY)
│   ├── annotator.py             # RoboticAnnotator — parallel A/B/C calls + verification
│   ├── pipeline.py              # Simple relabeling pipeline (ProcessPoolExecutor / Ray)
│   └── vlm/
│       ├── base.py              # VLMBackend ABC — one method: call(image_bytes, prompt)
│       ├── openai_backend.py    # OpenAI API (GPT-4o, GPT-4o-mini, …)
│       ├── gemini.py            # Google Gemini API (gemini-2.0-flash, …)
│       ├── vllm.py              # vLLM server — OpenAI-compatible, any hosted model
│       └── qwen_local.py        # Qwen3-VL-8B via transformers (local GPU)
│
├── scripts/
│   ├── download.py              # Entry point: download images + build metadata index
│   ├── relabel.py               # Entry point: simple single-caption relabeling
│   ├── annotate.py              # Entry point: multi-stage robotic annotation (A/B/C)
│   └── show_parquet.py          # Inspect any Parquet file; export PDF report
│
└── data/                        # Generated data (gitignored)
    ├── raw/                     # Downloaded images (000000.jpg, 000001.jpg, …)
    ├── metadata.parquet         # Image index: filename, caption, source_url, size
    ├── relabeled.parquet        # Simple relabeling output: + new_caption
    └── annotated.parquet        # Full annotation output: + type_a, type_b, type_c
```

## Data Flow

```
scripts/download.py
  └── downloads images → data/raw/
  └── writes data/metadata.parquet

scripts/relabel.py               (simple, single caption per image)
  └── reads  data/metadata.parquet
  └── writes data/relabeled.parquet  [+ new_caption]

scripts/annotate.py              (multi-stage robotic annotation)
  └── reads  data/metadata.parquet
  └── writes data/annotated.parquet  [+ type_a, type_b, type_c]
```

## Annotation Pipeline (annotate.py)

Three VLM calls run **in parallel** per image, followed by optional verification:

```
image + original_caption
    │
    ├─ [parallel] VLM call → Type A  (spatial scene description)
    ├─ [parallel] VLM call → Type B  (per-object referring expressions)
    └─ [parallel] VLM call → Type C  (action-conditioned caption)
    │
    └─ [optional, parallel] Verification — discard any caption with spatial errors
```

## VLM Backends

| Backend | Key | Best for |
|---|---|---|
| `openai` | `DATAGEN_OPENAI_API_KEY` | GPT-4o / GPT-4o-mini via API |
| `gemini` | `DATAGEN_GEMINI_API_KEY` | Gemini 2.0 Flash via API |
| `vllm` | — | Any model served by a vLLM server |
| `qwen_local` | — | Qwen3-VL-8B direct GPU inference |

To add a new backend: subclass `VLMBackend`, implement `call(image_bytes, prompt) -> str`, register in `vlm/__init__.py`.

## Configuration

Config is loaded in priority order (highest first):

1. Shell environment variables (`DATAGEN_*`)
2. `.env` file
3. `config.toml`
4. Defaults in `Config` class

**`config.toml` fields:**

| Field | Default | Description |
|---|---|---|
| `output_dir` | `data/raw` | Where downloaded images are saved |
| `timeout` | `10` | HTTP timeout for image downloads (seconds) |
| `metadata_path` | `data/metadata.parquet` | Input index for relabel/annotate |
| `relabeled_path` | `data/relabeled.parquet` | Output of relabel pipeline |
| `annotated_path` | `data/annotated.parquet` | Output of annotate pipeline |
| `vlm_backend` | `openai` | Backend to use |
| `vlm_model` | `gpt-4o-mini` | Model name passed to backend |
| `vlm_base_url` | `http://localhost:8000/v1` | vLLM server URL |
| `vlm_prompt` | `Describe this image…` | Prompt for simple relabeling |
| `concurrency` | `8` | Parallel workers (processes for API; GPU count for local) |
| `num_gpus_per_worker` | `0` | Set to `1` for `qwen_local` backend |
| `verify` | `true` | Run verification and discard incorrect captions |

## Common Commands

```bash
# Setup
cp .env.example .env              # then fill in your API key

# Download images
uv run python scripts/download.py
uv run python scripts/download.py --config configs/qwen_vllm.toml

# Simple relabeling (one new caption per image)
uv run python scripts/relabel.py
uv run python scripts/relabel.py --config configs/qwen_vllm.toml

# Multi-stage robotic annotation (Type A + B + C)
uv run python scripts/annotate.py
uv run python scripts/annotate.py --no-verify       # skip verification step
uv run python scripts/annotate.py --config configs/qwen_vllm.toml

# Inspect output
uv run python scripts/show_parquet.py                                      # print relabeled.parquet
uv run python scripts/show_parquet.py data/annotated.parquet               # print annotated.parquet
uv run python scripts/show_parquet.py --cols filename type_a type_b type_c # specific columns
uv run python scripts/show_parquet.py --head 20                            # first 20 rows
uv run python scripts/show_parquet.py --info                               # schema + row count
uv run python scripts/show_parquet.py --pdf report.pdf                     # PDF with images + captions
uv run python scripts/show_parquet.py data/annotated.parquet --pdf report.pdf

# Use vLLM server (start server first)
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192 --trust-remote-code
uv run python scripts/annotate.py --config configs/qwen_vllm.toml

# Override config via env vars (no file editing needed)
DATAGEN_CONCURRENCY=32 uv run python scripts/annotate.py
DATAGEN_VLM_MODEL=gpt-4o uv run python scripts/annotate.py
DATAGEN_VERIFY=false uv run python scripts/annotate.py
```

## Adding a Processing Stage

All per-row CPU work happens in `_process_row` in `pipeline.py` (for relabeling) and `_annotate_row` in `annotator.py` (for annotation). Add new stages there — they automatically run in parallel across all worker processes.

```python
# pipeline.py — _process_row
def _process_row(row: dict) -> dict:
    img_bytes = (_worker_img_dir / row["filename"]).read_bytes()
    row["new_caption"] = _worker_backend.call(img_bytes, _worker_cfg.vlm_prompt)
    # row["embedding"] = _worker_embedder.embed(img_bytes)   ← add here
    return row
```
