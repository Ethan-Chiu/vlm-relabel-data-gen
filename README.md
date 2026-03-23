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
│   ├── prompts.py               # Prompt templates (TYPE_A/B/C, SCENE_TYPE_A/B/C, VERIFY)
│   ├── annotator.py             # RoboticAnnotator / TwoCallAnnotator / SceneGraphAnnotator
│   ├── pipeline.py              # Simple relabeling pipeline (ProcessPoolExecutor / Ray)
│   ├── vlm/
│   │   ├── base.py              # VLMBackend ABC — one method: call(image_bytes, prompt)
│   │   ├── openai_backend.py    # OpenAI API (GPT-4o, GPT-4o-mini, …)
│   │   ├── gemini.py            # Google Gemini API (gemini-2.0-flash, …)
│   │   ├── vllm.py              # vLLM server — OpenAI-compatible, any hosted model
│   │   └── qwen_local.py        # Qwen3-VL-8B via transformers (local GPU)
│   └── scene/
│       ├── models.py            # Detection dataclass
│       ├── grounded_sam.py      # RAM++ → GroundingDINO → SAM
│       ├── depth.py             # Depth Anything V2 (HuggingFace transformers)
│       ├── geometry.py          # Assign positions, ranks; build scene_graph text
│       └── extractor.py         # SceneExtractor — orchestrates all CV models
│
├── scripts/
│   ├── download.py              # Entry point: download images + build metadata index
│   ├── relabel.py               # Entry point: simple single-caption relabeling
│   ├── annotate.py              # Entry point: robotic / two-call / scene-graph annotation
│   ├── setup_models.py          # Download scene-graph model weights
│   └── show_parquet.py          # Inspect any Parquet file; export PDF report
│
├── configs/
│   ├── qwen_local.toml          # Qwen3-VL-8B via local transformers
│   ├── qwen_vllm.toml           # Qwen3-VL-8B via vLLM server
│   └── scene_graph.toml         # Scene-graph pipeline (RAM++ + GroundingDINO + SAM)
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
  └── writes data/relabeled.parquet      [+ new_caption]

scripts/annotate.py              (multi-stage robotic annotation)
  └── reads  data/metadata.parquet
  └── writes data/annotated.parquet      [+ type_a, type_b, type_c]

scripts/extract_scene_graphs.py  (scene graph pre-stage — GPU required)
  └── reads  data/metadata.parquet
  └── writes data/scene_graphs.parquet   [+ scene_graph, scene_detections]

scripts/annotate.py --pipeline scene-graph
  └── reads  data/metadata.parquet
  └── reads  data/scene_graphs.parquet   (must run extract_scene_graphs.py first)
  └── writes data/annotated.parquet      [+ scene_type_a, scene_type_b, scene_type_c]
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

## Scene-Graph Pipeline (annotate.py --pipeline scene-graph)

Uses computer vision models to extract a structured scene graph before calling the VLM.
Each image gets grounded object detections, segmentation masks, and metric depth estimates,
which are formatted into a compact text scene graph and passed as additional context to the VLM.

```
image + original_caption
    │
    ├─ RAM++            → open-vocabulary tags
    ├─ GroundingDINO    → bounding boxes per tag
    ├─ SAM              → segmentation masks per box
    └─ Depth Anything V2→ per-object depth value (median within mask)
    │
    └─ scene_graph (text: label, position, depth_rank, area_rank, free_space)
        │
        ├─ [parallel] VLM + scene_graph → scene Type A  (spatially grounded layout)
        ├─ [parallel] VLM + scene_graph → scene Type B  (grounded referring expressions)
        └─ [parallel] VLM + scene_graph → scene Type C  (action-conditioned caption)
        │
        └─ [optional, parallel] Verification
```

**Setup:**
```bash
# 1. Install scene dependencies (torch, transformers, etc.)
uv sync --extra scene

# 2. Install packages not on PyPI (do this inside the venv)
source .venv/bin/activate
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/xinyu1205/recognize-anything.git
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Download model weights (~10 GB total)
uv run python scripts/setup_models.py

# 4. Run
uv run python scripts/annotate.py --config configs/scene_graph.toml --pipeline scene-graph
```

**Hardware:** Requires a GPU with ≥12 GB VRAM (SAM vit_h: ~7 GB, RAM++: ~2 GB, Depth Large: ~1.5 GB).
Set `scene_device = "cpu"` for CPU inference (very slow, useful for testing).

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
| `scene_ram_weights` | `models/ram_plus_swin_large_14m.pth` | RAM++ weights path |
| `scene_gdino_config` | `models/GroundingDINO_SwinT_OGC.cfg.py` | GroundingDINO config |
| `scene_gdino_weights` | `models/groundingdino_swint_ogc.pth` | GroundingDINO weights |
| `scene_sam_weights` | `models/sam_vit_h_4b8939.pth` | SAM weights |
| `scene_sam_type` | `vit_h` | SAM variant (vit_h/vit_l/vit_b) |
| `scene_depth_model` | `depth-anything/Depth-Anything-V2-Large-hf` | Depth model (HF ID or path) |
| `scene_box_threshold` | `0.30` | GroundingDINO box score cutoff |
| `scene_text_threshold` | `0.25` | GroundingDINO text score cutoff |
| `scene_device` | `cuda` | Device for scene CV models |
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

# Scene-graph pipeline (two steps)
uv sync --extra scene
uv run python scripts/setup_models.py          # download ~10 GB of weights once

# Step 1: extract scene graphs (GPU, saves intermediate parquet)
uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml
uv run python scripts/show_parquet.py data/scene_graphs.parquet --info   # inspect

# Step 2: VLM annotation using cached scene graphs (API, no GPU needed)
uv run python scripts/annotate.py --pipeline scene-graph

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
