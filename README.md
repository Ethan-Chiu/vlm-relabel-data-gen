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
│   ├── annotator.py             # All pipelines: Simple / TwoCall / Robotic / CachedScene
│   ├── scene_pipeline.py        # Scene graph extraction pipeline (GPU stage)
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
│   ├── annotate.py              # Entry point: all annotation pipelines
│   ├── extract_scene_graphs.py  # Entry point: GPU scene graph extraction (pre-stage)
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
    ├── scene_graphs.parquet     # Scene extraction output: + scene_graph, scene_detections
    └── annotated.parquet        # Annotation output (all pipelines write here)
```

## Data Flow

```
scripts/download.py
  └── downloads images → data/raw/
  └── writes data/metadata.parquet

scripts/annotate.py --pipeline relabel
  └── reads  data/metadata.parquet
  └── writes data/annotated.parquet      [+ new_caption]

scripts/annotate.py              (robotic annotation, default)
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

## Annotation Pipelines (annotate.py)

All pipelines share the same script and write to `annotated.parquet`.

| Pipeline | `--pipeline` | Output columns | VLM calls |
|---|---|---|---|
| Simple relabel | `relabel` | `new_caption` | 1 |
| Two-call | `two-call` | `spatial_caption` | 1–2 |
| Robotic | `robotic` (default) | `type_a`, `type_b`, `type_c` | 3–6 |
| Scene-graph | `scene-graph` | `scene_type_a`, `scene_type_b`, `scene_type_c` | 3–6 |

**Robotic pipeline** — three VLM calls in parallel, followed by optional verification:

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
| `metadata_path` | `data/metadata.parquet` | Input index for annotation pipelines |
| `annotated_path` | `data/annotated.parquet` | Output of all annotation pipelines |
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

# Download images (config only matters for output_dir / timeout / metadata_path)
uv run python scripts/download.py

# Annotation (all pipelines write to annotated.parquet)
uv run python scripts/annotate.py --pipeline relabel                       # single caption
uv run python scripts/annotate.py                                          # robotic (Type A+B+C)
uv run python scripts/annotate.py --pipeline two-call
uv run python scripts/annotate.py --no-verify       # skip verification step
uv run python scripts/annotate.py --config configs/qwen_vllm.toml

# Inspect output
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
uvx vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192 --trust-remote-code
uv run python scripts/annotate.py --config configs/qwen_vllm.toml

# Override config via env vars (no file editing needed)
DATAGEN_CONCURRENCY=32 uv run python scripts/annotate.py
DATAGEN_VLM_MODEL=gpt-4o uv run python scripts/annotate.py
DATAGEN_VERIFY=false uv run python scripts/annotate.py
```

## Adding a Processing Stage

All per-row work happens in `_annotate_row` in `annotator.py`. To add a new pipeline, create an annotator class with an `annotate(image_bytes, original_caption) -> dict` method, register it in `_worker_init`, and add it to the `--pipeline` choices in `scripts/annotate.py`.

```python
# annotator.py — example new annotator
class MyAnnotator:
    CAPTION_TYPES = ("my_output",)

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        result = self._backend.call(image_bytes, "my prompt")
        return {"my_output": result}
```
