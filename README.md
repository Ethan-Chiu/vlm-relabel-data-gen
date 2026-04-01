# data-gen

Large-scale image dataset generation and VLM annotation pipeline for robot manipulation training.

## Project Structure

```
data-gen/
├── config.toml                  # Runtime config (edit this to change behaviour)
├── .env                         # Secrets — API keys (gitignored, copy from .env.example)
├── .env.example                 # Template for .env
│
├── configs/                     # Alternative config files for different backends/pipelines
│   ├── download.toml            # Download config (source URL, output paths)
│   ├── annotate_openai.toml     # GPT-4o annotation with verification
│   ├── annotate_gemini.toml     # Gemini Robotics-ER annotation with verification
│   ├── qwen_local.toml          # Qwen3-VL-8B via local transformers
│   ├── qwen_vllm.toml           # Qwen3-VL-8B via vLLM server
│   └── scene_graph.toml         # Scene-graph pipeline (RAM++ + GroundingDINO + SAM)
│
├── src/datagen/
│   ├── config.py                # Pydantic-settings Config (TOML + env vars)
│   ├── download.py              # Download images → data/raw/, write metadata.parquet
│   ├── storage.py               # Parquet read/write helpers
│   ├── prompts.py               # All prompt templates
│   ├── annotator.py             # Pipeline runner: worker init, skip/overwrite, parquet I/O
│   ├── scene_pipeline.py        # Scene graph extraction pipeline (GPU pre-stage)
│   ├── semantic_pipeline.py     # Semantic extraction pipeline (API pre-stage)
│   │
│   ├── annotators/              # One class per annotation strategy
│   │   ├── base.py              # _ParallelVLMAnnotator base + verify_caption helper
│   │   ├── simple.py            # SimpleAnnotator   — single VLM call → new_caption
│   │   ├── two_call.py          # TwoCallAnnotator  — generate + verify → spatial_caption
│   │   ├── robotic.py           # RoboticAnnotator  — Type A/B/C in parallel
│   │   ├── scene.py             # CachedSceneAnnotator — scene-graph-conditioned A/B/C
│   │   └── semantic.py          # SemanticAnnotator — grounded single caption
│   │
│   ├── vlm/
│   │   ├── base.py              # VLMBackend ABC — one method: call(image_bytes, prompt)
│   │   ├── openai_backend.py    # OpenAI API (GPT-4o, GPT-4o-mini, …)
│   │   ├── gemini.py            # Google Gemini API (gemini-2.0-flash, …)
│   │   ├── vllm.py              # vLLM server — OpenAI-compatible, any hosted model
│   │   └── qwen_local.py        # Qwen3-VL-8B via transformers (local GPU)
│   │
│   ├── scene/                   # CV models for scene graph extraction
│   │   ├── models.py            # Detection dataclass
│   │   ├── grounded_sam.py      # RAM++ → GroundingDINO → SAM
│   │   ├── depth.py             # Depth Anything V2 (HuggingFace transformers)
│   │   ├── geometry.py          # Assign positions, ranks; build scene_graph text
│   │   └── extractor.py         # SceneExtractor — orchestrates all CV models
│   │
│   └── semantic/                # Structured semantic extraction (pre-caption stage)
│       ├── models.py            # SemanticAnnotation, ObjectProperties, Relationship dataclasses
│       └── extractor.py         # SemanticExtractor — VLM call → parsed SemanticAnnotation
│
├── scripts/
│   ├── download.py              # Entry point: download images + build metadata index
│   ├── annotate.py              # Entry point: all annotation pipelines
│   ├── extract_scene_graphs.py  # Entry point: GPU scene graph extraction (pre-stage)
│   ├── extract_semantic.py      # Entry point: API semantic extraction (pre-stage)
│   ├── setup_models.py          # Download scene-graph model weights
│   └── show_parquet.py          # Inspect any Parquet file; export PDF report
│
└── data/                        # Generated data (gitignored)
    ├── raw/                     # Downloaded images (000000.jpg, 000001.jpg, …)
    ├── metadata.parquet         # Image index: filename, caption, source_url, size
    ├── scene_graphs.parquet     # Scene extraction output: + scene_graph, scene_detections
    ├── semantic_annotations.parquet  # Semantic extraction output: + semantic_props, semantic_rels
    └── annotated.parquet        # Default annotation output (configurable per run)
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

scripts/extract_semantic.py      (semantic pre-stage — API, no GPU)
  └── reads  data/scene_graphs.parquet   (must run extract_scene_graphs.py first)
  └── writes data/semantic_annotations.parquet  [+ semantic_props, semantic_rels]

scripts/annotate.py --pipeline semantic
  └── reads  data/metadata.parquet
  └── reads  data/scene_graphs.parquet
  └── reads  data/semantic_annotations.parquet
  └── writes data/annotated.parquet      [+ semantic_caption]
```

## Annotation Pipelines (annotate.py)

All pipelines share the same script. The output parquet path is configurable via `annotated_path` in config or `DATAGEN_ANNOTATED_PATH` env var.

| Pipeline | `--pipeline` | Output columns | VLM calls |
|---|---|---|---|
| Simple relabel | `relabel` | `new_caption` | 1 |
| Two-call | `two-call` | `spatial_caption` | 1–2 |
| Robotic | `robotic` (default) | `type_a`, `type_b`, `type_c` | 3–6 |
| Scene-graph | `scene-graph` | `scene_type_a`, `scene_type_b`, `scene_type_c` | 3–6 |
| Semantic | `semantic` | `semantic_caption` | 1 (+1 optional verify) |

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
# 1. Install scene dependencies
uv sync --extra scene
uv add fairscale   # required by RAM++ but not declared as a dependency

# 2. Install packages not on PyPI
uv pip install git+https://github.com/xinyu1205/recognize-anything.git
uv pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. GroundingDINO — patch for PyTorch 2.x incompatibilities before building
git clone --depth=1 https://github.com/IDEA-Research/GroundingDINO.git /tmp/groundingdino
CU=/tmp/groundingdino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
sed -i 's/\.data<scalar_t>()/.data_ptr<scalar_t>()/g; s/\.data<int64_t>()/.data_ptr<int64_t>()/g' $CU
sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' $CU
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' $CU
uv pip install --no-build-isolation /tmp/groundingdino

# 4. Patch RAM++ for transformers 4.x (functions moved out of modeling_utils)
python3 - <<'EOF'
import site, pathlib
bert = pathlib.Path(site.getsitepackages()[0]) / "ram/models/bert.py"
old = ("from transformers.modeling_utils import (\n"
       "    PreTrainedModel,\n"
       "    apply_chunking_to_forward,\n"
       "    find_pruneable_heads_and_indices,\n"
       "    prune_linear_layer,\n"
       ")")
new = ("from transformers.modeling_utils import PreTrainedModel\n"
       "from transformers.pytorch_utils import (\n"
       "    apply_chunking_to_forward,\n"
       "    find_pruneable_heads_and_indices,\n"
       "    prune_linear_layer,\n"
       ")")
bert.write_text(bert.read_text().replace(old, new))
print("patched", bert)
EOF

# 5. Download model weights (~10 GB total)
uv run python scripts/setup_models.py

# 6. Run
uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml
uv run python scripts/annotate.py --pipeline scene-graph
```

**Hardware:** Requires a GPU with ≥12 GB VRAM (SAM vit_h: ~7 GB, RAM++: ~2 GB, Depth Large: ~1.5 GB).
Set `scene_device = "cpu"` for CPU inference (very slow, useful for testing).

## Semantic Pipeline (annotate.py --pipeline semantic)

Two-stage API pipeline that extracts structured semantic properties before generating a grounded caption.

```
Stage 0 (GPU, shared with scene-graph pipeline):
  image → SceneExtractor → scene_graphs.parquet
                           [scene_graph text, scene_detections JSON]

Stage 1 (API, extract_semantic.py):
  image + scene_graph + scene_detections
    └─ VLM call (SEMANTIC_EXTRACT prompt)
    └─ parsed into: scene_context, per-object appearance/state/affordances,
                    typed inter-object relationships with visual evidence
    └─ writes semantic_annotations.parquet
       [semantic_props JSON, semantic_rels JSON]

Stage 2 (API, annotate.py --pipeline semantic):
  image + scene_graph + semantic_props + semantic_rels
    └─ VLM call (SEMANTIC_CAPTION prompt)
    └─ single grounded paragraph: spatial positions, object appearance,
       visible relationships — no subjective language, no pipeline internals
    └─ [optional] verification call (--verify)
    └─ writes semantic_caption column
```

**Run:**
```bash
# Prerequisites: scene graphs must already be extracted
uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml

# Stage 1: extract semantic properties
uv run python scripts/extract_semantic.py
uv run python scripts/extract_semantic.py --limit 100   # first 100 rows only

# Stage 2: generate grounded captions
uv run python scripts/annotate.py --pipeline semantic
DATAGEN_ANNOTATED_PATH=data/annotated_semantic.parquet \
  uv run python scripts/annotate.py --pipeline semantic --verify
```

**Output columns:**

| Column | Source | Description |
|---|---|---|
| `semantic_props` | Stage 1 | JSON: `scene_context` + per-object `appearance`, `state`, `affordances` |
| `semantic_rels` | Stage 1 | JSON: typed relationship triples with visual evidence |
| `semantic_caption` | Stage 2 | Single grounded paragraph describing the scene |

## VLM Backends

| Backend | Key | Best for |
|---|---|---|
| `openai` | `DATAGEN_OPENAI_API_KEY` | GPT-4o / GPT-4o-mini via API |
| `gemini` | `DATAGEN_GEMINI_API_KEY` | Gemini Robotics-ER 1.5 / Gemini 2.0 Flash via API |
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
| `annotated_path` | `data/annotated.parquet` | Output of annotation pipelines |
| `scene_graph_path` | `data/scene_graphs.parquet` | Scene graph intermediate output |
| `semantic_annotations_path` | `data/semantic_annotations.parquet` | Semantic extraction intermediate output |
| `annotate_limit` | `null` | Process only the first N rows (applied before skip filter) |
| `overwrite` | `false` | Re-annotate rows that already exist in output |
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
uv run python scripts/download.py --config configs/download.toml

# Annotation (all pipelines)
uv run python scripts/annotate.py                                          # robotic (Type A+B+C)
uv run python scripts/annotate.py --pipeline relabel                       # single caption
uv run python scripts/annotate.py --pipeline two-call
uv run python scripts/annotate.py --pipeline scene-graph
uv run python scripts/annotate.py --pipeline semantic
uv run python scripts/annotate.py --no-verify                              # skip verification
uv run python scripts/annotate.py --limit 100                              # first 100 rows only
uv run python scripts/annotate.py --overwrite                              # re-annotate existing rows
uv run python scripts/annotate.py --config configs/annotate_openai.toml   # use GPT-4o
uv run python scripts/annotate.py --config configs/annotate_gemini.toml   # use Gemini Robotics-ER

# Route output to a separate file (useful when running multiple configs)
DATAGEN_ANNOTATED_PATH=data/annotated_openai_robotic.parquet \
  uv run python scripts/annotate.py --config configs/annotate_openai.toml

# Inspect output
uv run python scripts/show_parquet.py data/annotated.parquet               # print table
uv run python scripts/show_parquet.py --cols filename type_a type_b type_c # specific columns
uv run python scripts/show_parquet.py --head 20                            # first 20 rows
uv run python scripts/show_parquet.py --info                               # schema + row count
uv run python scripts/show_parquet.py --pdf report.pdf                     # PDF with images + captions
uv run python scripts/show_parquet.py data/annotated_semantic.parquet --pdf semantic.pdf

# Scene-graph pipeline (two steps)
uv sync --extra scene
uv run python scripts/setup_models.py          # download ~10 GB of weights once

uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml
uv run python scripts/annotate.py --pipeline scene-graph

# Semantic pipeline (three steps)
uv run python scripts/extract_scene_graphs.py --config configs/scene_graph.toml  # GPU stage
uv run python scripts/extract_semantic.py                                          # API stage 1
uv run python scripts/annotate.py --pipeline semantic                             # API stage 2

# Use vLLM server (start server first)
uvx vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192 --trust-remote-code
uv run python scripts/annotate.py --config configs/qwen_vllm.toml

# Override config via env vars (no file editing needed)
DATAGEN_CONCURRENCY=32 uv run python scripts/annotate.py
DATAGEN_VLM_MODEL=gpt-4o uv run python scripts/annotate.py
DATAGEN_VERIFY=false uv run python scripts/annotate.py
```

## Adding a New Annotation Pipeline

Create an annotator class in `src/datagen/annotators/`, register it in `annotators/__init__.py`, add a branch in `annotator.py`'s `_worker_init` and `_annotate_row`, and add the `--pipeline` choice in `scripts/annotate.py`.

```python
# src/datagen/annotators/my_annotator.py
class MyAnnotator:
    CAPTION_TYPES = ("my_output",)

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def annotate(self, image_bytes: bytes, original_caption: str) -> dict:
        result = self._backend.call(image_bytes, "my prompt")
        return {"my_output": result}
```
