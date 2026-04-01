# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync                    # basic (API backends only)
uv sync --extra scene      # include CV models (RAM++, GroundingDINO, SAM, Depth)

# Download
uv run python scripts/download.py

# Annotation pipelines
uv run python scripts/annotate.py                          # robotic (default)
uv run python scripts/annotate.py --pipeline relabel
uv run python scripts/annotate.py --pipeline two-call
uv run python scripts/annotate.py --pipeline scene-graph
uv run python scripts/annotate.py --pipeline semantic

# Scene-graph pre-stage (GPU required)
uv run python scripts/setup_models.py                      # download ~10GB weights
uv run python scripts/extract_scene_graphs.py              # extract scene graphs

# Semantic pre-stage (API, requires scene graphs first)
uv run python scripts/extract_semantic.py                  # extract semantic annotations

# Inspect results
uv run python scripts/show_parquet.py
uv run python scripts/show_parquet.py --pdf report.pdf

# Scene package extra install (non-PyPI packages, required once after uv sync --extra scene)
# See "Scene package install notes" section below for details.
```

No test framework or linter is configured.

## Architecture

**Data flow:** `download.py` → `metadata.parquet` → `annotate.py` → output parquet

Scene/semantic pipelines add pre-stages:
- `extract_scene_graphs.py` → `scene_graphs.parquet` (GPU)
- `extract_semantic.py` → `semantic_annotations.parquet` (API, requires scene graphs)

### Configuration (`src/datagen/config.py`)
Pydantic `BaseSettings` with priority: env vars (`DATAGEN_*` prefix) > `.env` > `config.toml` > defaults. Config is serialized to dict for `ProcessPoolExecutor` pickling — do not add non-serializable fields without updating `__reduce__`.

Key fields: `annotate_limit` (first N rows, applied before skip filter), `overwrite` (re-annotate existing rows), `scene_graph_path`, `semantic_annotations_path`.

### VLM Backends (`src/datagen/vlm/`)
Abstract `VLMBackend` with `call(image_bytes, prompt) -> str`. Concrete backends: `openai`, `gemini`, `vllm`, `qwen_local`. The factory `get_backend(config)` returns the appropriate backend by `config.vlm_backend`. Add new backends here and register in `__init__.py`.

### Pipelines

**`annotator.py`** — pipeline runner; imports annotator classes from `datagen.annotators/`. Handles `ProcessPoolExecutor` workers, skip/overwrite/limit logic, and parquet I/O. Worker state lives in module globals (`_worker_annotator`, `_worker_img_dir`, `_worker_pipeline`).

**`annotators/`** — one class per pipeline strategy:
- `SimpleAnnotator` — single VLM call with `vlm_prompt` → `new_caption`
- `TwoCallAnnotator` — generate + sequential verify → `spatial_caption`
- `RoboticAnnotator(_ParallelVLMAnnotator)` — TYPE_A/B/C in parallel + `_verify_dict`
- `CachedSceneAnnotator(_ParallelVLMAnnotator)` — SCENE_TYPE_A/B/C + `_verify_dict`
- `SemanticAnnotator` — single caption from pre-extracted semantic annotations → `semantic_caption`
- `_ParallelVLMAnnotator` base class — shared `_run_parallel`, `_verify_dict`, inner `ThreadPoolExecutor`

**`scene_pipeline.py`** + `src/datagen/scene/` — GPU pre-processing. `SceneExtractor` chains RAM++ → GroundingDINO → SAM → Depth Anything V2 → text `scene_graph`. Outputs `scene_graphs.parquet`.

**`semantic_pipeline.py`** + `src/datagen/semantic/` — API pre-processing. `SemanticExtractor` makes a single VLM call (SEMANTIC_EXTRACT prompt) and parses structured semantic properties. Outputs `semantic_annotations.parquet` with `semantic_props` (scene context + per-object appearance/state/affordances) and `semantic_rels` (typed relationship triples).

### Parallelism pattern
- Outer: `ProcessPoolExecutor` (one process per CPU/API concurrency slot)
- Inner (robotic/scene annotators only): `ThreadPoolExecutor` for parallel A/B/C VLM calls within each worker process
- Semantic annotator: no inner pool (single call per image)

### Prompts (`src/datagen/prompts.py`)
- `TYPE_A/B/C` — robotic annotation
- `TWO_CALL_GENERATE` — two-call pipeline
- `SCENE_TYPE_A/B/C` — scene-graph-conditioned annotation
- `VERIFY` — spatial error checking for robotic/two-call/scene pipelines
- `SEMANTIC_EXTRACT` — structured extraction: scene context, per-object appearance/state/affordances, typed relationships
- `SEMANTIC_CAPTION` — single grounded paragraph; uses scene_graph + semantic annotations; outputs plain text (no JSON)
- `SEMANTIC_VERIFY` — holistic verification for semantic captions

### Storage (`src/datagen/storage.py`)
All pipelines read/write Parquet via `read_metadata()` / `write_metadata()`. The output path is configurable via `annotated_path` in config or `DATAGEN_ANNOTATED_PATH` env var — use this to route different pipeline runs to separate parquets.

## Scene package install notes

After `uv sync --extra scene`, three non-PyPI packages and several patches are required:

```bash
# 1. Add missing fairscale dependency (RAM++ requires it but doesn't declare it)
uv add fairscale

# 2. Install RAM++ and SAM
uv pip install git+https://github.com/xinyu1205/recognize-anything.git
uv pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. GroundingDINO — must patch for PyTorch 2.x before building
git clone --depth=1 https://github.com/IDEA-Research/GroundingDINO.git /tmp/groundingdino
CU=/tmp/groundingdino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
sed -i 's/\.data<scalar_t>()/.data_ptr<scalar_t>()/g; s/\.data<int64_t>()/.data_ptr<int64_t>()/g' $CU
sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' $CU
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' $CU
uv pip install --no-build-isolation /tmp/groundingdino

# 4. Patch RAM++ bert.py for transformers 4.x (functions moved to pytorch_utils)
#    In .venv/lib/python3.13/site-packages/ram/models/bert.py, replace:
#
#    from transformers.modeling_utils import (
#        PreTrainedModel,
#        apply_chunking_to_forward,
#        find_pruneable_heads_and_indices,
#        prune_linear_layer,
#    )
#
#    with:
#
#    from transformers.modeling_utils import PreTrainedModel
#    from transformers.pytorch_utils import (
#        apply_chunking_to_forward,
#        find_pruneable_heads_and_indices,
#        prune_linear_layer,
#    )
```
