# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync                    # basic (API backends only)
uv sync --extra scene      # include CV models (RAM++, GroundingDINO, SAM, Depth)

# Run pipelines
uv run python scripts/download.py
uv run python scripts/relabel.py
uv run python scripts/annotate.py                          # robotic (default)
uv run python scripts/annotate.py --pipeline two-call
uv run python scripts/annotate.py --pipeline scene-graph

# Scene-graph prerequisites (GPU required)
uv run python scripts/setup_models.py                      # download ~10GB weights
uv run python scripts/extract_scene_graphs.py              # extract scene graphs

# Scene-graph extra install (non-PyPI packages, required once after uv sync --extra scene)
# See "Scene package install notes" section below for details.

# Inspect results
uv run python scripts/show_parquet.py
uv run python scripts/show_parquet.py --pdf report.pdf
```

No test framework or linter is configured.

## Architecture

**Data flow:** `download.py` → `metadata.parquet` → `relabel.py` or `annotate.py` → output parquet

### Configuration (`src/datagen/config.py`)
Pydantic `BaseSettings` with priority: env vars (`DATAGEN_*` prefix) > `.env` > `config.toml` > defaults. Config is serialized to dict for `ProcessPoolExecutor` pickling — do not add non-serializable fields without updating `__reduce__`.

### VLM Backends (`src/datagen/vlm/`)
Abstract `VLMBackend` with `call(image_bytes, prompt) -> str`. Concrete backends: `openai`, `gemini`, `vllm`, `qwen_local`. The factory `get_backend(config)` returns the appropriate backend by `config.vlm_backend`. Add new backends here and register in `__init__.py`.

### Pipelines

**`pipeline.py`** — simple per-row relabeling with a single VLM call per image. Uses `ProcessPoolExecutor` for API backends or a Ray actor pool for local GPU models (`qwen_local`). Workers initialize the backend once in `_worker_init()`.

**`annotator.py`** — three annotation strategies:
- `RoboticAnnotator`: 3 parallel VLM calls (Type A/B/C prompts) + parallel verification via inner `ThreadPoolExecutor`
- `TwoCallAnnotator`: 1 generate call + sequential verification
- `CachedSceneAnnotator`: like Robotic but reads pre-computed `scene_graph` text from parquet

**`scene_pipeline.py`** + `src/datagen/scene/` — GPU-intensive pre-processing stage. `SceneExtractor` chains RAM++ tagging → GroundingDINO detection → SAM segmentation → Depth Anything V2 → geometry assignment → text `scene_graph`. Outputs `scene_graphs.parquet` for use by `CachedSceneAnnotator`.

### Parallelism pattern
- Outer: `ProcessPoolExecutor` (one process per CPU/API concurrency) or Ray actor pool (one actor per GPU)
- Inner (annotator only): `ThreadPoolExecutor` for parallel A/B/C VLM calls within each worker process

### Prompts (`src/datagen/prompts.py`)
`TYPE_A/B/C` for robotic annotation, `TWO_CALL_GENERATE` for two-call pipeline, `SCENE_TYPE_A/B/C` for scene-graph-conditioned annotation, `VERIFY` for spatial error checking.

### Storage (`src/datagen/storage.py`)
All pipelines read/write Parquet via `read_metadata()` / `write_metadata()`. Writes append by reading existing data and concatenating.

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
