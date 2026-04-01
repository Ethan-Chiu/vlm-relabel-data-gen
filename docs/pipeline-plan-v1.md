# Semantic Enrichment Pipeline — Plan v1

> Simplified design: fewer LLM calls, no deterministic geometric stage,
> optional verification, VLM-determined caption diversity.
> Supersedes v0.

## Motivation

Enrich LAION image captions with VLA-relevant semantic information beyond the spatial
layout captured by the scene graph:

- Scene context (environment, apparent activity, temporal phase)
- Per-object properties (material, state, affordances)
- Typed relationships (part-whole, set membership, functional pairing, containment,
  support, activity-driven interactions)

## Key Design Decisions vs. v0

| | v0 | v1 |
|---|---|---|
| Extraction calls | 2 (parallel) | 1 (unified) |
| Geometric rule engine | Yes (Stage 2) | Removed — VLM handles geometry |
| Per-claim QA validation | Yes (selective) | Removed — rolled into optional holistic verify |
| Caption diversity | `focus_seed` hash → 1 caption | VLM selects applicable styles → 1–3 captions |
| Verification | Required, extended | Optional (config flag), same extended prompt |
| Total calls | 4–6 | 2–3 |

**Why remove the geometric rule engine**: hand-coded bbox + depth heuristics are brittle
under perspective distortion, diagonal arrangements, partial occlusion, and multi-object
clusters. The VLM reasons from the actual image and has world knowledge about physical
interactions — it is more reliable for geometric relationships than threshold-based rules.
The quantitative data (bboxes, depth values) is still passed to the VLM as context.

## Data Flow

```
LAION images
    │
    ▼  [existing, unchanged]
Stage 0 ── CV Extraction  (GPU)
    │         RAM++ → GroundingDINO → SAM → Depth Anything V2
    │         output: scene_graphs.parquet
    │                 columns: filename, scene_graph (text), scene_detections (JSON)
    │
    ▼  [new — run once, store]
Stage 1 ── Unified Semantic Extraction  (1 VLM call)
    │         input:  image + scene_graph + scene_detections
    │         output: scene context + per-object properties + typed relationships
    │                 stored in semantic_annotations.parquet
    │
    ▼  [new]
Stage 2 ── Enriched Caption Generation  (1 VLM call)
    │         input:  scene_graph + semantic_annotations
    │         VLM selects applicable focus styles, generates one caption per style
    │         output: list of 1–3 captioned paragraphs with focus labels
    │
    ▼  [optional — controlled by config verify flag]
Stage 3 ── Holistic Verification  (1 VLM call per caption)
              checks: spatial accuracy + semantic accuracy + over-claiming
              parallelized via ThreadPoolExecutor
              output: annotated_semantic.parquet
```

**Total VLM call budget: 2 calls (3 with verification).**

Stages 0 and 1 are extraction stages — run once and stored. Stage 2 onward is the
annotation stage, re-runnable cheaply with different prompt variants.

---

## Stage 1 — Unified Semantic Extraction

### Input

- `image_bytes`
- `scene_graph` text (existing format: labels, positions, depth ranks, free space)
- `scene_detections` JSON (bboxes, depth values, confidence, area ranks per object)

Passing `scene_detections` gives the VLM quantitative grounding for geometric relationships
(e.g., which object is closer, which bbox is above which) rather than requiring it to
estimate these purely from the image.

### Output Schema

```json
{
  "scene_context": {
    "environment": "kitchen counter",
    "apparent_activity": "meal preparation",
    "temporal_phase": "in_progress",
    "confidence": "high"
  },
  "objects": [
    {
      "label": "mug",
      "material": "ceramic",
      "material_confidence": "high",
      "state": "empty",
      "state_confidence": "medium",
      "affordances": ["graspable", "fillable"]
    }
  ],
  "relationships": [
    {
      "subject": "cup",
      "predicate": "set_with",
      "object": "saucer",
      "type": "set_membership",
      "confidence": "high",
      "evidence": "matching floral pattern and glaze color"
    },
    {
      "subject": "lid",
      "predicate": "rests_on",
      "object": "pot",
      "type": "support",
      "confidence": "high",
      "evidence": "lid sitting flush on pot rim, matching diameter"
    },
    {
      "subject": "knife",
      "predicate": "used_with",
      "object": "cutting_board",
      "type": "functional_pairing",
      "confidence": "medium",
      "evidence": "knife placed directly on cutting board surface"
    }
  ]
}
```

### Controlled Affordance Vocabulary

Closed set to limit hallucination:
`graspable, fillable, openable, stackable, pourable, cuttable, liftable`

### Controlled Relationship Predicate Vocabulary

| Type                 | Predicates                                        |
|----------------------|---------------------------------------------------|
| `part_whole`         | belongs_to, has_part, is_lid_of, is_handle_of     |
| `set_membership`     | set_with, matches, pairs_with                     |
| `containment`        | contains, filled_with, holds                      |
| `support`            | rests_on, stacked_on, mounted_on                  |
| `functional_pairing` | used_with, placed_for, positioned_beside          |
| `activity_driven`    | being_cut_by, being_poured_into, being_stirred_by |
| `proximity_semantic` | belongs_near, stored_with                         |

### Confidence Semantics

- `high`: claim is unambiguous from the image
- `medium`: claim is plausible with visible evidence but not certain
- `low`: not used in output — if confidence would be low, omit the claim entirely

---

## Stage 2 — Enriched Caption Generation

### Input

- `scene_graph` text
- `semantic_annotations` JSON (Stage 1 output)

### Behavior

Single VLM call. The model evaluates which caption opening styles are meaningfully
supported by the extracted facts and generates one paragraph per applicable style.

**Three focus styles:**

| Style              | When to generate                                                          |
|--------------------|---------------------------------------------------------------------------|
| `scene_activity`   | Only if `scene_context.confidence` is high and temporal phase is clear    |
| `object_centric`   | Always — lead with the most visually distinctive object and its properties |
| `relationship`     | Only if at least one relationship with `confidence: high` is present       |

The VLM must always generate at least one caption (`object_centric` is always viable).
A typical well-annotated kitchen image might yield all three; a sparse image might yield
only `object_centric`.

### Output Schema

```json
[
  {
    "focus": "scene_activity",
    "caption": "A meal is actively being prepared on a kitchen counter, with a pot of soup simmering on the stove..."
  },
  {
    "focus": "object_centric",
    "caption": "A ceramic mug sits empty near the center of the frame, its handle oriented to the right..."
  },
  {
    "focus": "relationship",
    "caption": "A ceramic cup and saucer, clearly a matching set by their shared floral pattern and cream glaze..."
  }
]
```

### Caption Requirements

- Single natural paragraph per caption, 80–150 words
- No bullet points, no enumeration
- Use hedged language for `confidence: medium` items: "appears to be", "suggesting", "likely"
- Prohibited: task prediction ("this could be used for..."), weight estimation, dynamics
  inferred from a single frame, events before or after what is shown

---

## Stage 3 — Holistic Verification (Optional)

Controlled by `verify: bool` in config (default: `false` for early runs).

One VLM call per caption from Stage 2, parallelized via `ThreadPoolExecutor`.

The VERIFY prompt checks three things in a single call:
1. Do any spatial claims in the caption contradict what is visible in the image?
2. Do any material, state, or relationship claims contradict visible evidence?
3. Does the caption assert anything that cannot be determined from a single static image
   (dynamics, future actions, weight, what happened before)?

Return `YES` (any issue found) → discard that caption.
Return `NO` → keep.

Each caption in the list is verified independently. An image may lose one focus variant
and keep others.

---

## Storage Schema

### `semantic_annotations.parquet` (Stage 1 output)

| Column              | Type   | Description                              |
|---------------------|--------|------------------------------------------|
| `filename`          | str    | matches scene_graphs.parquet             |
| `semantic_props`    | str    | JSON: scene_context + objects            |
| `semantic_rels`     | str    | JSON: relationships list                 |

Split into two columns (props vs. rels) for easier downstream filtering/analysis.

### `annotated_semantic.parquet` (Stage 2–3 output)

Base columns from merged metadata + scene_graphs + semantic_annotations, plus:

| Column              | Type   | Description                                        |
|---------------------|--------|----------------------------------------------------|
| `semantic_captions` | str    | JSON array of `{focus, caption}` objects           |

---

## New Code Components

```
scripts/
  extract_semantic.py          # CLI entry point for Stage 1
                               # mirrors extract_scene_graphs.py structure

src/datagen/
  semantic_pipeline.py         # ProcessPoolExecutor orchestration for Stage 1
                               # mirrors scene_pipeline.py structure

  semantic/
    __init__.py
    models.py                  # SemanticAnnotation, ObjectProperties,
                               # Relationship, CaptionVariant dataclasses
    extractor.py               # SemanticExtractor — single unified VLM call

  prompts.py                   # additions:
                               #   SEMANTIC_EXTRACT  (Stage 1)
                               #   SEMANTIC_CAPTION  (Stage 2)
                               #   VERIFY            (extended, Stage 3)

  annotator.py                 # + SemanticAnnotator class
  config.py                    # + semantic pipeline config fields
  storage.py                   # + read/write for semantic_annotations.parquet

scripts/annotate.py            # + --pipeline semantic
```

### Execution Order

```bash
# Stage 0 (existing)
uv run python scripts/extract_scene_graphs.py

# Stage 1 (new)
uv run python scripts/extract_semantic.py

# Stages 2–3 (new pipeline option)
uv run python scripts/annotate.py --pipeline semantic
uv run python scripts/annotate.py --pipeline semantic --verify  # with verification
```
