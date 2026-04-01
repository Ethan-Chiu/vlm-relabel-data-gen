# Semantic Enrichment Pipeline — Plan v0

> Checkpoint: initial full design before simplification.
> Status: reference / superseded by v1

## Motivation

The goal is to enrich LAION image captions with VLA-relevant semantic information beyond
the spatial layout already captured by the scene graph. Properties to add:

- Object state (empty/full, open/closed, on/off)
- Scene context (environment type, apparent activity, temporal phase)
- Physical properties (material, texture)
- Object affordances (graspable, fillable, openable, ...)
- Typed relationships: part-whole, set membership, functional pairing, containment,
  activity-driven interactions

## Data Flow

```
LAION images
    │
    ▼  [already done]
Stage 0 ── CV Extraction (GPU)
    │         RAM++ → GroundingDINO → SAM → Depth Anything V2
    │         output: scene_graphs.parquet
    │
    ▼  [new — run once, store]
Stage 1 ── Semantic Extraction (VLM API, 2 parallel calls)
    │         Call A: per-object properties + scene context
    │         Call B: typed relationships
    │         output: semantic_annotations.parquet
    │
    ▼  [new — deterministic, no VLM]
Stage 2 ── Geometric Relationship Augmentation
    │         derive contact/support/occlusion from bboxes + depth
    │         augments + hard-filters Stage 1 relationship output
    │
    ▼  [new — selective VLM]
Stage 3 ── Claim Validation
    │         Pass 1: geometric hard constraints filter
    │         Pass 2: binary QA for medium-confidence claims
    │         Pass 3: redundancy filter
    │         output: verified fact sheet per image
    │
    ▼  [new]
Stage 4 ── Enriched Caption Generation (1 VLM call)
    │         focus_seed biases opening: scene / object / relationship
    │         output: single natural paragraph, 80-150 words
    │
    ▼  [extended from existing]
Stage 5 ── Holistic Verification (1 VLM call)
              checks: spatial accuracy + semantic accuracy + over-claiming
              output: annotated_semantic.parquet
```

## Stage 1 — Semantic Extraction

### Call A — Object Properties + Scene Context

Output schema:
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
      "affordances": ["graspable", "fillable", "stackable"]
    }
  ]
}
```

Affordance vocabulary (closed set to reduce hallucination):
`graspable, fillable, openable, stackable, pourable, cuttable, liftable`

### Call B — Relationships

Output schema:
```json
{
  "relationships": [
    {
      "subject": "lid",
      "predicate": "belongs_to",
      "object": "pot",
      "type": "part_whole",
      "grounding": "visual",
      "confidence": "high",
      "evidence": "matching material and size, positioned on top"
    },
    {
      "subject": "cup",
      "predicate": "set_with",
      "object": "saucer",
      "type": "set_membership",
      "grounding": "visual",
      "confidence": "high",
      "evidence": "matching floral pattern and glaze color"
    }
  ]
}
```

Controlled predicate vocabulary by type:

| Type               | Predicates                                          |
|--------------------|-----------------------------------------------------|
| `part_whole`       | belongs_to, has_part, is_lid_of, is_handle_of       |
| `set_membership`   | set_with, matches, pairs_with                       |
| `containment`      | contains, filled_with, holds                        |
| `support`          | rests_on, stacked_on, mounted_on                    |
| `functional_pairing` | used_with, placed_for, positioned_beside          |
| `activity_driven`  | being_cut_by, being_poured_into, being_stirred_by   |
| `proximity_semantic` | belongs_near, stored_with                         |

Calls A and B run in parallel (both take image + scene_graph).

## Stage 2 — Geometric Relationship Augmentation (Deterministic)

Rules applied to `scene_detections` (bbox + depth_value per object):

**Support**: bbox_A bottom ≈ bbox_B top (within 5% image height), ≥30% horizontal overlap,
depth_A > depth_B → `A rests_on B`, grounding: geometric, confidence: high

**Occlusion**: IoU(bbox_A, bbox_B) > 40%, depth_A > depth_B
→ `A partially_behind B`, grounding: geometric, confidence: high

**Stacking**: multiple objects, similar bbox centers, increasing depth
→ `stack_of: [top, mid, bottom]`, grounding: geometric, confidence: high

Geometric relationships are injected directly into the fact sheet without VLM verification.
Any Stage 1 relationship contradicting geometry is discarded.

> **Note (v0 concern)**: these heuristics may be brittle for complex 3D arrangements,
> perspective distortion, diagonal stacking, etc. See v1 for revised approach.

## Stage 3 — Claim Validation

Three passes:

1. **Geometric hard constraints** (instant): discard VLM claims contradicting geometry;
   drop all `confidence: low` items
2. **Binary QA verification** (VLM, only `confidence: medium`): targeted yes/no questions
   using the `evidence` string from Stage 1 Call B as the question body
3. **Redundancy filter** (instant): remove claims already expressed by existing scene_graph

Parallelize QA calls with `ThreadPoolExecutor`.

## Stage 4 — Caption Generation

Single VLM call. Inputs: verified fact sheet + scene_graph.

- Single natural paragraph, 80-150 words
- `focus_seed` (hash of filename) biases opening: 0=scene activity, 1=object, 2=relationship
- Hedged language required for medium-confidence surviving items
- Prohibited: task prediction, weight estimation, dynamics from single frame

## Stage 5 — Holistic Verification

Extended VERIFY prompt — three checks in one call:
1. Spatial claims clearly wrong?
2. Semantic (material/state/relationship) claims clearly wrong?
3. Caption over-claims (dynamics, future actions, weight)?

`YES` → discard. Otherwise keep.

## VLM Call Budget per Image

| Stage    | Calls         | Parallelizable |
|----------|---------------|----------------|
| Stage 1  | 2             | Yes            |
| Stage 3  | 0–3 typical   | Yes            |
| Stage 4  | 1             | —              |
| Stage 5  | 1             | —              |
| **Total**| **4–6**       |                |

## New Code Components

```
scripts/
  extract_semantic.py          # CLI for Stage 1

src/datagen/
  semantic_pipeline.py         # ProcessPoolExecutor orchestration

  semantic/
    __init__.py
    models.py                  # SemanticAnnotation, ObjectProperties, Relationship
    extractor.py               # SemanticExtractor (parallel calls A + B)
    geometry_rels.py           # Stage 2 deterministic logic
    validator.py               # Stage 3 (geometric filter + QA)

  prompts.py                   # + SEMANTIC_PROPS, SEMANTIC_RELS,
                               #   SEMANTIC_VERIFY_QA, SEMANTIC_CAPTION, VERIFY (ext.)
  annotator.py                 # + SemanticAnnotator
  config.py                    # + semantic pipeline fields
  storage.py                   # + read/write semantic_annotations.parquet

scripts/annotate.py            # + --pipeline semantic
```
