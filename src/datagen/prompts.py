"""Prompt templates for the robotic annotation pipeline.

Each template takes keyword arguments via .format(**kwargs).
Keep prompt logic here and annotation logic in annotator.py.
"""

from datagen.semantic.predicates import build_predicate_prompt_block as _build_pred_block
_PREDICATE_VOCAB_BLOCK = _build_pred_block()

TYPE_A = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Look at the image carefully. Write a single concise paragraph \
describing the spatial layout of the scene.

Requirements:
- Mention every visible object with its position (left/center/right, near/far)
- Describe relative positions between objects
- Estimate approximate distances where possible
- Do NOT mention robot actions
- Be factual and precise, not poetic\
"""

TYPE_B = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Look at the image. For each distinct object in the scene, write:
1. One unambiguous referring expression using spatial cues \
(position, depth, size, color) so the object can be uniquely identified
2. One sentence describing the free space immediately around it

Format your response as:
Object: <label>
Referring: <sentence>
Free space: <sentence>

Repeat for every object.\
"""

TYPE_C = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Look at the image. Write a short paragraph from the perspective \
of a robot arm about to interact with this scene.

Requirements:
- For each object, identify the best approach direction for grasping
- Mention nearby obstacles for each target object
- Identify the clearest placement zone on the surface
- Be concise and action-relevant\
"""

TWO_CALL_GENERATE = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Look at the image carefully. Write a single natural paragraph that:
- Describes the spatial layout (what is where, near/far, left/right)
- Uniquely identifies each object using spatial and visual cues
- Notes free space and best approach directions for manipulation

Be concise, factual, and natural — like a scene briefing for a robot.\
"""

VERIFY = """\
Look at this image and the caption below. Answer only YES or NO:
Does the caption contain any spatial claims that are clearly \
wrong based on what you see?

Caption: {caption}\
"""

# ── Scene-graph-conditioned prompts (scene-graph pipeline) ────────────────────
# These receive an additional {scene_graph} argument from the SceneExtractor.

SCENE_TYPE_A = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

A computer vision system has detected the following objects and their spatial layout:

{scene_graph}

Look at the image and use the scene graph above as grounding. \
Write a single concise paragraph describing the spatial layout of the scene.

Requirements:
- Reference every object from the scene graph with its position
- Describe relative positions and distances between objects
- Correct any scene-graph entries that look wrong based on the image
- Do NOT mention robot actions
- Be factual and precise\
"""

SCENE_TYPE_B = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Detected objects with spatial context:

{scene_graph}

Look at the image. For each object listed in the scene graph, write:
1. One unambiguous referring expression using spatial cues \
(position, depth, size, color) so the object can be uniquely identified
2. One sentence describing the free space immediately around it \
(use or correct the free-space hint from the scene graph)

Format your response as:
Object: <label>
Referring: <sentence>
Free space: <sentence>

Repeat for every object.\
"""

SCENE_TYPE_C = """\
You are annotating images for robot manipulation training.

Original caption: {original_caption}

Scene layout from computer vision analysis:

{scene_graph}

Look at the image. Write a short paragraph from the perspective \
of a robot arm about to interact with this scene.

Requirements:
- Use the depth and position information from the scene graph
- For each object, identify the best approach direction for grasping
- Mention nearby obstacles (use depth_rank to reason about what is in the way)
- Identify the clearest placement zone based on the free-space information
- Be concise and action-relevant\
"""

# ── Semantic enrichment prompts (semantic pipeline) ───────────────────────────
# Stage 1: unified extraction of properties + relationships from image + scene graph.
# Stage 2: caption generation from verified semantic fact sheet.
# Stage 3 (optional): extended holistic verification.

SEMANTIC_EXTRACT = """\
You are extracting semantic properties and relationships from an image \
to build VLA (Vision-Language-Action) training data.

A computer vision system has detected the following objects:

{scene_graph}

Detection details (bounding boxes [x1,y1,x2,y2] as fractions of image dimensions; \
depth: 0.0 = nearest, 1.0 = farthest):
{scene_detections}

Look at the image carefully and return a single JSON object with this exact structure:

{{
  "scene_context": {{
    "environment": "<type of surface or room, e.g. kitchen counter, workshop bench, dining table>",
    "apparent_activity": "<what is happening or has just happened — null if unclear>",
    "temporal_phase": "<setup | in_progress | completed — null if unclear>",
    "confidence": "<high | medium>"
  }},
  "objects": [
    {{
      "label": "<label exactly as it appears in the scene graph line>",
      "position": "<position exactly as it appears in the scene graph line, e.g. right-near, center-mid>",
      "appearance": "<prominent visual traits visible in the image: color (e.g. bright red), \
surface material (e.g. ceramic, stainless steel, wood), texture (e.g. matte, glossy, rough), \
pattern (e.g. striped, floral), shape, or any other salient visible characteristic — \
combine multiple traits if needed, e.g. 'white ceramic with glossy finish'; null if nothing distinctive>",
      "appearance_confidence": "<high | medium>",
      "state": "<e.g. empty, full, open, closed, on, off, upright, inverted — null if not applicable or unclear>",
      "state_confidence": "<high | medium>",
      "affordances": ["<from this list only: graspable, fillable, openable, stackable, pourable, cuttable, liftable>"]
    }}
  ],
  "relationships": [
    {{
      "subject": "<label from scene graph>",
      "predicate": "<predicate from the vocabulary below>",
      "object": "<label from scene graph, OR a content noun such as soup or liquid>",
      "type": "<relationship type from vocabulary below>",
      "confidence": "<high | medium>",
      "evidence": "<one sentence: the specific visual signal that supports this claim>"
    }}
  ]
}}

__PREDICATE_VOCAB__

Rules:
- Include ONE entry in "objects" per numbered line in the scene graph — \
if the same label appears multiple times at different positions \
(e.g. several people), create one entry per occurrence with its own position.
- Omit any property or relationship where your confidence would be low — \
do not include it at all rather than guessing.
- Use the detection bounding boxes and depth values to reason about which \
objects are touching, stacked, or contained.
- Output only valid JSON. No markdown fences, no explanation outside the JSON.\
""".replace("__PREDICATE_VOCAB__", _PREDICATE_VOCAB_BLOCK)

SEMANTIC_CAPTION = """\
You are writing an image caption for VLA (Vision-Language-Action) robot training data.

Spatial layout detected by computer vision (use this to determine where things \
are and what is in front of or behind what — do NOT quote these numbers directly):
{scene_graph}

Semantic properties and relationships extracted from the image:
{semantic_annotation}

Write a single natural paragraph that describes the scene. \
The paragraph should cover:
- Where the main objects are (use natural spatial language: \
"on the left", "in the foreground", "behind the pool", \
"in the upper-right corner" — never cite raw depth ranks or bounding boxes)
- What the objects look like (color, material, texture, shape — \
only what is clearly visible)
- How objects relate to each other spatially or functionally \
(e.g. "rests on", "is stacked inside", "sits beside") — \
only when directly visible, using plain language

Length: 80–150 words for scenes with 4 or fewer objects; \
150–250 words for scenes with 5 or more objects.

Rules:
- Write one fluent paragraph; no lists, no section headers.
- Every object should appear in the description with a natural spatial reference.
- Translate depth/position data into plain English \
(nearest object = "in the foreground", farthest = "in the background", etc.).
- For medium-confidence properties use hedged language: \
"appears to be", "looks like", "likely".
- Avoid subjective, atmospheric, or emotive language: \
adjectives such as beautiful, vibrant, charming, picturesque, inviting, elegant, \
cozy, rustic, stunning, lush, luxurious, serene, relaxing, peaceful, and similar; \
phrases like "creating a relaxing atmosphere", "offering a serene view", \
"complementary elements", or any sentence that evaluates the mood or feeling \
of the scene rather than describing what is physically present.
- Do NOT mention "depth rank", "bounding box", "scene graph", \
"computer vision", or any pipeline internals.
- Do NOT predict future tasks, estimate weight, describe motion, \
or infer events not directly visible.
- Output only the paragraph text. No JSON, no markdown, no extra text.\
"""

SEMANTIC_VERIFY = """\
Look at the image and the caption below. Answer only YES or NO:

Does the caption contain any claim that is clearly wrong based on what \
you can see? This includes: incorrect spatial positions (left/center/right \
or near/far), wrong appearance (color, material, texture), wrong object \
states, incorrect relationships between objects, or confident assertions \
about things that cannot be determined from a single static image \
(dynamics, object weight, events before or after the moment shown).

Caption: {caption}\
"""

SEMANTIC_VERIFY_OBJECT = """\
You are verifying properties for one object detected in a scene image.

The cropped image shows: {label}

Currently extracted properties:
  appearance: {appearance}
  state: {state}

Look at the crop and return a JSON object:

{{
  "appearance": "<corrected or confirmed description; \
color, material, texture, shape; null if nothing distinctive is visible>",
  "appearance_confidence": "<high | medium>",
  "appearance_corrected": <true | false>,
  "state": "<corrected or confirmed state; \
e.g. empty, full, open, closed, on, off, upright, inverted; null if not applicable or unclear>",
  "state_confidence": "<high | medium>",
  "state_corrected": <true | false>
}}

Rules:
- Judge only from what is directly visible in this crop.
- If a property is already correct, return the same value with corrected=false.
- If it is wrong or significantly incomplete, provide a better description with corrected=true.
- Do NOT infer affordances, relationships, or scene context — focus only on appearance and state.
- Output only valid JSON. No markdown fences, no extra text.\
"""
