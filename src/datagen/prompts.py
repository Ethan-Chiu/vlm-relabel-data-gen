"""Prompt templates for the robotic annotation pipeline.

Each template takes keyword arguments via .format(**kwargs).
Keep prompt logic here and annotation logic in annotator.py.
"""

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
