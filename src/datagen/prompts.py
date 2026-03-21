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

VERIFY = """\
Look at this image and the caption below. Answer only YES or NO:
Does the caption contain any spatial claims that are clearly \
wrong based on what you see?

Caption: {caption}\
"""
