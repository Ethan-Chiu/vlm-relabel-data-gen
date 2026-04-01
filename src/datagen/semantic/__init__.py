"""Semantic property and relationship extraction for the VLA captioning pipeline.

Modules
───────
  models.py    — SemanticAnnotation, ObjectProperties, Relationship, SceneContext
  extractor.py — SemanticExtractor: unified VLM call → parsed SemanticAnnotation
"""

from datagen.semantic.extractor import SemanticExtractor
from datagen.semantic.models import (
    ObjectProperties,
    Relationship,
    SceneContext,
    SemanticAnnotation,
)

__all__ = [
    "SemanticExtractor",
    "SemanticAnnotation",
    "SceneContext",
    "ObjectProperties",
    "Relationship",
]
