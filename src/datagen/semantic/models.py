"""Dataclasses for the semantic extraction stage.

These represent the structured intermediate output of the SEMANTIC_EXTRACT
VLM call, before it is serialised to Parquet or fed into caption generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SceneContext:
    """Top-level scene classification and activity state."""
    environment: str
    apparent_activity: str | None = None
    temporal_phase: str | None = None          # "setup" | "in_progress" | "completed" | None
    confidence: str = "medium"                 # "high" | "medium"


@dataclass
class ObjectProperties:
    """Per-object semantic properties aligned to a scene-graph label."""
    label: str
    appearance: str | None = None
    appearance_confidence: str = "medium"      # "high" | "medium"
    state: str | None = None
    state_confidence: str = "medium"           # "high" | "medium"
    affordances: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A typed, evidenced relationship triple between two objects (or an object
    and a content noun such as "soup" or "liquid")."""
    subject: str
    predicate: str
    object: str
    type: str                                  # relationship category
    confidence: str                            # "high" | "medium"
    evidence: str                              # one-sentence visual grounding


@dataclass
class SemanticAnnotation:
    """Full semantic annotation for one image, produced by SemanticExtractor."""
    scene_context: SceneContext
    objects: list[ObjectProperties]
    relationships: list[Relationship]

    # ── Serialisation helpers ──────────────────────────────────────────────────

    def props_to_json(self) -> str:
        """Serialise scene_context + objects to a JSON string (stored as semantic_props)."""
        return json.dumps({
            "scene_context": {
                "environment": self.scene_context.environment,
                "apparent_activity": self.scene_context.apparent_activity,
                "temporal_phase": self.scene_context.temporal_phase,
                "confidence": self.scene_context.confidence,
            },
            "objects": [
                {
                    "label": o.label,
                    "appearance": o.appearance,
                    "appearance_confidence": o.appearance_confidence,
                    "state": o.state,
                    "state_confidence": o.state_confidence,
                    "affordances": o.affordances,
                }
                for o in self.objects
            ],
        })

    def rels_to_json(self) -> str:
        """Serialise relationships to a JSON string (stored as semantic_rels)."""
        return json.dumps({
            "relationships": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "type": r.type,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                }
                for r in self.relationships
            ],
        })

    @classmethod
    def from_json(cls, props_json: str, rels_json: str) -> SemanticAnnotation:
        """Reconstruct from the two Parquet column strings."""
        props = json.loads(props_json)
        rels = json.loads(rels_json)

        sc = props["scene_context"]
        scene_context = SceneContext(
            environment=sc["environment"],
            apparent_activity=sc.get("apparent_activity"),
            temporal_phase=sc.get("temporal_phase"),
            confidence=sc.get("confidence", "medium"),
        )

        objects = [
            ObjectProperties(
                label=o["label"],
                appearance=o.get("appearance"),
                appearance_confidence=o.get("appearance_confidence", "medium"),
                state=o.get("state"),
                state_confidence=o.get("state_confidence", "medium"),
                affordances=o.get("affordances", []),
            )
            for o in props.get("objects", [])
        ]

        relationships = [
            Relationship(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                type=r["type"],
                confidence=r.get("confidence", "medium"),
                evidence=r.get("evidence", ""),
            )
            for r in rels.get("relationships", [])
        ]

        return cls(scene_context=scene_context, objects=objects, relationships=relationships)
