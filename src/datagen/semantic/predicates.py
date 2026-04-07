"""Canonical predicate vocabulary for the semantic extraction pipeline.

Organized by relationship type. build_predicate_prompt_block() formats this
for inclusion in the SEMANTIC_EXTRACT prompt.

Sources:
- spatial/activity_driven verbs: CALVIN (arXiv 2112.03227), BridgeData V2 (arXiv 2308.12952)
- containment/support: LIBERO (arXiv 2310.01943)
- occlusion/connectivity: nuScenes KG (arXiv 2312.09676)
- part_whole/set_membership/functional_pairing/proximity_semantic: original baseline
"""

PREDICATE_VOCAB: dict[str, list[str]] = {
    "spatial": [
        "left_of", "right_of", "in_front_of", "behind",
        "above", "below", "beside", "near", "adjacent_to", "facing",
    ],
    "support": [
        "rests_on", "stacked_on", "mounted_on",
        "attached_to", "leaning_on", "hanging_from",
    ],
    "containment": [
        "contains", "inside", "holds", "filled_with", "wrapped_in",
    ],
    "part_whole": [
        "has_part", "is_part_of", "is_lid_of", "is_handle_of",
        "belongs_to",
    ],
    "set_membership": [
        "set_with", "matches", "pairs_with", "grouped_with",
    ],
    "functional_pairing": [
        "used_with", "placed_for", "positioned_beside", "covers",
    ],
    "proximity_semantic": [
        "belongs_near", "stored_with",
    ],
    "activity_driven": [
        "being_cut_by", "being_poured_into", "being_stirred_by",
        "being_grasped_by", "being_lifted_by", "being_placed_on",
        "being_pushed_by", "being_folded", "being_wiped",
        "being_twisted", "being_opened", "being_closed",
        "being_flipped", "being_stacked_by",
    ],
    "occlusion": [
        "occluded_by", "partially_behind", "blocks_view_of",
    ],
    "connectivity": [
        "connected_to",
    ],
}

ALL_PREDICATES: frozenset[str] = frozenset(
    p for ps in PREDICATE_VOCAB.values() for p in ps
)


def build_predicate_prompt_block() -> str:
    """Format the vocabulary as a prompt-ready block."""
    lines = ["Relationship predicate vocabulary (use ONLY these predicates):"]
    for rel_type, predicates in PREDICATE_VOCAB.items():
        lines.append(f"  {rel_type + ':':22s} {', '.join(predicates)}")
    return "\n".join(lines)
