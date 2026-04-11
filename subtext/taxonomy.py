"""
Shared taxonomy infrastructure for Subtext classification pipeline.

Provides a single source of truth for taxonomy loading, validation,
label/color extraction, and dynamic system prompt generation.

All classification scripts (13-19) import from here instead of
hardcoding VALID_FUNCTIONS, VALID_EMOTIONS, FUNC_COLORS, EMOTION_COLORS.
"""

import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FUNCTIONAL_PATH = os.path.join(BASE, "taxonomies", "default_functional.json")
DEFAULT_EMOTIONAL_PATH  = os.path.join(BASE, "taxonomies", "default_emotional.json")

REQUIRED_FIELDS = {"schema_version", "taxonomy_type", "taxonomy_id", "categories"}
REQUIRED_CATEGORY_FIELDS = {"label", "description"}


def load_taxonomy(path):
    """Read and validate a taxonomy JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        tax = json.load(f)

    missing = REQUIRED_FIELDS - set(tax.keys())
    if missing:
        raise ValueError(f"Taxonomy at {path} missing required fields: {missing}")

    if not tax["categories"] or len(tax["categories"]) < 2:
        raise ValueError(f"Taxonomy at {path} must have at least 2 categories")

    for i, cat in enumerate(tax["categories"]):
        cat_missing = REQUIRED_CATEGORY_FIELDS - set(cat.keys())
        if cat_missing:
            raise ValueError(f"Category {i} missing fields: {cat_missing}")
        if not cat["label"].replace("_", "").isalpha():
            raise ValueError(f"Category label must be lowercase_underscore: {cat['label']}")

    return tax


def get_labels(taxonomy):
    """Return ordered list of category labels."""
    return [c["label"] for c in taxonomy["categories"]]


def get_colors(taxonomy):
    """Return label -> color hex mapping."""
    return {c["label"]: c.get("color", "#999999") for c in taxonomy["categories"]}


def get_descriptions(taxonomy):
    """Return label -> description mapping."""
    return {c["label"]: c["description"] for c in taxonomy["categories"]}


def build_system_prompt(taxonomy):
    """Dynamically assemble the classification system prompt from taxonomy JSON.

    Produces the same format as the original hardcoded SYSTEM_PROMPT constants
    in scripts 13 and 14, but driven by taxonomy data.
    """
    tax_type = taxonomy["taxonomy_type"]

    if tax_type == "functional":
        return _build_functional_prompt(taxonomy)
    elif tax_type == "emotional":
        return _build_emotional_prompt(taxonomy)
    else:
        raise ValueError(f"Unknown taxonomy_type: {tax_type}")


def _build_functional_prompt(taxonomy):
    lines = [
        "You are a research assistant classifying conversations by their primary function.",
        "",
        "Given a summary of a conversation between a user and an AI assistant, classify the",
        "conversation into exactly ONE of these categories:",
        "",
    ]

    for cat in taxonomy["categories"]:
        desc = cat["description"]
        lines.append(f"- {cat['label']}: {desc}")

    lines.append("")
    lines.append("Choose the BEST single category. If a conversation spans multiple functions, choose the")
    lines.append("one that best describes the PRIMARY purpose.")

    distinctions = taxonomy.get("key_distinctions", [])
    if distinctions:
        lines.append("")
        lines.append("Key distinctions:")
        for d in distinctions:
            lines.append(f"- {d}")

    lines.extend([
        "",
        'Respond with ONLY a JSON object in this exact format:',
        '{"function": "<category>", "confidence": <0.0-1.0>, "secondary": ["<category>", ...]}',
        "",
        '- "function": The primary function label (one of the categories above)',
        '- "confidence": Your confidence in the primary classification (0.0 to 1.0)',
        '- "secondary": A list of 0-2 additional function labels if the conversation clearly spans',
        "  multiple functions. Empty list [] if single-function.",
        "",
        "Do not include any other text, explanation, or markdown formatting. Only the JSON object.",
    ])

    return "\n".join(lines)


def _build_emotional_prompt(taxonomy):
    lines = [
        "You are a research assistant classifying the emotional state of conversations.",
        "",
        "Given a summary of a conversation between a user and an AI assistant, classify the",
        "PRIMARY emotional state of the user during the conversation. Focus on the user's",
        "emotional state, not the AI's tone.",
        "",
        "Categories:",
        "",
    ]

    for cat in taxonomy["categories"]:
        desc = cat["description"]
        lines.append(f"- {cat['label']}: {desc}")

    distinctions = taxonomy.get("key_distinctions", [])
    if distinctions:
        lines.append("")
        lines.append("Key distinctions:")
        for d in distinctions:
            lines.append(f"- {d}")

    lines.extend([
        "",
        "Choose the BEST single category for the PRIMARY emotional state. Many conversations will",
        "blend states -- choose the dominant one.",
        "",
        'Respond with ONLY a JSON object in this exact format:',
        '{"emotion": "<category>", "confidence": <0.0-1.0>, "secondary": ["<category>", ...]}',
        "",
        '- "emotion": The primary emotional state label (one of the categories above)',
        '- "confidence": Your confidence in the primary classification (0.0 to 1.0)',
        '- "secondary": A list of 0-2 additional emotional state labels if the conversation clearly',
        "  carries multiple emotional textures. Empty list [] if single-state.",
        "",
        "Do not include any other text, explanation, or markdown formatting. Only the JSON object.",
    ])

    return "\n".join(lines)


def resolve_taxonomy_path(args, taxonomy_type):
    """Resolve CLI arguments to a taxonomy file path.

    Args:
        args: argparse Namespace with optional taxonomy_path attribute
        taxonomy_type: 'functional' or 'emotional'

    Returns:
        str: path to taxonomy JSON file
    """
    if hasattr(args, "taxonomy_path") and args.taxonomy_path:
        return args.taxonomy_path

    if taxonomy_type == "functional":
        return DEFAULT_FUNCTIONAL_PATH
    elif taxonomy_type == "emotional":
        return DEFAULT_EMOTIONAL_PATH
    else:
        raise ValueError(f"Unknown taxonomy_type: {taxonomy_type}")


def add_taxonomy_args(parser):
    """Add standard taxonomy CLI arguments to an argparse parser."""
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=None,
        help="Path to a taxonomy JSON file (overrides default)",
    )
    parser.add_argument(
        "--use-default-taxonomy",
        action="store_true",
        default=False,
        help="Explicitly use the default taxonomy (this is the default behavior)",
    )
