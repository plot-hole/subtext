"""
Taxonomy Review: Stage 2 of Adaptive Taxonomy Generation

Generates a human-readable markdown report comparing a proposed taxonomy
against the default baseline. The human reviews the report, edits the
proposed taxonomy JSON directly, then runs classification with it.

Usage:
    python scripts/taxonomy_review.py --proposed-taxonomy taxonomies/proposed_functional_20260403.json
    python scripts/taxonomy_review.py --proposed-taxonomy taxonomies/proposed_emotional_20260403.json
    python scripts/taxonomy_review.py --proposed-taxonomy taxonomies/proposed_functional_20260403.json --output review.md
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMMARIES_PATH = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
REPORTS_DIR    = os.path.join(BASE, "outputs", "reports")

sys.path.insert(0, BASE)
from taxonomy import load_taxonomy, get_labels, DEFAULT_FUNCTIONAL_PATH, DEFAULT_EMOTIONAL_PATH


def load_summaries():
    """Load conversation summaries for example extraction."""
    if not os.path.exists(SUMMARIES_PATH):
        return {}
    df = pd.read_csv(SUMMARIES_PATH)
    return {
        str(row["conversation_id"]): {
            "title": row.get("title", "Untitled") or "Untitled",
            "summary": str(row.get("summary", "")),
        }
        for _, row in df.iterrows()
        if pd.notna(row.get("conversation_id"))
    }


def compute_diff(proposed_labels, default_labels):
    """Compute added/removed/shared categories between proposed and default."""
    proposed_set = set(proposed_labels)
    default_set  = set(default_labels)

    added   = sorted(proposed_set - default_set)
    removed = sorted(default_set - proposed_set)
    shared  = sorted(proposed_set & default_set)

    return added, removed, shared


def find_possible_renames(proposed_cats, default_cats, added, removed):
    """Heuristic: check if any added/removed pair has high word overlap in descriptions."""
    renames = []

    def tokenize(text):
        return set(text.lower().split())

    added_descs = {
        cat["label"]: tokenize(cat.get("description", ""))
        for cat in proposed_cats if cat["label"] in added
    }
    removed_descs = {
        cat["label"]: tokenize(cat.get("description", ""))
        for cat in default_cats if cat["label"] in removed
    }

    for a_label, a_tokens in added_descs.items():
        for r_label, r_tokens in removed_descs.items():
            if not a_tokens or not r_tokens:
                continue
            overlap = len(a_tokens & r_tokens) / min(len(a_tokens), len(r_tokens))
            if overlap > 0.4:
                renames.append((r_label, a_label, round(overlap, 2)))

    return renames


def generate_report(proposed_tax, default_tax, summaries):
    """Generate the full markdown review report."""
    proposed_cats   = proposed_tax["categories"]
    default_cats    = default_tax["categories"]
    proposed_labels = get_labels(proposed_tax)
    default_labels  = get_labels(default_tax)
    tax_type        = proposed_tax["taxonomy_type"]
    taxonomy_id     = proposed_tax.get("taxonomy_id", "unknown")
    sample_size     = proposed_tax.get("derived_from_sample_size", "?")

    # Compute metrics
    freqs = [c.get("estimated_frequency", 0) for c in proposed_cats]
    entropy = proposed_tax.get("validation", {}).get("shannon_entropy", 0)
    max_bits = np.log2(len(proposed_cats)) if proposed_cats else 0

    added, removed, shared = compute_diff(proposed_labels, default_labels)
    renames = find_possible_renames(proposed_cats, default_cats, added, removed)

    lines = []

    # -- Header ---------------------------------------------------------------
    lines.append(f"# Taxonomy Review: {tax_type.title()}")
    lines.append("")
    lines.append(f"- **Taxonomy ID:** {taxonomy_id}")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Derived from sample:** {sample_size} conversations")
    lines.append(f"- **Categories:** {len(proposed_cats)}")
    lines.append(f"- **Shannon entropy:** {entropy:.2f} bits (max = {max_bits:.2f})")
    lines.append("")

    if entropy < 2.0:
        lines.append("> **WARNING:** Entropy below 2.0 bits. The taxonomy may be too coarse -- one or two categories dominate. Consider splitting.")
        lines.append("")

    # -- Category table -------------------------------------------------------
    lines.append("## Proposed Categories")
    lines.append("")
    lines.append("| # | Label | Frequency | Description |")
    lines.append("|---|-------|-----------|-------------|")
    for i, cat in enumerate(sorted(proposed_cats, key=lambda c: c.get("estimated_frequency", 0), reverse=True), 1):
        label = cat["label"]
        freq  = cat.get("estimated_frequency", 0)
        desc  = cat.get("description", "")
        flag  = " **SPLIT?**" if freq > 0.25 else ""
        lines.append(f"| {i} | `{label}` | {freq:.1%}{flag} | {desc} |")
    lines.append("")

    # -- Boundary notes -------------------------------------------------------
    has_notes = any(c.get("boundary_notes") for c in proposed_cats)
    if has_notes:
        lines.append("## Boundary Notes")
        lines.append("")
        for cat in proposed_cats:
            notes = cat.get("boundary_notes")
            if notes:
                lines.append(f"- **{cat['label']}:** {notes}")
        lines.append("")

    # -- Key distinctions -----------------------------------------------------
    distinctions = proposed_tax.get("key_distinctions", [])
    if distinctions:
        lines.append("## Key Distinctions")
        lines.append("")
        for d in distinctions:
            lines.append(f"- {d}")
        lines.append("")

    # -- Diff against default -------------------------------------------------
    lines.append("## Diff Against Default Taxonomy")
    lines.append("")
    lines.append(f"Default taxonomy: `{default_tax.get('taxonomy_id', 'unknown')}` ({len(default_labels)} categories)")
    lines.append("")

    if shared:
        lines.append(f"**Shared ({len(shared)}):** {', '.join(f'`{s}`' for s in shared)}")
        lines.append("")

    if added:
        lines.append(f"**Added ({len(added)}):**")
        for label in added:
            cat = next((c for c in proposed_cats if c["label"] == label), {})
            lines.append(f"- `{label}`: {cat.get('description', '')}")
        lines.append("")

    if removed:
        lines.append(f"**Removed ({len(removed)}):**")
        for label in removed:
            cat = next((c for c in default_cats if c["label"] == label), {})
            lines.append(f"- `{label}`: {cat.get('description', '')}")
        lines.append("")

    if renames:
        lines.append("**Possible renames:**")
        for old, new, overlap in renames:
            lines.append(f"- `{old}` -> `{new}` (description overlap: {overlap:.0%})")
        lines.append("")

    if not added and not removed:
        lines.append("No differences from default taxonomy.")
        lines.append("")

    # -- Uncategorized boundary cases -----------------------------------------
    uncategorized = proposed_tax.get("uncategorized_conversations", [])
    if uncategorized:
        lines.append("## Uncategorized Boundary Cases")
        lines.append("")
        for u in uncategorized:
            cid = u.get("conversation_id", "?")
            reason = u.get("reason", "")
            summary_data = summaries.get(str(cid), {})
            title = summary_data.get("title", "?")
            lines.append(f"- **{cid}** ({title}): {reason}")
        lines.append("")

    # -- Split candidates -----------------------------------------------------
    split_candidates = [c for c in proposed_cats if c.get("estimated_frequency", 0) > 0.25]
    if split_candidates:
        lines.append("## Split Candidates (>25% frequency)")
        lines.append("")
        for cat in split_candidates:
            lines.append(f"- `{cat['label']}` at {cat['estimated_frequency']:.1%} -- consider whether this category covers distinct subcategories")
        lines.append("")

    # -- Instructions ---------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Review the categories above. Edit the proposed taxonomy JSON directly:")
    lines.append(f"   - Merge categories that are too granular")
    lines.append(f"   - Split categories flagged above (>25%)")
    lines.append(f"   - Rename labels that don't fit")
    lines.append(f"   - Add domain-specific categories the LLM missed")
    lines.append("")
    lines.append("2. Save the edited file as `taxonomies/approved_{type}_{date}.json`".format(
        type=tax_type, date=datetime.now().strftime("%Y%m%d")
    ))
    lines.append("")
    if tax_type == "functional":
        lines.append("3. Run classification:")
        lines.append(f"   ```")
        lines.append(f"   python scripts/13b_functional_classify.py --taxonomy-path taxonomies/approved_{tax_type}_{datetime.now().strftime('%Y%m%d')}.json")
        lines.append(f"   ```")
    elif tax_type == "emotional":
        lines.append("3. Run classification:")
        lines.append(f"   ```")
        lines.append(f"   python scripts/14b_emotional_state.py --taxonomy-path taxonomies/approved_{tax_type}_{datetime.now().strftime('%Y%m%d')}.json")
        lines.append(f"   ```")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Taxonomy Review: generate a human-readable report for proposed taxonomies"
    )
    parser.add_argument("--proposed-taxonomy", required=True,
                        help="Path to proposed taxonomy JSON from Stage 1")
    parser.add_argument("--default-taxonomy", default=None,
                        help="Path to default taxonomy JSON (auto-detected if not specified)")
    parser.add_argument("--output", default=None,
                        help="Path for output markdown report (auto-generated if not specified)")
    args = parser.parse_args()

    # Load proposed taxonomy
    proposed = load_taxonomy(args.proposed_taxonomy)
    tax_type = proposed["taxonomy_type"]
    print(f"Proposed taxonomy: {proposed.get('taxonomy_id', '?')} ({tax_type}, {len(proposed['categories'])} categories)")

    # Load default taxonomy
    if args.default_taxonomy:
        default_path = args.default_taxonomy
    elif tax_type == "functional":
        default_path = DEFAULT_FUNCTIONAL_PATH
    else:
        default_path = DEFAULT_EMOTIONAL_PATH

    default = load_taxonomy(default_path)
    print(f"Default taxonomy:  {default.get('taxonomy_id', '?')} ({len(default['categories'])} categories)")

    # Load summaries for examples
    summaries = load_summaries()
    print(f"Summaries loaded:  {len(summaries):,}")

    # Generate report
    report_md = generate_report(proposed, default, summaries)

    # Save report
    if args.output:
        out_path = args.output
    else:
        today = datetime.now().strftime("%Y%m%d")
        out_path = os.path.join(REPORTS_DIR, f"taxonomy_review_{tax_type}_{today}.md")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport saved: {out_path}")
    print(f"Edit the proposed taxonomy JSON, then run classification with --taxonomy-path.")


if __name__ == "__main__":
    main()
