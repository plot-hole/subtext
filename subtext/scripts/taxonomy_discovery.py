"""
Taxonomy Discovery: Stage 1 of Adaptive Taxonomy Generation

Derives corpus-specific functional and/or emotional taxonomies from a
stratified sample of conversation summaries using Claude (direct API).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/taxonomy_discovery.py --taxonomy-type both
    python scripts/taxonomy_discovery.py --taxonomy-type functional --sample-size 100
    python scripts/taxonomy_discovery.py --taxonomy-type emotional --dry-run
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

CONV_PATH      = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
SUMMARIES_PATH = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
CONFIG_PATH    = os.path.join(BASE, "config", "quality_config.json")
INTERIM_DIR    = os.path.join(BASE, "data", "interim")

PROMPT_FUNCTIONAL = os.path.join(BASE, "prompts", "taxonomy_discovery_functional.txt")
PROMPT_EMOTIONAL  = os.path.join(BASE, "prompts", "taxonomy_discovery_emotional.txt")

os.makedirs(INTERIM_DIR, exist_ok=True)

# -- Tableau 20 palette for auto-assigning colors ----------------------------
TABLEAU_20 = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#F1CE63", "#A0CBE8", "#FFBE7D", "#8CD17D",
    "#D4A6C8", "#FABFD2", "#D7B5A6", "#79706E", "#D37295",
]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_data():
    """Load conversations and summaries, join on conversation_id."""
    print("\n-- Loading data ------------------------------------------------------")

    if not os.path.exists(CONV_PATH):
        print(f"ERROR: {CONV_PATH} not found")
        sys.exit(1)
    if not os.path.exists(SUMMARIES_PATH):
        print(f"ERROR: {SUMMARIES_PATH} not found")
        sys.exit(1)

    conv = pd.read_parquet(CONV_PATH)
    conv = conv[conv["is_analysable"]].copy()
    summaries = pd.read_csv(SUMMARIES_PATH)

    merged = conv.merge(summaries.drop(columns=["title"], errors="ignore"),
                        on="conversation_id", how="inner")
    merged = merged[merged["summary"].notna() & (merged["summary"].str.strip() != "")]
    merged = merged[~merged["summary"].str.contains(r"\[SUMMARIZATION FAILED\]", na=False)]

    print(f"  Analysable conversations : {len(conv):,}")
    print(f"  With valid summaries     : {len(merged):,}")
    return merged


def stratified_sample(df, sample_size, seed):
    """Proportional stratified sample by model_era, minimum 3 per stratum."""
    print(f"\n-- Stratified sampling (n={sample_size}, by model_era) ----------------")

    if len(df) <= sample_size:
        print(f"  Corpus smaller than sample size -- using all {len(df)} conversations")
        return df

    era_counts = df["model_era"].value_counts()
    total = len(df)

    allocations = {}
    for era, count in era_counts.items():
        alloc = max(3, round(count / total * sample_size))
        allocations[era] = min(alloc, count)

    # Adjust to hit target
    while sum(allocations.values()) > sample_size:
        largest = max(allocations, key=allocations.get)
        if allocations[largest] > 3:
            allocations[largest] -= 1
    while sum(allocations.values()) < sample_size:
        for era in allocations:
            if allocations[era] < era_counts[era]:
                allocations[era] += 1
                if sum(allocations.values()) >= sample_size:
                    break

    samples = []
    for era, n in allocations.items():
        era_df = df[df["model_era"] == era]
        samples.append(era_df.sample(n=n, random_state=seed))

    result = pd.concat(samples, ignore_index=True)
    print(f"  Sample size              : {len(result):,}")
    for era, n in sorted(allocations.items()):
        print(f"    {era:25s}: {n:3d}")
    return result


def format_sample_for_prompt(sample_df):
    """Format conversations as numbered list for the LLM."""
    lines = []
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        cid = row["conversation_id"]
        title = row.get("title", "") or "Untitled"
        summary = row.get("summary", "").strip()
        lines.append(f"[{i}] ID: {cid}\nTitle: {title}\nSummary: {summary}\n")
    return "\n".join(lines)


def run_discovery(client, prompt_path, sample_text, taxonomy_type, model):
    """Run a single discovery API call."""
    print(f"\n-- Running {taxonomy_type} discovery ----------------------------------")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    user_message = (
        f"Here are {taxonomy_type} classification summaries from the corpus:\n\n"
        f"{sample_text}"
    )

    print(f"  Model                    : {model}")
    print(f"  System prompt            : {prompt_path}")
    print(f"  User message length      : {len(user_message):,} chars")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    print(f"  Input tokens             : {input_tokens:,}")
    print(f"  Output tokens            : {output_tokens:,}")

    return raw_text.strip(), input_tokens, output_tokens


def parse_discovery_response(raw_text, taxonomy_type):
    """Parse and validate the LLM's JSON response."""
    import re
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)

    categories = parsed.get("categories", [])
    if len(categories) < 2:
        raise ValueError(f"Too few categories: {len(categories)}")
    if len(categories) > 20:
        raise ValueError(f"Too many categories: {len(categories)}")

    # Validate labels
    for cat in categories:
        label = cat.get("label", "")
        if not label or not label.replace("_", "").isalpha():
            raise ValueError(f"Invalid label: {label!r}")
        if "description" not in cat:
            raise ValueError(f"Category {label} missing description")

    # Check frequency sum
    freqs = [cat.get("estimated_frequency", 0) for cat in categories]
    freq_sum = sum(freqs)
    if freq_sum < 0.5 or freq_sum > 1.5:
        print(f"  WARNING: Frequency sum = {freq_sum:.2f} (expected ~1.0)")

    return parsed


def compute_entropy(categories):
    """Compute Shannon entropy from estimated frequencies."""
    freqs = [cat.get("estimated_frequency", 0) for cat in categories]
    freqs = [f for f in freqs if f > 0]
    if not freqs:
        return 0.0
    probs = np.array(freqs) / sum(freqs)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def assign_colors(categories):
    """Auto-assign colors from Tableau 20 palette."""
    for i, cat in enumerate(categories):
        if "color" not in cat or not cat["color"]:
            cat["color"] = TABLEAU_20[i % len(TABLEAU_20)]
    return categories


def build_taxonomy_json(parsed, taxonomy_type, sample_size, entropy):
    """Build the canonical taxonomy JSON structure."""
    today = datetime.now().strftime("%Y%m%d")
    categories = assign_colors(parsed.get("categories", []))

    # Compute validation stats
    freqs = [cat.get("estimated_frequency", 0) for cat in categories]
    max_freq = max(freqs) if freqs else 0
    min_freq = min(freqs) if freqs else 0

    taxonomy = {
        "schema_version": "1.0",
        "taxonomy_type": taxonomy_type,
        "taxonomy_id": f"{taxonomy_type}_proposed_{today}",
        "created_at": datetime.now().isoformat(),
        "derived_from_sample_size": sample_size,
        "source": "discovery",
        "categories": categories,
        "key_distinctions": parsed.get("key_distinctions", []),
        "uncategorized_conversations": parsed.get("uncategorized_conversations", []),
        "validation": {
            "shannon_entropy": round(entropy, 3),
            "max_category_frequency": round(max_freq, 3),
            "min_category_frequency": round(min_freq, 3),
        },
    }

    return taxonomy


def print_summary(taxonomy, taxonomy_type):
    """Print a human-readable summary of the discovered taxonomy."""
    categories = taxonomy["categories"]
    entropy = taxonomy["validation"]["shannon_entropy"]
    max_bits = np.log2(len(categories))

    print(f"\n{'='*70}")
    print(f"  DISCOVERED {taxonomy_type.upper()} TAXONOMY")
    print(f"{'='*70}")
    print(f"  Categories: {len(categories)}")
    print(f"  Shannon entropy: {entropy:.2f} bits (max = {max_bits:.2f})")

    if entropy < 2.0:
        print(f"  WARNING: Entropy below 2.0 -- taxonomy may be too coarse")

    print(f"\n  {'Label':<30s} {'Freq':>6s}  Description")
    print(f"  {'-'*28}  {'-'*6}  {'-'*40}")
    for cat in sorted(categories, key=lambda c: c.get("estimated_frequency", 0), reverse=True):
        label = cat["label"]
        freq = cat.get("estimated_frequency", 0)
        desc = cat["description"][:50]
        flag = " <-- SPLIT?" if freq > 0.25 else ""
        print(f"  {label:<30s} {freq:>5.1%}  {desc}{flag}")

    distinctions = taxonomy.get("key_distinctions", [])
    if distinctions:
        print(f"\n  Key distinctions:")
        for d in distinctions:
            print(f"    - {d}")

    uncategorized = taxonomy.get("uncategorized_conversations", [])
    if uncategorized:
        print(f"\n  Uncategorized boundary cases: {len(uncategorized)}")
        for u in uncategorized[:5]:
            print(f"    - {u.get('conversation_id', '?')}: {u.get('reason', '')[:60]}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Taxonomy Discovery: derive corpus-specific taxonomies from conversation summaries"
    )
    parser.add_argument("--taxonomy-type", choices=["functional", "emotional", "both"],
                        default="both", help="Which taxonomy to discover (default: both)")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of conversations to sample (default: 200)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(BASE, "taxonomies"),
                        help="Directory to save proposed taxonomy JSONs")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model for discovery (default: claude-sonnet-4-20250514)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling (default: from quality_config.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and sample, but don't call the API")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config()
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    print(f"  Random seed              : {seed}")

    df = load_data()
    sample = stratified_sample(df, args.sample_size, seed)
    sample_text = format_sample_for_prompt(sample)

    print(f"\n  Formatted sample         : {len(sample_text):,} chars")

    if args.dry_run:
        print(f"\n-- DRY RUN: Sample built ({len(sample)} conversations). Not calling API. --")
        # Save sample for inspection
        sample_path = os.path.join(INTERIM_DIR, "taxonomy_discovery_sample.csv")
        sample[["conversation_id", "title", "summary", "model_era"]].to_csv(
            sample_path, index=False
        )
        print(f"  Sample saved to: {sample_path}")
        return

    # -- API setup ------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        sys.exit(1)
    print("API key found: ****")

    import anthropic
    client = anthropic.Anthropic()

    today = datetime.now().strftime("%Y%m%d")
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    types_to_run = []
    if args.taxonomy_type in ("functional", "both"):
        types_to_run.append(("functional", PROMPT_FUNCTIONAL))
    if args.taxonomy_type in ("emotional", "both"):
        types_to_run.append(("emotional", PROMPT_EMOTIONAL))

    total_input = 0
    total_output = 0
    assignments = {}

    for tax_type, prompt_path in types_to_run:
        raw_text, inp_tok, out_tok = run_discovery(
            client, prompt_path, sample_text, tax_type, args.model
        )
        total_input += inp_tok
        total_output += out_tok

        try:
            parsed = parse_discovery_response(raw_text, tax_type)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ERROR parsing {tax_type} response: {e}")
            # Save raw response for debugging
            err_path = os.path.join(INTERIM_DIR, f"taxonomy_discovery_{tax_type}_raw.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"  Raw response saved to: {err_path}")
            continue

        entropy = compute_entropy(parsed["categories"])
        taxonomy = build_taxonomy_json(parsed, tax_type, len(sample), entropy)

        # Save proposed taxonomy
        out_path = os.path.join(args.output_dir, f"proposed_{tax_type}_{today}{seed_suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2)
        print(f"  Saved: {out_path}")

        print_summary(taxonomy, tax_type)

        # Store assignments for the review step
        for uc in parsed.get("uncategorized_conversations", []):
            cid = uc.get("conversation_id", "")
            assignments[cid] = {"category": "uncategorized", "reason": uc.get("reason", "")}

    # Save assignments
    assignments_path = os.path.join(INTERIM_DIR, "taxonomy_discovery_assignments.json")
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)

    # Cost summary
    # Sonnet pricing (non-batch): $3/MTok input, $15/MTok output
    cost = total_input / 1_000_000 * 3.0 + total_output / 1_000_000 * 15.0
    print(f"\n-- Cost Summary ------------------------------------------------------")
    print(f"  Input tokens             : {total_input:,}")
    print(f"  Output tokens            : {total_output:,}")
    print(f"  Estimated cost           : ${cost:.2f}")
    print(f"\nDone! Next step: python scripts/taxonomy_review.py --proposed-taxonomy <path>")


if __name__ == "__main__":
    main()
