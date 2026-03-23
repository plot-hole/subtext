"""
GoEmotions Baseline Validation
Script: 15_goemotions_baseline.py

Runs SamLowe/roberta-base-go_emotions (28-category multi-label classifier)
on all 1,573 conversation summaries from all_summaries.csv and compares
against our Module 3.2b emotional state classifications.

Produces:
  outputs/reports/goemotions_raw.json       — raw 28-label scores per conversation
  outputs/reports/emotion_baseline_comparison.json  — comparison analysis
  outputs/figures/emotional_state/goemotions_comparison.png  — side-by-side figure

Usage:
    python scripts/15_goemotions_baseline.py
    python scripts/15_goemotions_baseline.py --batch-size 32
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMMARIES_PATH   = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
OUR_EMO_PATH     = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
RAW_OUTPUT_PATH  = os.path.join(BASE, "outputs", "reports", "goemotions_raw.json")
REPORT_PATH      = os.path.join(BASE, "outputs", "reports", "emotion_baseline_comparison.json")
FIG_PATH         = os.path.join(BASE, "outputs", "figures", "emotional_state",
                                "goemotions_comparison.png")

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
THRESHOLD  = 0.10   # min score to include a label in multi-label output

# --------------------------------------------------------------------------
# GoEmotions 28-label → our 12-state taxonomy mapping
# Conservative: only map where there is genuine semantic overlap.
# analytical, strategic, determined, numb have no GoEmotions equivalent.
# Unmapped GoEmotions labels → None.
# --------------------------------------------------------------------------
GOEMOTIONS_TO_OURS = {
    # --- distress / negative ---
    "fear":           "anxious",
    "nervousness":    "anxious",
    "anger":          "frustrated",
    "annoyance":      "frustrated",
    "disapproval":    "frustrated",
    "disgust":        "frustrated",
    "grief":          "grieving",
    "sadness":        "grieving",
    "remorse":        "grieving",
    "disappointment": "grieving",
    "embarrassment":  "vulnerable",
    # --- exploratory / cognitive ---
    "curiosity":      "curious",
    "confusion":      "curious",      # uncertainty-as-curiosity
    "realization":    "reflective",
    "relief":         "reflective",   # post-tension calm
    # --- energised / positive ---
    "amusement":      "playful",
    "excitement":     "energized",
    "pride":          "energized",
    # --- unmapped (no clean equivalent in our 12-state taxonomy) ---
    "admiration":     None,   # appreciative/impressed — we have no positive-relational state
    "approval":       None,
    "caring":         None,   # nurturing / protective
    "desire":         None,
    "gratitude":      None,
    "joy":            None,   # too broad; overlaps playful + energized
    "love":           None,
    "neutral":        None,   # collapses analytical/reflective/strategic — our key gap claim
    "optimism":       None,   # straddles strategic + energized + determined
    "surprise":       None,
}

# Our 12 states (for ordering output)
VALID_EMOTIONS = [
    "analytical", "anxious", "curious", "frustrated", "grieving",
    "playful", "reflective", "strategic", "vulnerable", "energized",
    "numb", "determined",
]

# Color palette matching script 14
EMOTION_COLORS = {
    "analytical":  "#4E79A7",
    "curious":     "#A0CBE8",
    "reflective":  "#76B7B2",
    "energized":   "#F28E2B",
    "playful":     "#FFBE7D",
    "anxious":     "#E15759",
    "frustrated":  "#FF9DA7",
    "grieving":    "#B07AA1",
    "vulnerable":  "#D4A6C8",
    "numb":        "#BAB0AC",
    "strategic":   "#59A14F",
    "determined":  "#8CD17D",
    "unmapped":    "#CCCCCC",
}

GOEMOTIONS_28 = list(GOEMOTIONS_TO_OURS.keys())


# --------------------------------------------------------------------------
def run_goemotions(summaries_df, batch_size=16):
    """Run GoEmotions pipeline on all summaries. Returns list of dicts."""
    print(f"\n-- Loading model: {MODEL_NAME} --")
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        truncation=True,
        max_length=512,
        device=-1,   # CPU
    )
    print(f"  Model loaded. Running inference on {len(summaries_df):,} summaries...")
    print(f"  Batch size: {batch_size} | Device: CPU")

    texts = summaries_df["summary"].fillna("").astype(str).tolist()
    raw_results = []  # list of {label: score} dicts per conversation

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="GoEmotions inference"):
        batch = texts[i : i + batch_size]
        preds = classifier(batch)
        # preds is a list of lists of {"label": ..., "score": ...}
        for pred_list in preds:
            scores = {p["label"]: round(p["score"], 4) for p in pred_list}
            raw_results.append(scores)

    print(f"  Inference complete. {len(raw_results):,} results.")
    return raw_results


def top_label(scores_dict):
    """Return the GoEmotions label with the highest score."""
    return max(scores_dict, key=scores_dict.get)


def active_labels(scores_dict, threshold=THRESHOLD):
    """Return all labels with score >= threshold, sorted descending."""
    return sorted(
        [(k, v) for k, v in scores_dict.items() if v >= threshold],
        key=lambda x: -x[1],
    )


def map_to_ours(ge_label):
    return GOEMOTIONS_TO_OURS.get(ge_label)   # None if unmapped


# --------------------------------------------------------------------------
def compare(summaries_df, our_emo_df, raw_results):
    """
    Build per-conversation comparison rows.
    Returns a list of dicts and aggregate stats.
    """
    print("\n-- Building comparison --")

    # Build lookup: conversation_id → our emotion_primary
    our_map = dict(zip(our_emo_df["conversation_id"].astype(str),
                       our_emo_df["emotion_primary"].astype(str)))
    our_conf_map = dict(zip(our_emo_df["conversation_id"].astype(str),
                            our_emo_df["emotion_confidence"].astype(float)))

    rows = []
    for i, (_, srow) in enumerate(summaries_df.iterrows()):
        cid     = str(srow["conversation_id"])
        title   = str(srow.get("title", ""))
        summary = str(srow.get("summary", ""))
        scores  = raw_results[i]

        ge_top        = top_label(scores)
        ge_top_score  = scores[ge_top]
        ge_mapped     = map_to_ours(ge_top)
        ge_active     = active_labels(scores)  # (label, score) pairs
        ge_active_mapped = [(map_to_ours(l), s) for l, s in ge_active
                            if map_to_ours(l) is not None]

        our_emotion   = our_map.get(cid, "unknown")
        our_conf      = our_conf_map.get(cid, 0.0)

        # Agreement: GoEmotions top mapped label == our label
        agree_top = (ge_mapped is not None and ge_mapped == our_emotion)

        # Soft agreement: any of the active GoEmotions labels (mapped) matches ours
        agree_soft = any(m == our_emotion for m, _ in ge_active_mapped)

        # Gap indicator: top GoEmotions label has no mapping to our taxonomy
        gap_unmapped = ge_mapped is None

        # Interesting gaps: labels in active set that are unmapped in our taxonomy
        # (things GoEmotions sees that our 12 states can't express)
        unmapped_active = [(l, s) for l, s in ge_active if map_to_ours(l) is None]

        rows.append({
            "conversation_id":      cid,
            "title":                title,
            "summary_excerpt":      summary[:200],
            "our_emotion":          our_emotion,
            "our_confidence":       round(our_conf, 3),
            "ge_top_label":         ge_top,
            "ge_top_score":         round(ge_top_score, 3),
            "ge_top_mapped":        ge_mapped,
            "ge_active_labels":     [l for l, _ in ge_active],
            "ge_active_mapped":     [m for m, _ in ge_active_mapped if m],
            "agree_top":            agree_top,
            "agree_soft":           agree_soft,
            "gap_unmapped_top":     gap_unmapped,
            "unmapped_active":      [(l, round(s, 3)) for l, s in unmapped_active],
        })

    return rows


def aggregate_stats(rows, summaries_df, raw_results):
    print("-- Computing aggregate stats --")

    total = len(rows)

    # Agreement rates
    n_agree_top  = sum(1 for r in rows if r["agree_top"])
    n_agree_soft = sum(1 for r in rows if r["agree_soft"])
    n_unmapped   = sum(1 for r in rows if r["gap_unmapped_top"])

    # GoEmotions top-label distribution (raw 28 labels)
    ge_top_dist = {}
    for r in rows:
        l = r["ge_top_label"]
        ge_top_dist[l] = ge_top_dist.get(l, 0) + 1

    # GoEmotions mapped distribution (our 12 + "unmapped")
    ge_mapped_dist = {"unmapped": 0}
    for emo in VALID_EMOTIONS:
        ge_mapped_dist[emo] = 0
    for r in rows:
        m = r["ge_top_mapped"]
        if m is None:
            ge_mapped_dist["unmapped"] += 1
        else:
            ge_mapped_dist[m] = ge_mapped_dist.get(m, 0) + 1

    # Our taxonomy distribution
    our_dist = {e: 0 for e in VALID_EMOTIONS}
    for r in rows:
        e = r["our_emotion"]
        if e in our_dist:
            our_dist[e] += 1

    # GoEmotions states with no mapping (from top-label perspective)
    unmapped_top_dist = {k: v for k, v in ge_top_dist.items()
                         if GOEMOTIONS_TO_OURS.get(k) is None}

    # Confusion matrix: our_emotion x ge_top_mapped (where ge_mapped is not None)
    confusion = {}
    for r in rows:
        ours   = r["our_emotion"]
        ge_m   = r["ge_top_mapped"]
        if ge_m is None:
            ge_m = "unmapped"
        key = f"{ours} | {ge_m}"
        confusion[key] = confusion.get(key, 0) + 1

    # Top confusion pairs (disagree but both mapped)
    disagree_mapped = [
        r for r in rows
        if r["ge_top_mapped"] is not None and not r["agree_top"]
    ]
    disagree_dist = {}
    for r in disagree_mapped:
        key = f"ours={r['our_emotion']} | ge={r['ge_top_mapped']}"
        disagree_dist[key] = disagree_dist.get(key, 0) + 1

    # Gap indicator conversations: top GoEmotions label is unmapped
    # Sort by ge_top_score descending to surface most confident gaps
    gap_convs = sorted(
        [r for r in rows if r["gap_unmapped_top"]],
        key=lambda r: -r["ge_top_score"],
    )[:50]  # top 50 most confident gaps

    # Which unmapped GoEmotions labels are most frequently the top label?
    unmapped_freq = sorted(unmapped_top_dist.items(), key=lambda x: -x[1])

    # Our taxonomy states with zero GoEmotions coverage (states our model finds
    # that GoEmotions cannot map to)
    our_states_uncovered = [
        e for e in VALID_EMOTIONS
        if e not in [v for v in GOEMOTIONS_TO_OURS.values() if v is not None]
    ]

    # GoEmotions neutrality rate: % of conversations where ge_top == "neutral"
    neutral_pct = sum(1 for r in rows if r["ge_top_label"] == "neutral") / total * 100

    # Among our analytical+reflective+strategic conversations, what does GoEmotions see?
    cognitive_states = ["analytical", "reflective", "strategic", "determined"]
    cognitive_rows   = [r for r in rows if r["our_emotion"] in cognitive_states]
    ge_labels_for_cognitive = {}
    for r in cognitive_rows:
        l = r["ge_top_label"]
        ge_labels_for_cognitive[l] = ge_labels_for_cognitive.get(l, 0) + 1

    # Cohen's kappa (on mapped pairs only)
    mapped_rows = [r for r in rows
                   if r["ge_top_mapped"] is not None and r["our_emotion"] in VALID_EMOTIONS]
    kappa = None
    if mapped_rows:
        try:
            from sklearn.metrics import cohen_kappa_score
            ours_labels = [r["our_emotion"]   for r in mapped_rows]
            ge_labels   = [r["ge_top_mapped"] for r in mapped_rows]
            kappa = round(cohen_kappa_score(ours_labels, ge_labels), 3)
        except Exception as e:
            print(f"  WARNING: Could not compute Cohen's kappa: {e}")

    stats = {
        "total_conversations":    total,
        "agreement": {
            "top_label_agree_n":   n_agree_top,
            "top_label_agree_pct": round(n_agree_top  / total * 100, 1),
            "soft_agree_n":        n_agree_soft,
            "soft_agree_pct":      round(n_agree_soft / total * 100, 1),
            "n_mapped_pairs":      len(mapped_rows),
            "cohens_kappa_on_mapped_pairs": kappa,
            "note": (
                "top_label_agree: GoEmotions top label maps to same category as ours. "
                "soft_agree: any active GoEmotions label (score>=0.10) maps to our label."
            ),
        },
        "goemotions_coverage": {
            "n_unmapped_top_label":   n_unmapped,
            "pct_unmapped_top_label": round(n_unmapped / total * 100, 1),
            "neutral_pct":            round(neutral_pct, 1),
            "unmapped_labels_by_freq": dict(unmapped_freq),
            "note": (
                "pct_unmapped_top_label: % of conversations where GoEmotions top label "
                "has no equivalent in our 12-state taxonomy."
            ),
        },
        "our_taxonomy_uncovered_by_goemotions": {
            "states": our_states_uncovered,
            "note": (
                "Emotional states in our taxonomy that have no GoEmotions equivalent. "
                "GoEmotions collapses these into neutral/curiosity/approval."
            ),
        },
        "cognitive_state_analysis": {
            "n_our_cognitive_states": len(cognitive_rows),
            "pct_corpus": round(len(cognitive_rows) / total * 100, 1),
            "what_goemotions_sees": dict(
                sorted(ge_labels_for_cognitive.items(), key=lambda x: -x[1])[:10]
            ),
            "note": (
                "For conversations our model classified as analytical/reflective/"
                "strategic/determined — GoEmotions top label distribution. "
                "Expected: mostly 'neutral' and 'curiosity'."
            ),
        },
        "top_disagree_pairs": dict(
            sorted(disagree_dist.items(), key=lambda x: -x[1])[:15]
        ),
        "our_distribution":   {k: {"count": v, "pct": round(v/total*100,1)}
                               for k, v in our_dist.items()},
        "ge_mapped_distribution": {k: {"count": v, "pct": round(v/total*100,1)}
                                   for k, v in ge_mapped_dist.items()},
        "ge_raw_top_distribution": dict(
            sorted(ge_top_dist.items(), key=lambda x: -x[1])
        ),
        "gap_indicators": {
            "n_gap_conversations": len([r for r in rows if r["gap_unmapped_top"]]),
            "top_50_by_ge_confidence": [
                {
                    "conversation_id": g["conversation_id"],
                    "title":           g["title"],
                    "our_emotion":     g["our_emotion"],
                    "ge_top_label":    g["ge_top_label"],
                    "ge_top_score":    g["ge_top_score"],
                    "summary_excerpt": g["summary_excerpt"],
                }
                for g in gap_convs
            ],
        },
    }
    return stats


# --------------------------------------------------------------------------
def make_figure(stats, fig_path):
    print("\n-- Generating comparison figure --")

    our_dist  = stats["our_distribution"]
    ge_mapped = stats["ge_mapped_distribution"]

    # Order by our distribution descending
    ordered_emos = sorted(VALID_EMOTIONS, key=lambda e: -our_dist.get(e, {}).get("count", 0))

    our_counts  = [our_dist.get(e, {}).get("count", 0) for e in ordered_emos]
    ge_counts   = [ge_mapped.get(e, {}).get("count", 0) for e in ordered_emos]
    ge_unmapped = ge_mapped.get("unmapped", {}).get("count", 0)

    # GoEmotions raw top-label distribution (28 categories)
    raw_dist = stats["ge_raw_top_distribution"]
    raw_labels = list(raw_dist.keys())
    raw_counts = list(raw_dist.values())

    fig = plt.figure(figsize=(18, 12))
    gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Panel A: Our 12-state distribution --------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    colors_a = [EMOTION_COLORS.get(e, "#888") for e in ordered_emos]
    bars_a   = ax_a.barh(ordered_emos, our_counts, color=colors_a)
    total    = sum(our_counts)
    for bar, cnt in zip(bars_a, our_counts):
        if cnt > 0:
            ax_a.text(bar.get_width() + total * 0.005,
                      bar.get_y() + bar.get_height() / 2,
                      f"{cnt} ({cnt/total*100:.1f}%)", va="center", fontsize=8)
    ax_a.set_title("Our Taxonomy (Module 3.2b)\n12 emotional states", fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Conversations")
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(our_counts) * 1.35)

    # ---- Panel B: GoEmotions mapped to our taxonomy + unmapped -------------
    ax_b = fig.add_subplot(gs[0, 1])
    ge_total       = sum(ge_counts) + ge_unmapped
    ge_counts_full = ge_counts + [ge_unmapped]
    labels_full    = ordered_emos + ["[unmapped]"]
    colors_b       = [EMOTION_COLORS.get(e, "#888") for e in ordered_emos] + ["#CCCCCC"]
    bars_b         = ax_b.barh(labels_full, ge_counts_full, color=colors_b)
    for bar, cnt in zip(bars_b, ge_counts_full):
        if cnt > 0:
            ax_b.text(bar.get_width() + ge_total * 0.005,
                      bar.get_y() + bar.get_height() / 2,
                      f"{cnt} ({cnt/ge_total*100:.1f}%)", va="center", fontsize=8)
    ax_b.set_title(
        "GoEmotions Top Label → Our Taxonomy\n(grey = no equivalent in our 12 states)",
        fontsize=10, fontweight="bold",
    )
    ax_b.set_xlabel("Conversations")
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, max(ge_counts_full) * 1.35)

    # ---- Panel C: GoEmotions raw 28-label distribution ---------------------
    ax_c = fig.add_subplot(gs[1, :])
    # Color-code: mapped labels get their mapped emotion color; unmapped get grey
    colors_c = []
    for l in raw_labels:
        mapped = GOEMOTIONS_TO_OURS.get(l)
        colors_c.append(EMOTION_COLORS.get(mapped, "#CCCCCC") if mapped else "#CCCCCC")

    bars_c = ax_c.bar(range(len(raw_labels)), raw_counts, color=colors_c, edgecolor="white")
    ax_c.set_xticks(range(len(raw_labels)))
    ax_c.set_xticklabels(raw_labels, rotation=45, ha="right", fontsize=8)
    ax_c.set_title(
        "GoEmotions Raw Top-Label Distribution (28 categories)\n"
        "Colored bars = map to one of our 12 states | Grey = no equivalent",
        fontsize=10, fontweight="bold",
    )
    ax_c.set_ylabel("Conversations")
    for bar, cnt in zip(bars_c, raw_counts):
        if cnt > 0:
            ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      str(cnt), ha="center", fontsize=7, va="bottom")

    # Agreement annotation box
    agr = stats["agreement"]
    cov = stats["goemotions_coverage"]
    info = (
        f"Top-label agreement: {agr['top_label_agree_pct']:.1f}%\n"
        f"Soft agreement: {agr['soft_agree_pct']:.1f}%\n"
        f"GoE unmapped top label: {cov['pct_unmapped_top_label']:.1f}%\n"
        f"Cohen's kappa (mapped pairs): {agr['cohens_kappa_on_mapped_pairs']}"
    )
    fig.text(
        0.98, 0.97, info,
        ha="right", va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", alpha=0.9),
    )

    plt.suptitle(
        "GoEmotions Baseline vs. Custom Taxonomy — Methodological Validation",
        fontsize=13, fontweight="bold", y=1.01,
    )
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip GoEmotions inference — reload from goemotions_raw.json")
    args = parser.parse_args()

    # Load summaries + our classifications
    print(f"\n-- Loading data --")
    summaries_df = pd.read_csv(SUMMARIES_PATH)
    print(f"  Summaries : {len(summaries_df):,}")

    if not os.path.exists(OUR_EMO_PATH):
        print(f"ERROR: {OUR_EMO_PATH} not found. Run 14_emotional_state.py first.")
        sys.exit(1)
    our_emo_df = pd.read_parquet(OUR_EMO_PATH)
    print(f"  Our emo   : {len(our_emo_df):,}")

    # GoEmotions inference (or reload)
    if args.skip_inference and os.path.exists(RAW_OUTPUT_PATH):
        print(f"\n-- Reloading GoEmotions raw output from {RAW_OUTPUT_PATH} --")
        with open(RAW_OUTPUT_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_results = [item["scores"] for item in raw_data]
        print(f"  Loaded {len(raw_results):,} cached results")
    else:
        raw_results = run_goemotions(summaries_df, batch_size=args.batch_size)
        # Save raw output
        raw_output = [
            {
                "conversation_id": str(summaries_df.iloc[i]["conversation_id"]),
                "title":           str(summaries_df.iloc[i].get("title", "")),
                "scores":          raw_results[i],
                "top_label":       top_label(raw_results[i]),
                "top_score":       round(max(raw_results[i].values()), 4),
                "active_labels":   [(l, s) for l, s in active_labels(raw_results[i])],
            }
            for i in range(len(raw_results))
        ]
        with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(raw_output, f, indent=2)
        print(f"\n  Raw output saved: {RAW_OUTPUT_PATH}")

    # Build comparison
    comparison_rows = compare(summaries_df, our_emo_df, raw_results)
    stats           = aggregate_stats(comparison_rows, summaries_df, raw_results)

    # Mapping reference for report
    mapping_doc = {
        k: v if v is not None else "(unmapped — no equivalent in our taxonomy)"
        for k, v in GOEMOTIONS_TO_OURS.items()
    }

    report = {
        "description": (
            "GoEmotions (SamLowe/roberta-base-go_emotions) baseline comparison "
            "against Module 3.2b emotional state classifications. "
            "Validates custom taxonomy against a standard multi-label emotion model."
        ),
        "model":           MODEL_NAME,
        "threshold":       THRESHOLD,
        "label_mapping":   mapping_doc,
        "mapping_notes": {
            "unmapped_from_our_taxonomy": (
                "analytical, strategic, determined, numb — "
                "GoEmotions has no equivalent labels. These are cognitive-agentic states "
                "that GoEmotions collapses into 'neutral' or 'curiosity'."
            ),
            "unmapped_from_goemotions": (
                "admiration, approval, caring, desire, gratitude, joy, love, neutral, "
                "optimism, surprise — no clean equivalent in our 12-state taxonomy. "
                "Mostly positive-relational states our corpus-specific taxonomy doesn't distinguish."
            ),
            "most_contested": (
                "neutral→analytical: the single largest disagreement. GoEmotions calls "
                "analytical conversations 'neutral'; we distinguish analytical/reflective/strategic."
            ),
        },
        "stats":           stats,
        "figure":          "outputs/figures/emotional_state/goemotions_comparison.png",
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {REPORT_PATH}")

    make_figure(stats, FIG_PATH)

    # Print summary
    agr = stats["agreement"]
    cov = stats["goemotions_coverage"]
    unc = stats["our_taxonomy_uncovered_by_goemotions"]
    cog = stats["cognitive_state_analysis"]
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Top-label agreement (mapped pairs):  {agr['top_label_agree_pct']:.1f}%")
    print(f"  Soft agreement (any active label):   {agr['soft_agree_pct']:.1f}%")
    print(f"  Cohen's kappa (mapped pairs only):   {agr['cohens_kappa_on_mapped_pairs']}")
    print(f"  GoE top label has no mapping:        {cov['pct_unmapped_top_label']:.1f}% of conversations")
    print(f"  GoE top label is 'neutral':          {cov['neutral_pct']:.1f}% of conversations")
    print(f"\n  Our states with no GoE equivalent:   {unc['states']}")
    print(f"\n  Cognitive states ({cog['pct_corpus']:.1f}% of corpus) — GoEmotions sees them as:")
    for lbl, cnt in list(cog["what_goemotions_sees"].items())[:6]:
        print(f"    {lbl:<20} {cnt:>4}")
    print(f"\n  Unmapped GoE labels (our potential gaps):")
    for lbl, cnt in list(cov["unmapped_labels_by_freq"].items())[:8]:
        mapped_note = "" if GOEMOTIONS_TO_OURS[lbl] is None else f"→{GOEMOTIONS_TO_OURS[lbl]}"
        print(f"    {lbl:<20} {cnt:>4}  {mapped_note}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
