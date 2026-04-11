"""
Module 3.2d: Conversation Length × Classification Analysis
Script: 16_length_weight_analysis.py

Joins conversations_clean with functional_classifications and emotional_states to
compute count-based vs message-weighted and token-weighted distributions for each
functional category and emotional state.

The key finding lives in the gap between the left (count) and right (message-weighted)
panel of the output figure: categories over-represented by short conversations shrink
when weighted by messages; categories with few but long conversations grow.

Usage:
    python scripts/16_length_weight_analysis.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV_PATH  = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
FUNC_PATH  = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH  = os.path.join(BASE, "data", "processed", "emotional_states.parquet")

FIG_DIR    = os.path.join(BASE, "outputs", "figures", "length_analysis")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")

OUT_FIGURE  = os.path.join(FIG_DIR, "weighted_distributions.png")
OUT_REPORT  = os.path.join(REPORT_DIR, "length_weight_report.json")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DPI = 150
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"

FUNC_COLORS = {
    "interpersonal_analysis": "#4E79A7",
    "emotional_processing":   "#F28E2B",
    "creative_expression":    "#E15759",
    "career_strategy":        "#76B7B2",
    "self_modeling":          "#59A14F",
    "practical":              "#EDC948",
    "learning":               "#B07AA1",
    "problem_solving":        "#FF9DA7",
    "coding":                 "#9C755F",
    "social_rehearsal":       "#BAB0AC",
    "work_professional":      "#86BCB6",
    "planning":               "#F1CE63",
}

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
}


# -- Data loading & joining --------------------------------------------------
def load_data():
    print("Loading data...")
    conv = pd.read_parquet(CONV_PATH)
    func = pd.read_parquet(FUNC_PATH)
    emot = pd.read_parquet(EMOT_PATH)

    # Only analysable conversations, exclude unknowns
    conv = conv[conv["is_analysable"]].copy()

    # Total tokens per conversation
    conv["total_tokens"] = (
        conv["user_token_total"].astype(int) +
        conv["assistant_token_total"].astype(int)
    )

    # Join
    df = conv[["conversation_id", "msg_count", "turns", "total_tokens",
               "user_token_total", "assistant_token_total"]].copy()
    df = df.merge(
        func[["conversation_id", "function_primary"]],
        on="conversation_id", how="left",
    )
    df = df.merge(
        emot[["conversation_id", "emotion_primary"]],
        on="conversation_id", how="left",
    )

    # Drop any conversation where either classification is missing or unknown
    df = df[
        df["function_primary"].astype(str).ne("unknown") &
        df["emotion_primary"].astype(str).ne("unknown") &
        df["function_primary"].notna() &
        df["emotion_primary"].notna()
    ].copy()

    print(f"  Conversations in analysis: {len(df):,}")
    print(f"  Total messages:            {df['msg_count'].sum():,}")
    print(f"  Total tokens:              {df['total_tokens'].sum():,}")
    return df


# -- Stats computation -------------------------------------------------------
def compute_weighted_stats(df, group_col, color_map):
    """
    Returns a DataFrame with count-based and message/token-weighted distributions.
    Columns:
        category, n, pct_count, sum_msgs, pct_msgs, mean_msgs, median_msgs,
        sum_tokens, pct_tokens, mean_tokens, median_tokens,
        msg_weight_ratio, token_weight_ratio
    """
    total_n      = len(df)
    total_msgs   = df["msg_count"].sum()
    total_tokens = df["total_tokens"].sum()

    rows = []
    for cat, grp in df.groupby(group_col, observed=True):
        cat = str(cat)
        n           = len(grp)
        sum_msgs    = grp["msg_count"].sum()
        mean_msgs   = grp["msg_count"].mean()
        median_msgs = grp["msg_count"].median()
        sum_tokens  = grp["total_tokens"].sum()
        mean_tokens = grp["total_tokens"].mean()
        median_tokens = grp["total_tokens"].median()

        pct_count  = n / total_n * 100
        pct_msgs   = sum_msgs / total_msgs * 100
        pct_tokens = sum_tokens / total_tokens * 100

        rows.append({
            "category":         cat,
            "n":                n,
            "pct_count":        round(pct_count, 2),
            "sum_msgs":         int(sum_msgs),
            "pct_msgs":         round(pct_msgs, 2),
            "mean_msgs":        round(mean_msgs, 1),
            "median_msgs":      float(median_msgs),
            "sum_tokens":       int(sum_tokens),
            "pct_tokens":       round(pct_tokens, 2),
            "mean_tokens":      round(mean_tokens, 0),
            "median_tokens":    float(median_tokens),
            # ratio > 1.0 → longer-than-average convos (grows when weighted)
            # ratio < 1.0 → shorter-than-average convos (shrinks when weighted)
            "msg_weight_ratio":   round(pct_msgs / pct_count, 3) if pct_count else None,
            "token_weight_ratio": round(pct_tokens / pct_count, 3) if pct_count else None,
        })

    stats = pd.DataFrame(rows).sort_values("pct_count", ascending=False).reset_index(drop=True)
    return stats


def print_stats_table(stats, label):
    print(f"\n{'='*72}")
    print(f"  {label} — Count vs Message-Weighted Distribution")
    print(f"{'='*72}")
    header = f"{'Category':<28} {'N':>6} {'%Cnt':>7} {'%Msg':>7} {'%Tok':>7}  {'MsgRatio':>9}  {'Mean msg':>9}  {'Med msg':>8}"
    print(header)
    print("-" * 72)
    for _, row in stats.iterrows():
        flag = ""
        if row["msg_weight_ratio"] is not None:
            if row["msg_weight_ratio"] > 1.25:
                flag = " ▲"   # grows when weighted — long convos
            elif row["msg_weight_ratio"] < 0.75:
                flag = " ▼"   # shrinks when weighted — short convos
        print(
            f"{row['category']:<28} {row['n']:>6,} {row['pct_count']:>7.1f} "
            f"{row['pct_msgs']:>7.1f} {row['pct_tokens']:>7.1f}  "
            f"{row['msg_weight_ratio']:>9.3f}  {row['mean_msgs']:>9.1f}  {row['median_msgs']:>8.1f}"
            f"{flag}"
        )


# -- Figure ------------------------------------------------------------------
def _panel_bars(ax, stats, pct_col, color_map, title, xlabel,
                order=None, show_delta=None, delta_ref_col=None):
    """
    Horizontal bar chart for one panel.
    order:       list of category names (defines top-to-bottom order)
    show_delta:  if True, annotate with Δ pp vs. delta_ref_col values
    """
    if order is None:
        order = stats["category"].tolist()

    # Re-index to order (reverse so top item is at top of horizontal chart)
    stats_ord = stats.set_index("category").reindex(order[::-1])

    values = stats_ord[pct_col].values
    cats   = stats_ord.index.tolist()
    colors = [color_map.get(c, COLOR_SECONDARY) for c in cats]

    bars = ax.barh(cats, values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)

    # Annotate with value and delta
    for i, (bar, cat, val) in enumerate(zip(bars, cats, values)):
        label = f"{val:.1f}%"
        if show_delta is not None and delta_ref_col is not None:
            ref_val = stats_ord.loc[cat, delta_ref_col] if cat in stats_ord.index else None
            if ref_val is not None:
                delta = val - ref_val
                sign  = "+" if delta >= 0 else ""
                label = f"{val:.1f}%  ({sign}{delta:.1f} pp)"

        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8, color="#333333",
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xlim(0, max(values) * 1.55)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(func_stats, emot_stats):
    print("\nGenerating figure...")

    func_order = func_stats.sort_values("pct_count", ascending=False)["category"].tolist()
    emot_order = emot_stats.sort_values("pct_count", ascending=False)["category"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Count-Based vs Message-Weighted Distributions\n"
        "Gap between panels reveals which categories are over/under-represented by conversation length",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ---- Row 0: Functional ---------------------------------------------------
    _panel_bars(
        axes[0, 0], func_stats, "pct_count", FUNC_COLORS,
        "Functional Category — By Conversation Count",
        "% of conversations",
        order=func_order,
    )
    _panel_bars(
        axes[0, 1], func_stats, "pct_msgs", FUNC_COLORS,
        "Functional Category — By Message Count (weighted)",
        "% of all messages",
        order=func_order,
        show_delta=True, delta_ref_col="pct_count",
    )

    # ---- Row 1: Emotional ----------------------------------------------------
    _panel_bars(
        axes[1, 0], emot_stats, "pct_count", EMOTION_COLORS,
        "Emotional State — By Conversation Count",
        "% of conversations",
        order=emot_order,
    )
    _panel_bars(
        axes[1, 1], emot_stats, "pct_msgs", EMOTION_COLORS,
        "Emotional State — By Message Count (weighted)",
        "% of all messages",
        order=emot_order,
        show_delta=True, delta_ref_col="pct_count",
    )

    # Legend for delta annotation
    legend_text = (
        "Δ pp = message-weighted % minus count-based %\n"
        "Positive Δ → category has longer-than-average conversations\n"
        "Negative Δ → category has shorter-than-average conversations"
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=8.5,
             color=COLOR_SECONDARY, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(OUT_FIGURE, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_FIGURE}")


# -- Report ------------------------------------------------------------------
def save_report(func_stats, emot_stats, df):
    report = {
        "n_conversations":  len(df),
        "total_messages":   int(df["msg_count"].sum()),
        "total_tokens":     int(df["total_tokens"].sum()),
        "functional": {
            "overall": {
                "mean_msgs_per_conv":   round(df["msg_count"].mean(), 1),
                "median_msgs_per_conv": float(df["msg_count"].median()),
            },
            "by_category": func_stats.to_dict(orient="records"),
        },
        "emotional": {
            "overall": {
                "mean_msgs_per_conv":   round(df["msg_count"].mean(), 1),
                "median_msgs_per_conv": float(df["msg_count"].median()),
            },
            "by_category": emot_stats.to_dict(orient="records"),
        },
        "figure": OUT_FIGURE,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_REPORT}")


# -- Main --------------------------------------------------------------------
def main():
    print("\n== Module 3.2d: Length × Classification Analysis ==\n")

    df = load_data()

    print("\n-- Functional category stats --")
    func_stats = compute_weighted_stats(df, "function_primary", FUNC_COLORS)
    print_stats_table(func_stats, "Functional Categories")

    print("\n-- Emotional state stats --")
    emot_stats = compute_weighted_stats(df, "emotion_primary", EMOTION_COLORS)
    print_stats_table(emot_stats, "Emotional States")

    make_figure(func_stats, emot_stats)

    print("\n-- Saving report --")
    save_report(func_stats, emot_stats, df)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
