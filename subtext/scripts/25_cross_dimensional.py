"""
Module 25: Cross-Dimensional Analysis
Joins all enrichment layers (emotional states, functional classifications,
entity attention, frame adoption, conversation shapes, vocab transfer,
opening types, topics) at the conversation level and runs statistical tests
to discover relationships between dimensions.

Zero API cost — local computation only.

Usage:
    python scripts/25_cross_dimensional.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_PARQUET = PROCESSED / "cross_dimensional_merged.parquet"
OUT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "cross_dimensional_report.json"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "cross_dimensional"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "outputs" / "reports").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = [
    "#2E75B6", "#C55A11", "#E91E90", "#E15759",
    "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2",
    "#FF9DA7", "#9C755F", "#BAB0AC", "#4E79A7",
]
DPI = 150
sns.set_theme(style="whitegrid", font_scale=0.9)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def clean(v):
    """Round floats to 3dp; convert numpy scalars to Python natives."""
    if isinstance(v, (float, np.floating)):
        val = float(v)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def clean_dict(d):
    """Recursively clean a dict/list for JSON serialisation."""
    if isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict(v) for v in d]
    return clean(d)


def figpath(name):
    return str(FIG_DIR / name)


def cramers_v(contingency_table):
    """Compute Cramer's V from a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def standardized_residuals(observed):
    """Compute standardized Pearson residuals from a contingency table."""
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    residuals = (observed - expected_df) / np.sqrt(expected_df)
    return residuals


# ---------------------------------------------------------------------------
# Enrichment Registry
# ---------------------------------------------------------------------------
ENRICHMENT_REGISTRY = {
    "emotional_states": {
        "path": PROCESSED / "emotional_states.parquet",
        "level": "conversation",
        "columns": ["emotion_primary", "emotion_secondary", "emotion_confidence"],
    },
    "functional_classifications": {
        "path": PROCESSED / "functional_classifications.parquet",
        "level": "conversation",
        "columns": ["function_primary", "function_secondary", "function_confidence"],
    },
    "entity_attention": {
        "path": PROCESSED / "entity_attention.parquet",
        "level": "conversation_multi",
        "columns": ["entity", "mention_count"],
    },
    "shape_clusters": {
        "path": PROCESSED / "shape_clusters.parquet",
        "level": "conversation_index",  # conversation_id is the index
        "columns": ["shape_cluster_id", "shape_cluster_name"],
    },
    "opening_classifications": {
        "path": PROCESSED / "opening_classifications.parquet",
        "level": "conversation_index",  # conversation_id is the index
        "columns": ["opening_level_1", "opening_level_2"],
    },
    "topic_assignments": {
        "path": PROCESSED / "topic_assignments.parquet",
        "level": "conversation",
        "columns": ["topic_id", "topic_name", "topic_probability"],
    },
    "frame_adoption": {
        "path": PROCESSED / "frame_adoption.parquet",
        "level": "message",
        "columns": ["frame_adoption", "frame_confidence"],
    },
    "vocab_transfer": {
        "path": PROCESSED / "vocab_transfer.parquet",
        "level": "conversation_multi_adoption",
        "columns": ["token", "adoption_conversation_id"],
    },
    "hypotheses": {
        "path": PROCESSED / "hypotheses.parquet",
        "level": "message",
        "columns": ["type", "confidence_level", "topic"],
    },
}


# ===========================================================================
# MASTER MERGE
# ===========================================================================
def build_master_frame():
    """Load conversations_clean and LEFT JOIN all available enrichments."""
    print("=== Building Master Frame ===")

    # Spine
    convos = pd.read_parquet(PROCESSED / "conversations_clean.parquet")
    master = convos[convos["is_analysable"]].copy()
    print(f"  Spine: {len(master):,} analysable conversations")

    available = {}
    missing = []

    for name, cfg in ENRICHMENT_REGISTRY.items():
        path = cfg["path"]
        if not path.exists():
            print(f"  [{name}] MISSING — skipping")
            missing.append(name)
            continue

        try:
            df = pd.read_parquet(path)
            level = cfg["level"]

            if level == "conversation":
                # Direct join on conversation_id column
                cols = [c for c in cfg["columns"] if c in df.columns]
                merge_df = df[["conversation_id"] + cols].drop_duplicates("conversation_id")
                master = master.merge(merge_df, on="conversation_id", how="left")

            elif level == "conversation_index":
                # conversation_id is the index
                df = df.reset_index()
                cols = [c for c in cfg["columns"] if c in df.columns]
                merge_df = df[["conversation_id"] + cols].drop_duplicates("conversation_id")
                master = master.merge(merge_df, on="conversation_id", how="left")

            elif level == "conversation_multi" and name == "entity_attention":
                # Multiple rows per conversation — aggregate to top entity + total mentions
                agg = df.groupby("conversation_id").agg(
                    entity_mention_total=("mention_count", "sum"),
                    top_entity=("entity", lambda x: x.iloc[df.loc[x.index, "mention_count"].argmax()] if len(x) > 0 else None),
                    entity_count=("entity", "nunique"),
                ).reset_index()
                master = master.merge(agg, on="conversation_id", how="left")
                master["entity_mention_total"] = master["entity_mention_total"].fillna(0).astype(int)
                master["entity_count"] = master["entity_count"].fillna(0).astype(int)

            elif level == "conversation_multi_adoption" and name == "vocab_transfer":
                # Count adoptions per conversation
                if "adoption_conversation_id" in df.columns:
                    agg = df.groupby("adoption_conversation_id").size().reset_index(name="vocab_adoptions")
                    agg = agg.rename(columns={"adoption_conversation_id": "conversation_id"})
                    master = master.merge(agg, on="conversation_id", how="left")
                    master["vocab_adoptions"] = master["vocab_adoptions"].fillna(0).astype(int)

            elif level == "message" and name == "frame_adoption":
                # Aggregate to dominant frame per conversation
                if "frame_adoption" in df.columns:
                    dom = df.groupby("conversation_id")["frame_adoption"].agg(
                        lambda x: x.value_counts().index[0] if len(x) > 0 else None
                    ).reset_index()
                    dom.columns = ["conversation_id", "dominant_frame"]
                    div = df.groupby("conversation_id")["frame_adoption"].nunique().reset_index()
                    div.columns = ["conversation_id", "frame_diversity"]
                    merge_df = dom.merge(div, on="conversation_id", how="left")
                    master = master.merge(merge_df, on="conversation_id", how="left")

            elif level == "message" and name == "hypotheses":
                # Aggregate to belief count + dominant type per conversation
                if "type" in df.columns:
                    # Count beliefs per conversation
                    count_df = df.groupby("conversation_id").size().reset_index(name="belief_count")
                    # Dominant belief type
                    dom_type = df.groupby("conversation_id")["type"].agg(
                        lambda x: x.value_counts().index[0] if len(x) > 0 else None
                    ).reset_index()
                    dom_type.columns = ["conversation_id", "dominant_belief_type"]
                    # Average confidence (convert to numeric if needed)
                    conf_df = None
                    if "confidence_level" in df.columns:
                        df_conf = df.copy()
                        df_conf["confidence_numeric"] = pd.to_numeric(df_conf["confidence_level"], errors="coerce")
                        if df_conf["confidence_numeric"].notna().any():
                            conf_df = df_conf.groupby("conversation_id")["confidence_numeric"].mean().reset_index()
                            conf_df.columns = ["conversation_id", "avg_belief_confidence"]
                    agg = count_df.merge(dom_type, on="conversation_id", how="left")
                    if conf_df is not None:
                        agg = agg.merge(conf_df, on="conversation_id", how="left")
                    master = master.merge(agg, on="conversation_id", how="left")
                    master["belief_count"] = master["belief_count"].fillna(0).astype(int)

            available[name] = len(df)
            print(f"  [{name}] Joined ({len(df):,} source rows)")

        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            missing.append(name)

    print(f"\n  Master frame: {len(master):,} rows x {len(master.columns)} columns")
    print(f"  Available: {list(available.keys())}")
    print(f"  Missing:   {missing}")

    return master, available, missing


# ===========================================================================
# ANALYSIS 1: Emotion x Function
# ===========================================================================
def analysis_emotion_x_function(master):
    """Chi-squared test: emotion_primary x function_primary."""
    print("\n--- Analysis 1: Emotion x Function ---")
    required = ["emotion_primary", "function_primary"]
    if not all(c in master.columns for c in required):
        print("  SKIPPED — missing columns")
        return None

    df = master.dropna(subset=required)
    ct = pd.crosstab(df["emotion_primary"], df["function_primary"])

    # Filter to categories with 10+ observations
    ct = ct.loc[ct.sum(axis=1) >= 10, ct.sum(axis=0) >= 10]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print("  SKIPPED — insufficient categories after filtering")
        return None

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    cv = cramers_v(ct)
    resid = standardized_residuals(ct)

    # Top associations (highest absolute residuals)
    flat = resid.stack().reset_index()
    flat.columns = ["emotion", "function", "residual"]
    flat["abs_residual"] = flat["residual"].abs()
    top_assoc = flat.nlargest(10, "abs_residual").to_dict("records")

    print(f"  Chi2={chi2:.1f}, p={p:.2e}, Cramer's V={cv:.3f}, df={dof}")

    # Figure 1: Observed counts heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Emotion x Function: Observed Counts", fontsize=14)
    ax.set_xlabel("Function", fontsize=11)
    ax.set_ylabel("Emotion", fontsize=11)
    plt.tight_layout()
    plt.savefig(figpath("01_emotion_function_heatmap.png"), dpi=DPI)
    plt.close()

    # Figure 2: Standardized residuals heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = max(abs(resid.values.min()), abs(resid.values.max()), 3)
    sns.heatmap(resid, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, ax=ax, linewidths=0.5)
    ax.set_title(f"Emotion x Function: Standardized Residuals (Cramer's V = {cv:.3f})", fontsize=14)
    ax.set_xlabel("Function", fontsize=11)
    ax.set_ylabel("Emotion", fontsize=11)
    plt.tight_layout()
    plt.savefig(figpath("02_emotion_function_residuals.png"), dpi=DPI)
    plt.close()

    print("  Saved figures 01, 02")
    return clean_dict({
        "chi2": chi2, "p_value": p, "cramers_v": cv, "dof": dof,
        "n_observations": int(ct.sum().sum()),
        "emotions": list(ct.index),
        "functions": list(ct.columns),
        "top_associations": top_assoc,
    })


# ===========================================================================
# ANALYSIS 2: Frame Adoption x Beliefs
# ===========================================================================
def analysis_frame_x_beliefs(master):
    """ANOVA: belief_count ~ dominant_frame. Plus frame x belief_type contingency."""
    print("\n--- Analysis 2: Frame Adoption x Beliefs ---")
    required = ["dominant_frame", "belief_count"]
    if not all(c in master.columns for c in required):
        print("  SKIPPED — missing columns")
        return None

    df = master.dropna(subset=["dominant_frame"]).copy()
    df["belief_count"] = df["belief_count"].fillna(0)

    # Filter to frames with 20+ conversations
    frame_counts = df["dominant_frame"].value_counts()
    top_frames = frame_counts[frame_counts >= 20].index.tolist()
    if len(top_frames) < 2:
        print("  SKIPPED — fewer than 2 frames with 20+ conversations")
        return None

    df_filtered = df[df["dominant_frame"].isin(top_frames)]
    groups = [g["belief_count"].values for _, g in df_filtered.groupby("dominant_frame")]
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size: eta-squared
    ss_between = sum(len(g) * (g.mean() - df_filtered["belief_count"].mean())**2 for g in groups)
    ss_total = ((df_filtered["belief_count"] - df_filtered["belief_count"].mean())**2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    print(f"  ANOVA F={f_stat:.2f}, p={p_value:.2e}, eta²={eta_sq:.3f}")

    # Figure 3: Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df_filtered.groupby("dominant_frame")["belief_count"].median().sort_values(ascending=False).index
    sns.boxplot(data=df_filtered, x="dominant_frame", y="belief_count", order=order,
                palette=PALETTE[:len(order)], ax=ax)
    ax.set_title(f"Belief Count by Dominant Frame (F={f_stat:.1f}, p={p_value:.2e})", fontsize=14)
    ax.set_xlabel("Dominant Frame", fontsize=11)
    ax.set_ylabel("Beliefs per Conversation", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figpath("03_frame_belief_count_box.png"), dpi=DPI)
    plt.close()

    # Figure 4: Frame x belief type contingency (if available)
    result = {
        "anova_f": f_stat, "anova_p": p_value, "eta_squared": eta_sq,
        "frames_tested": top_frames,
        "frame_means": {f: float(df_filtered[df_filtered["dominant_frame"]==f]["belief_count"].mean()) for f in top_frames},
    }

    if "dominant_belief_type" in master.columns:
        df2 = master.dropna(subset=["dominant_frame", "dominant_belief_type"])
        df2 = df2[df2["dominant_frame"].isin(top_frames)]
        ct = pd.crosstab(df2["dominant_frame"], df2["dominant_belief_type"])
        ct = ct.loc[:, ct.sum() >= 5]
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            cv = cramers_v(ct)
            fig, ax = plt.subplots(figsize=(12, 8))
            # Normalize rows for comparison
            ct_norm = ct.div(ct.sum(axis=1), axis=0)
            sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, linewidths=0.5)
            ax.set_title(f"Frame x Belief Type (normalized, Cramer's V = {cv:.3f})", fontsize=14)
            ax.set_xlabel("Belief Type", fontsize=11)
            ax.set_ylabel("Dominant Frame", fontsize=11)
            plt.tight_layout()
            plt.savefig(figpath("04_frame_belief_type_heatmap.png"), dpi=DPI)
            plt.close()
            result["frame_belief_type_cramers_v"] = cv
            print("  Saved figures 03, 04")
        else:
            print("  Saved figure 03 (insufficient belief type data for fig 04)")
    else:
        print("  Saved figure 03 (no belief type data for fig 04)")

    return clean_dict(result)


# ===========================================================================
# ANALYSIS 3: Conversation Shape x Function x Emotion
# ===========================================================================
def analysis_shape_x_function(master):
    """Chi-squared: shape_cluster x function, shape_cluster x emotion."""
    print("\n--- Analysis 3: Shape x Function x Emotion ---")

    shape_col = "shape_cluster_name" if "shape_cluster_name" in master.columns else "shape_cluster_id"
    if shape_col not in master.columns:
        print("  SKIPPED — no shape cluster column")
        return None

    result = {}

    for target, fig_num, fig_name in [
        ("function_primary", "05", "shape_function_mosaic"),
        ("emotion_primary", "06", "shape_emotion_mosaic"),
    ]:
        if target not in master.columns:
            print(f"  SKIPPED {target} — missing column")
            continue

        df = master.dropna(subset=[shape_col, target])
        ct = pd.crosstab(df[shape_col], df[target])
        ct = ct.loc[ct.sum(axis=1) >= 10, ct.sum(axis=0) >= 10]

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cv = cramers_v(ct)

        label = target.replace("_primary", "")
        result[f"shape_x_{label}"] = {
            "chi2": chi2, "p_value": p, "cramers_v": cv,
            "shapes": list(ct.index), f"{label}s": list(ct.columns),
        }
        print(f"  Shape x {label}: Chi2={chi2:.1f}, p={p:.2e}, V={cv:.3f}")

        # Stacked bar chart (normalized)
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(14, 7))
        ct_norm.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:ct_norm.shape[1]], edgecolor="white")
        ax.set_title(f"Shape Cluster x {label.title()} (Cramer's V = {cv:.3f})", fontsize=14)
        ax.set_xlabel("Shape Cluster", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=11)
        ax.legend(title=label.title(), bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figpath(f"{fig_num}_{fig_name}.png"), dpi=DPI)
        plt.close()

    if result:
        print(f"  Saved figures 05, 06")
    return clean_dict(result) if result else None


# ===========================================================================
# ANALYSIS 4: Usage Intensity x Emotion Over Time
# ===========================================================================
def analysis_volume_x_emotion(master):
    """Correlation between weekly conversation volume and emotional state distribution."""
    print("\n--- Analysis 4: Volume x Emotion Over Time ---")
    required = ["year_week", "emotion_primary"]
    if not all(c in master.columns for c in required):
        print("  SKIPPED — missing columns")
        return None

    df = master.dropna(subset=required)

    # Weekly aggregation
    weekly_vol = df.groupby("year_week").size().reset_index(name="conversation_count")

    # Emotion fractions per week
    emotion_counts = df.groupby(["year_week", "emotion_primary"]).size().unstack(fill_value=0)
    emotion_frac = emotion_counts.div(emotion_counts.sum(axis=1), axis=0)

    # Top 5 emotions
    top_emotions = df["emotion_primary"].value_counts().head(5).index.tolist()

    # Merge for correlations
    merged = weekly_vol.set_index("year_week").join(emotion_frac[top_emotions])
    merged = merged.dropna()

    if len(merged) < 10:
        print("  SKIPPED — fewer than 10 weeks of data")
        return None

    correlations = {}
    for em in top_emotions:
        if em in merged.columns:
            r, p = stats.pearsonr(merged["conversation_count"], merged[em])
            correlations[em] = {"pearson_r": r, "p_value": p}

    print(f"  Top emotion correlations with volume:")
    for em, vals in correlations.items():
        print(f"    {em}: r={vals['pearson_r']:.3f}, p={vals['p_value']:.3f}")

    # Figure 7: Dual-axis time series
    fig, ax1 = plt.subplots(figsize=(14, 6))
    weeks = merged.index.tolist()
    x = range(len(weeks))

    ax1.bar(x, merged["conversation_count"], color="#CCCCCC", alpha=0.6, label="Volume")
    ax1.set_ylabel("Conversations per Week", fontsize=11)
    ax1.set_xlabel("Week", fontsize=11)

    ax2 = ax1.twinx()
    for i, em in enumerate(top_emotions[:3]):
        if em in merged.columns:
            ax2.plot(x, merged[em], color=PALETTE[i], linewidth=2, label=em, marker="o", markersize=3)
    ax2.set_ylabel("Emotion Fraction", fontsize=11)

    # Month labels on x-axis
    from datetime import datetime as _dt
    tick_positions, tick_labels = [], []
    seen_months = set()
    for idx, wk in enumerate(weeks):
        try:
            dt = _dt.strptime(wk + "-1", "%Y-W%W-%w")
            mk = dt.strftime("%Y-%m")
            if mk not in seen_months:
                seen_months.add(mk)
                tick_positions.append(idx)
                tick_labels.append(dt.strftime("%b %Y"))
        except ValueError:
            pass
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Weekly Volume x Top Emotions", fontsize=14)
    plt.tight_layout()
    plt.savefig(figpath("07_volume_emotion_timeseries.png"), dpi=DPI)
    plt.close()

    # Figure 8: Scatter matrix
    plot_cols = ["conversation_count"] + [e for e in top_emotions[:3] if e in merged.columns]
    if len(plot_cols) >= 3:
        fig, axes = plt.subplots(len(plot_cols)-1, 1, figsize=(10, 4*(len(plot_cols)-1)))
        if len(plot_cols) - 1 == 1:
            axes = [axes]
        for i, em in enumerate(plot_cols[1:]):
            ax = axes[i]
            ax.scatter(merged["conversation_count"], merged[em], alpha=0.6, color=PALETTE[i], s=30)
            r = correlations.get(em, {}).get("pearson_r", 0)
            ax.set_xlabel("Conversations per Week", fontsize=10)
            ax.set_ylabel(f"{em} fraction", fontsize=10)
            ax.set_title(f"Volume vs {em} (r={r:.3f})", fontsize=11)
        plt.tight_layout()
        plt.savefig(figpath("08_volume_emotion_correlation.png"), dpi=DPI)
        plt.close()

    print("  Saved figures 07, 08")
    return clean_dict({"correlations": correlations, "weeks_analyzed": len(merged)})


# ===========================================================================
# ANALYSIS 5: Entity Attention x Emotion x Function
# ===========================================================================
def analysis_entity_profiles(master):
    """Compare emotional/functional profiles of top entity conversations vs baseline."""
    print("\n--- Analysis 5: Entity Profiles ---")
    if "top_entity" not in master.columns:
        print("  SKIPPED — no entity data")
        return None
    required = ["emotion_primary", "function_primary"]
    if not all(c in master.columns for c in required):
        print("  SKIPPED — missing emotion/function columns")
        return None

    # Get top 6 entities by conversation count
    entity_counts = master["top_entity"].value_counts()
    top_entities = entity_counts[entity_counts >= 20].head(6).index.tolist()
    if len(top_entities) < 2:
        print("  SKIPPED — fewer than 2 entities with 20+ conversations")
        return None

    # Baseline distributions
    baseline_emo = master["emotion_primary"].value_counts(normalize=True)
    baseline_func = master["function_primary"].value_counts(normalize=True)

    profiles = {}
    for entity in top_entities:
        ent_df = master[master["top_entity"] == entity]
        emo_dist = ent_df["emotion_primary"].value_counts(normalize=True)
        func_dist = ent_df["function_primary"].value_counts(normalize=True)

        # Divergence from baseline (KL-divergence-like, using top categories)
        top_emos = baseline_emo.head(8).index
        emo_diff = sum(abs(emo_dist.get(e, 0) - baseline_emo.get(e, 0)) for e in top_emos) / 2

        top_funcs = baseline_func.head(8).index
        func_diff = sum(abs(func_dist.get(f, 0) - baseline_func.get(f, 0)) for f in top_funcs) / 2

        profiles[entity] = {
            "conversations": int(len(ent_df)),
            "top_emotion": emo_dist.index[0] if len(emo_dist) > 0 else None,
            "top_function": func_dist.index[0] if len(func_dist) > 0 else None,
            "emotion_divergence": emo_diff,
            "function_divergence": func_diff,
            "emotion_distribution": emo_dist.head(5).to_dict(),
            "function_distribution": func_dist.head(5).to_dict(),
        }
        print(f"  {entity}: {len(ent_df)} convos, top_emo={profiles[entity]['top_emotion']}, "
              f"top_func={profiles[entity]['top_function']}")

    # Figure 9: Entity x Emotion profiles (grouped bar)
    top_emos = master["emotion_primary"].value_counts().head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.12
    x = np.arange(len(top_emos))
    for i, entity in enumerate(top_entities):
        ent_df = master[master["top_entity"] == entity]
        emo_dist = ent_df["emotion_primary"].value_counts(normalize=True)
        vals = [emo_dist.get(e, 0) for e in top_emos]
        ax.bar(x + i * bar_width, vals, bar_width, label=entity, color=PALETTE[i % len(PALETTE)])

    # Add baseline
    baseline_vals = [baseline_emo.get(e, 0) for e in top_emos]
    ax.bar(x + len(top_entities) * bar_width, baseline_vals, bar_width,
           label="Baseline", color="#999999", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + bar_width * len(top_entities) / 2)
    ax.set_xticklabels(top_emos, rotation=45, ha="right")
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("Emotional Profile by Top Entity", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath("09_entity_emotion_profiles.png"), dpi=DPI)
    plt.close()

    # Figure 10: Entity x Function profiles (grouped bar)
    top_funcs = master["function_primary"].value_counts().head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top_funcs))
    for i, entity in enumerate(top_entities):
        ent_df = master[master["top_entity"] == entity]
        func_dist = ent_df["function_primary"].value_counts(normalize=True)
        vals = [func_dist.get(f, 0) for f in top_funcs]
        ax.bar(x + i * bar_width, vals, bar_width, label=entity, color=PALETTE[i % len(PALETTE)])

    baseline_vals = [baseline_func.get(f, 0) for f in top_funcs]
    ax.bar(x + len(top_entities) * bar_width, baseline_vals, bar_width,
           label="Baseline", color="#999999", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + bar_width * len(top_entities) / 2)
    ax.set_xticklabels(top_funcs, rotation=45, ha="right")
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("Functional Profile by Top Entity", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath("10_entity_function_profiles.png"), dpi=DPI)
    plt.close()

    # Figure 11: Radar chart comparing top 2 entities
    top2 = top_entities[:2]
    if len(top2) == 2:
        categories = top_emos[:5] + top_funcs[:5]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        radar_colors = [PALETTE[0], "#E91E90"]
        for i, entity in enumerate(top2):
            ent_df = master[master["top_entity"] == entity]
            emo_dist = ent_df["emotion_primary"].value_counts(normalize=True)
            func_dist = ent_df["function_primary"].value_counts(normalize=True)
            values = [emo_dist.get(c, 0) for c in top_emos[:5]] + [func_dist.get(c, 0) for c in top_funcs[:5]]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=entity, color=radar_colors[i])
            ax.fill(angles, values, alpha=0.15, color=radar_colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_title(f"{top2[0]} vs {top2[1]}: Emotion + Function Profile", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(figpath("11_top2_entity_radar.png"), dpi=DPI)
        plt.close()
        print("  Saved figures 09, 10, 11")
    else:
        print("  Saved figures 09, 10 (radar skipped — need 2+ entities)")

    return clean_dict(profiles)


# ===========================================================================
# ANALYSIS 6: Opening Type x Function x Emotion
# ===========================================================================
def analysis_opening_predictive(master):
    """Chi-squared: opening_level_1 x function, opening x emotion."""
    print("\n--- Analysis 6: Opening Type Predictive Power ---")

    opening_col = "opening_level_1" if "opening_level_1" in master.columns else None
    if opening_col is None:
        print("  SKIPPED — no opening classification column")
        return None

    result = {}

    for target, fig_num, fig_name in [
        ("function_primary", "12", "opening_function_heatmap"),
        ("emotion_primary", "13", "opening_emotion_heatmap"),
    ]:
        if target not in master.columns:
            continue

        df = master.dropna(subset=[opening_col, target])
        ct = pd.crosstab(df[opening_col], df[target])
        ct = ct.loc[ct.sum(axis=1) >= 10, ct.sum(axis=0) >= 10]

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p, dof, _ = stats.chi2_contingency(ct)
        cv = cramers_v(ct)

        label = target.replace("_primary", "")
        result[f"opening_x_{label}"] = {
            "chi2": chi2, "p_value": p, "cramers_v": cv,
        }
        print(f"  Opening x {label}: Chi2={chi2:.1f}, p={p:.2e}, V={cv:.3f}")

        # Normalized heatmap
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, linewidths=0.5)
        ax.set_title(f"Opening Type x {label.title()} (Cramer's V = {cv:.3f})", fontsize=14)
        ax.set_xlabel(label.title(), fontsize=11)
        ax.set_ylabel("Opening Type", fontsize=11)
        plt.tight_layout()
        plt.savefig(figpath(f"{fig_num}_{fig_name}.png"), dpi=DPI)
        plt.close()

    if result:
        print(f"  Saved figures 12, 13")
    return clean_dict(result) if result else None


# ===========================================================================
# ANALYSIS 7: Vocab Transfer Context
# ===========================================================================
def analysis_vocab_transfer_context(master):
    """Compare emotional/functional context of vocab adoption conversations vs others."""
    print("\n--- Analysis 7: Vocab Transfer Context ---")

    if "vocab_adoptions" not in master.columns:
        print("  SKIPPED — no vocab transfer data")
        return None
    if "function_primary" not in master.columns or "emotion_primary" not in master.columns:
        print("  SKIPPED — missing function/emotion columns")
        return None

    master = master.copy()
    master["has_adoption"] = master["vocab_adoptions"] > 0

    n_adopt = master["has_adoption"].sum()
    n_no_adopt = (~master["has_adoption"]).sum()
    print(f"  Conversations with vocab adoption: {n_adopt}, without: {n_no_adopt}")

    if n_adopt < 10:
        print("  SKIPPED — fewer than 10 adoption conversations")
        return None

    result = {}

    # Function distribution comparison
    for dim, fig_num, fig_name in [
        ("function_primary", "14", "vocab_adoption_by_function"),
        ("emotion_primary", "15", "vocab_adoption_by_emotion"),
    ]:
        df = master.dropna(subset=[dim])
        adopt_dist = df[df["has_adoption"]][dim].value_counts(normalize=True)
        no_adopt_dist = df[~df["has_adoption"]][dim].value_counts(normalize=True)

        # Get top categories
        top_cats = df[dim].value_counts().head(8).index.tolist()

        label = dim.replace("_primary", "")
        result[f"adoption_by_{label}"] = {
            "adoption_distribution": {c: adopt_dist.get(c, 0) for c in top_cats},
            "non_adoption_distribution": {c: no_adopt_dist.get(c, 0) for c in top_cats},
        }

        # Bar chart comparing adoption vs non-adoption
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_cats))
        width = 0.35
        ax.bar(x - width/2, [adopt_dist.get(c, 0) for c in top_cats], width,
               label=f"With Adoption (n={n_adopt})", color=PALETTE[0])
        ax.bar(x + width/2, [no_adopt_dist.get(c, 0) for c in top_cats], width,
               label=f"Without (n={n_no_adopt})", color=PALETTE[1], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(top_cats, rotation=45, ha="right")
        ax.set_ylabel("Proportion", fontsize=11)
        ax.set_title(f"Vocab Adoption by {label.title()}", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(figpath(f"{fig_num}_{fig_name}.png"), dpi=DPI)
        plt.close()

    print("  Saved figures 14, 15")
    return clean_dict(result)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("  Module 25: Cross-Dimensional Analysis")
    print("=" * 70)
    start_time = datetime.now(timezone.utc)

    # Step 1: Build master frame
    master, available, missing = build_master_frame()

    # Save merged parquet
    master.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  Saved {OUT_PARQUET} ({len(master):,} rows x {len(master.columns)} cols)")

    # Step 2: Run analyses
    report = {
        "module": "cross_dimensional",
        "generated_at": start_time.isoformat(),
        "input_data": {
            "conversations": int(len(master)),
            "enrichments_available": list(available.keys()),
            "enrichments_missing": missing,
            "enrichment_row_counts": available,
        },
        "analyses": {},
        "key_findings": [],
    }

    analyses = [
        ("emotion_x_function", analysis_emotion_x_function),
        ("frame_x_beliefs", analysis_frame_x_beliefs),
        ("shape_x_function", analysis_shape_x_function),
        ("volume_x_emotion", analysis_volume_x_emotion),
        ("entity_profiles", analysis_entity_profiles),
        ("opening_predictive_power", analysis_opening_predictive),
        ("vocab_transfer_context", analysis_vocab_transfer_context),
    ]

    for name, func in analyses:
        try:
            result = func(master)
            if result is not None:
                report["analyses"][name] = result
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            report["analyses"][name] = {"error": str(e)}

    # Step 3: Generate key findings
    findings = []
    a = report["analyses"]

    if "emotion_x_function" in a and "cramers_v" in a["emotion_x_function"]:
        cv = a["emotion_x_function"]["cramers_v"]
        findings.append(f"Emotion x Function: Cramer's V = {cv:.3f} — "
                       f"{'strong' if cv > 0.3 else 'moderate' if cv > 0.15 else 'weak'} association")
        # Top association
        if "top_associations" in a["emotion_x_function"]:
            top = a["emotion_x_function"]["top_associations"][0]
            findings.append(f"  Strongest link: {top.get('emotion','')} x {top.get('function','')} "
                          f"(residual={top.get('residual', 0):.1f})")

    if "frame_x_beliefs" in a and "eta_squared" in a["frame_x_beliefs"]:
        eta = a["frame_x_beliefs"]["eta_squared"]
        findings.append(f"Frame → Belief count: eta² = {eta:.3f} — "
                       f"{'large' if eta > 0.14 else 'medium' if eta > 0.06 else 'small'} effect")

    if "entity_profiles" in a:
        for entity in list(a["entity_profiles"].keys())[:2]:
            p = a["entity_profiles"][entity]
            findings.append(f"{entity}: top_emotion={p.get('top_emotion')}, "
                          f"top_function={p.get('top_function')}, "
                          f"emo_divergence={p.get('emotion_divergence', 0):.3f}")

    if "opening_predictive_power" in a:
        for key in ["opening_x_function", "opening_x_emotion"]:
            if key in a["opening_predictive_power"]:
                cv = a["opening_predictive_power"][key].get("cramers_v", 0)
                findings.append(f"Opening → {key.split('_')[-1]}: V={cv:.3f}")

    report["key_findings"] = findings

    # Save report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2, default=str)
    print(f"\n  Saved {OUT_REPORT}")

    # Step 4: Validation checklist
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKLIST")
    print("=" * 70)

    checks = []

    # 1. Row count
    ok = len(master) > 1000
    checks.append(("Master frame row count", ok, f"{len(master):,} rows"))

    # 2. Enrichments available
    ok = len(available) >= 5
    checks.append(("At least 5 enrichments", ok, f"{len(available)} available"))

    # 3. Statistical tests valid
    valid_p = True
    for aname, adata in report["analyses"].items():
        if isinstance(adata, dict):
            for k, v in adata.items():
                if "p_value" in str(k) and isinstance(v, (int, float)):
                    if not (0 <= v <= 1):
                        valid_p = False
    checks.append(("Valid p-values", valid_p, "all in [0, 1]"))

    # 4. No NaN/Inf in report
    report_str = json.dumps(report, default=str)
    no_nan = "NaN" not in report_str and "Infinity" not in report_str
    checks.append(("No NaN/Inf in report", no_nan, "clean"))

    # 5. Figures exist
    figs = list(FIG_DIR.glob("*.png"))
    fig_ok = len(figs) >= 10
    checks.append(("Figure PNGs exist", fig_ok, f"{len(figs)} figures"))

    # 6. Cramer's V in range
    v_ok = True
    for aname, adata in report["analyses"].items():
        if isinstance(adata, dict):
            for k, v in adata.items():
                if "cramers_v" in str(k) and isinstance(v, (int, float)):
                    if not (0 <= v <= 1):
                        v_ok = False
    checks.append(("Cramer's V in [0, 1]", v_ok, "valid"))

    # 7. Entity profiles
    has_entities = "entity_profiles" in report["analyses"]
    if has_entities:
        ep = report["analyses"]["entity_profiles"]
        has_top2 = len(ep) >= 2
    else:
        has_top2 = False
    checks.append(("Entity profiles (top 2 entities)", has_top2, "present" if has_top2 else "missing"))

    # 8. Parquet saved
    parquet_ok = OUT_PARQUET.exists()
    checks.append(("Merged parquet saved", parquet_ok, str(OUT_PARQUET.name)))

    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {detail}")

    passed_count = sum(1 for _, p, _ in checks if p)
    print(f"\n  {passed_count}/{len(checks)} checks passed")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Module 25 Complete")
    print(f"  Enrichments joined: {len(available)}")
    print(f"  Analyses run: {len(report['analyses'])}")
    print(f"  Figures generated: {len(figs)}")
    print(f"  Key findings: {len(findings)}")
    for f in findings:
        print(f"    - {f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
