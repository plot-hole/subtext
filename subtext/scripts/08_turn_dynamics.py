"""
Module 3.1: Structural Analysis — Turn Dynamics
Reads clean Parquet files, produces analysis outputs: JSON report + 12 figures.
"""

import os
import sys
import json
import warnings
import traceback
from datetime import datetime, timezone

# Force stdout to UTF-8 so non-ASCII print statements don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# --- Paths --------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
FIGS_DIR  = os.path.join(BASE, "outputs", "figures", "turn_dynamics")
REPORT_PATH = os.path.join(BASE, "outputs", "reports", "turn_dynamics_report.json")
CONFIG_PATH = os.path.join(BASE, "config", "quality_config.json")

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# --- Style constants ----------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
COLOR_USER      = "#2E75B6"
COLOR_ASSISTANT = "#A9D18E"
COLOR_BALANCED  = "#BDD7EE"

FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE     = (14, 6)
FIGSIZE_TALL     = (10, 8)
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE  = 10
DPI = 150

# --- Helpers ------------------------------------------------------------------
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
    return os.path.join(FIGS_DIR, name)

# Day-of-week labels (pandas: 0=Mon)
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
TOD_ORDER  = ["morning", "afternoon", "evening", "night"]

report = {
    "module": "turn_dynamics",
    "module_version": "1.0",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "warnings": [],
}
figures_generated = []

# ==============================================================================
# STEP 0 — Load & filter
# ==============================================================================
print("\n-- Step 0: Load & filter ----------------------------------------------")

conversations = pd.read_parquet(CONV_PATH)
messages      = pd.read_parquet(MSGS_PATH)

conv = conversations[conversations["is_analysable"]].copy()
msgs = messages[messages["conversation_id"].isin(conv["conversation_id"])].copy()

print(f"  Analysable conversations : {len(conv):,}")
print(f"  Messages in scope        : {len(msgs):,}")
print(f"  Excluded conversations   : {len(conversations) - len(conv):,}")

report["input_data"] = {
    "conversations_analysed": len(conv),
    "messages_analysed": len(msgs),
    "date_range": [
        str(conv["created_at"].min().date()),
        str(conv["created_at"].max().date()),
    ],
    "excluded_conversations": len(conversations) - len(conv),
}

# Load config for era boundaries
try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    era_boundaries = config.get("model_era_boundaries", {})
except Exception as e:
    era_boundaries = {}
    report["warnings"].append(f"Could not load config: {e}")

# ==============================================================================
# STEP 1 — Token ratio analysis
# ==============================================================================
print("\n-- Step 1: Token ratio analysis ---------------------------------------")

token_ratio_report = {}
try:
    ratio = conv["user_token_ratio"].dropna()

    # 1a. Distribution stats
    ratio_stats = {
        "count": int(len(ratio)),
        "mean":   clean(ratio.mean()),
        "median": clean(ratio.median()),
        "std":    clean(ratio.std()),
        "skew":   clean(ratio.skew()),
        "kurtosis": clean(ratio.kurt()),
        "p5":  clean(ratio.quantile(0.05)),
        "p25": clean(ratio.quantile(0.25)),
        "p75": clean(ratio.quantile(0.75)),
        "p95": clean(ratio.quantile(0.95)),
        "pct_user_dominated":      clean((ratio > 0.5).mean() * 100),
        "pct_assistant_dominated": clean((ratio < 0.5).mean() * 100),
        "pct_balanced":            clean(((ratio >= 0.4) & (ratio <= 0.6)).mean() * 100),
    }
    token_ratio_report["overall_stats"] = ratio_stats
    print(f"  Overall — mean={ratio_stats['mean']:.3f}  median={ratio_stats['median']:.3f}  "
          f"user_dom={ratio_stats['pct_user_dominated']:.1f}%  asst_dom={ratio_stats['pct_assistant_dominated']:.1f}%")

    # 1b. By conversation type
    ratio_by_type = (
        conv.groupby("conversation_type")["user_token_ratio"]
        .agg(["count", "mean", "median", "std"])
        .round(3)
        .sort_values("median")
    )
    token_ratio_report["by_conversation_type"] = clean_dict(ratio_by_type.to_dict(orient="index"))

    # 1c. Monthly trend
    monthly = conv.groupby("year_month")["user_token_ratio"].agg(["mean", "median", "std", "count"])
    conv_sorted = conv.sort_values("created_at").copy()
    conv_sorted["ratio_rolling_90d"] = (
        conv_sorted["user_token_ratio"]
        .rolling(window=90, min_periods=30, center=True)
        .mean()
    )
    token_ratio_report["monthly_trend"] = clean_dict(monthly.to_dict(orient="index"))

    # 1d. By model era
    ratio_by_era = (
        conv.groupby("model_era")["user_token_ratio"]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )
    token_ratio_report["by_model_era"] = clean_dict(ratio_by_era.to_dict(orient="index"))

    # 1e. By temporal context
    ratio_by_tod = conv.groupby("time_of_day")["user_token_ratio"].agg(["mean", "median", "count"]).round(3)
    ratio_by_weekend = conv.groupby("is_weekend")["user_token_ratio"].agg(["mean", "median", "count"]).round(3)
    token_ratio_report["by_time_of_day"] = clean_dict(ratio_by_tod.to_dict(orient="index"))
    token_ratio_report["by_weekend"] = clean_dict(ratio_by_weekend.to_dict(orient="index"))

    print("  Step 1 complete.")
except Exception as e:
    report["warnings"].append(f"Step 1 error: {traceback.format_exc()}")
    print(f"  Step 1 ERROR: {e}")

# ==============================================================================
# STEP 2 — Turn depth analysis
# ==============================================================================
print("\n-- Step 2: Turn depth analysis ----------------------------------------")

turn_depth_report = {}
try:
    turns = conv["turns"].dropna()

    # 2a. Distribution
    turn_stats = {
        "mean":   clean(turns.mean()),
        "median": clean(turns.median()),
        "std":    clean(turns.std()),
        "max":    int(turns.max()),
        "p75": clean(turns.quantile(0.75)),
        "p90": clean(turns.quantile(0.90)),
        "p95": clean(turns.quantile(0.95)),
        "p99": clean(turns.quantile(0.99)),
        "pct_single_exchange": clean((turns <= 2).mean() * 100),
        "pct_deep":     clean((turns >= 20).mean() * 100),
        "pct_marathon": clean((turns >= 50).mean() * 100),
    }
    turn_depth_report["overall_stats"] = turn_stats
    print(f"  Turns — mean={turn_stats['mean']:.1f}  median={turn_stats['median']:.1f}  max={turn_stats['max']}")

    # 2b. Correlation with token ratio
    pair = conv[["turns", "user_token_ratio"]].dropna()
    correlation = pair.corr().iloc[0, 1]
    turn_depth_report["correlation_with_ratio"] = clean(correlation)

    conv["turn_bin"] = pd.cut(
        conv["turns"],
        bins=[0, 2, 5, 10, 20, 50, float("inf")],
        labels=["1-2", "3-5", "6-10", "11-20", "21-50", "50+"],
    )
    ratio_by_turn_depth = (
        conv.groupby("turn_bin", observed=True)["user_token_ratio"]
        .agg(["mean", "median", "count"])
    )
    turn_depth_report["by_turn_bin"] = clean_dict(ratio_by_turn_depth.to_dict(orient="index"))

    # 2c. Monthly trend
    turns_monthly = conv.groupby("year_month")["turns"].agg(["mean", "median", "count"])
    turn_depth_report["monthly_trend"] = clean_dict(turns_monthly.to_dict(orient="index"))

    # 2d. By type and era
    turns_by_type = conv.groupby("conversation_type")["turns"].agg(["mean", "median", "count"]).round(3)
    turns_by_era  = conv.groupby("model_era")["turns"].agg(["mean", "median", "count"]).round(3)
    turn_depth_report["by_conversation_type"] = clean_dict(turns_by_type.to_dict(orient="index"))
    turn_depth_report["by_model_era"] = clean_dict(turns_by_era.to_dict(orient="index"))

    print("  Step 2 complete.")
except Exception as e:
    report["warnings"].append(f"Step 2 error: {traceback.format_exc()}")
    print(f"  Step 2 ERROR: {e}")

# ==============================================================================
# STEP 3 — Contribution asymmetry
# ==============================================================================
print("\n-- Step 3: Contribution asymmetry -------------------------------------")

asymmetry_report = {}
try:
    user_msgs = msgs[msgs["role"] == "user"].copy()
    asst_msgs = msgs[msgs["role"] == "assistant"].copy()

    user_per_conv = user_msgs.groupby("conversation_id")["token_count"].agg(
        user_msg_mean="mean",
        user_msg_median="median",
        user_msg_std="std",
        user_msg_max="max",
        user_msg_min="min",
    )
    asst_per_conv = asst_msgs.groupby("conversation_id")["token_count"].agg(
        asst_msg_mean="mean",
        asst_msg_median="median",
        asst_msg_std="std",
        asst_msg_max="max",
        asst_msg_min="min",
    )

    asymm_df = user_per_conv.join(asst_per_conv, how="inner")
    denom = asymm_df["user_msg_mean"] + asymm_df["asst_msg_mean"]
    asymm_df["asymmetry_score"] = np.where(
        denom > 0,
        (asymm_df["user_msg_mean"] - asymm_df["asst_msg_mean"]) / denom,
        np.nan,
    )
    asymm_df["user_cv"] = asymm_df["user_msg_std"] / asymm_df["user_msg_mean"].replace(0, np.nan)

    # Merge type info
    asymm_df = asymm_df.join(conv.set_index("conversation_id")[["conversation_type"]])

    asym_scores = asymm_df["asymmetry_score"].dropna()
    asymmetry_report = {
        "mean_asymmetry_score":   clean(asym_scores.mean()),
        "median_asymmetry_score": clean(asym_scores.median()),
        "user_msg_length_cv_median": clean(asymm_df["user_cv"].median()),
        "user_per_conv_stats": {
            "mean_user_msg_tokens":   clean(user_per_conv["user_msg_mean"].mean()),
            "median_user_msg_tokens": clean(user_per_conv["user_msg_median"].median()),
            "mean_asst_msg_tokens":   clean(asst_per_conv["asst_msg_mean"].mean()),
            "median_asst_msg_tokens": clean(asst_per_conv["asst_msg_median"].median()),
        },
    }
    print(f"  Asymmetry — mean={asymmetry_report['mean_asymmetry_score']:.3f}  "
          f"median={asymmetry_report['median_asymmetry_score']:.3f}")
    print("  Step 3 complete.")
except Exception as e:
    asymm_df = pd.DataFrame()
    report["warnings"].append(f"Step 3 error: {traceback.format_exc()}")
    print(f"  Step 3 ERROR: {e}")

# ==============================================================================
# STEP 4 — Positional dynamics
# ==============================================================================
print("\n-- Step 4: Positional dynamics ----------------------------------------")

positional_report = {}
try:
    msgs["position_bin"] = pd.cut(
        msgs["position_in_conversation"],
        bins=10,
        labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)],
    )

    positional = (
        msgs.groupby(["position_bin", "role"], observed=True)["token_count"]
        .agg(["mean", "median", "count"])
    )
    positional_report["user_length_by_position"] = clean_dict(
        positional.xs("user", level="role")["mean"].to_dict()
    )
    positional_report["assistant_length_by_position"] = clean_dict(
        positional.xs("assistant", level="role")["mean"].to_dict()
    )

    # 4b. By type (merge conversation_type from conv into msgs copy)
    msgs_wtype = msgs.merge(
        conv[["conversation_id", "conversation_type"]], on="conversation_id", how="left"
    )
    positional_by_type = (
        msgs_wtype.groupby(["conversation_type", "position_bin", "role"], observed=True)["token_count"]
        .mean()
    )

    # 4c. Opening vs closing
    first_user = msgs[msgs["is_first_user_msg"] & (msgs["role"] == "user")]
    last_user = (
        user_msgs.sort_values("msg_index")
        .groupby("conversation_id")
        .last()
    )
    first_mean = first_user["token_count"].dropna().mean()
    last_mean  = last_user["token_count"].dropna().mean()

    if first_mean > last_mean * 1.1:
        escl = "shortening"
    elif last_mean > first_mean * 1.1:
        escl = "lengthening"
    else:
        escl = "stable"

    positional_report["opening_vs_closing"] = {
        "first_user_msg_mean_tokens": clean(first_mean),
        "last_user_msg_mean_tokens":  clean(last_mean),
        "pattern": escl,
    }
    print(f"  Opening/closing pattern: {escl} "
          f"(first={first_mean:.0f} tok, last={last_mean:.0f} tok)")
    print("  Step 4 complete.")
except Exception as e:
    report["warnings"].append(f"Step 4 error: {traceback.format_exc()}")
    print(f"  Step 4 ERROR: {e}")

# ==============================================================================
# STEP 5 — Engagement intensity
# ==============================================================================
print("\n-- Step 5: Engagement intensity ---------------------------------------")

engagement_report = {}
try:
    # Filter inter_msg_seconds: keep 0 < x <= 3600; negative = data artifact
    valid_ims = msgs[msgs["inter_msg_seconds"].notna()].copy()
    filtered_gt1hr = int((valid_ims["inter_msg_seconds"] > 3600).sum())
    neg_count = int((valid_ims["inter_msg_seconds"] < 0).sum())
    valid_ims = valid_ims[(valid_ims["inter_msg_seconds"] > 0) & (valid_ims["inter_msg_seconds"] <= 3600)]

    user_responses = valid_ims[valid_ims["role"] == "user"]
    asst_responses = valid_ims[valid_ims["role"] == "assistant"]

    def response_stats(s):
        s = s.dropna()
        return {
            "median": clean(s.median()),
            "mean":   clean(s.mean()),
            "p25":    clean(s.quantile(0.25)),
            "p75":    clean(s.quantile(0.75)),
            "p90":    clean(s.quantile(0.90)),
            "count":  int(len(s)),
        }

    engagement_report["user_response_time_stats"]      = response_stats(user_responses["inter_msg_seconds"])
    engagement_report["assistant_response_time_stats"] = response_stats(asst_responses["inter_msg_seconds"])
    engagement_report["filtered_gt_1hr"] = filtered_gt1hr
    engagement_report["filtered_negative"] = neg_count

    # 5b. Pace by time-of-day  (attach conv time_of_day to msgs)
    msgs_tod = msgs.merge(
        conv[["conversation_id", "time_of_day", "day_of_week"]],
        on="conversation_id", how="left"
    )
    valid_ims_tod = msgs_tod[
        msgs_tod["inter_msg_seconds"].notna() &
        (msgs_tod["inter_msg_seconds"] > 0) &
        (msgs_tod["inter_msg_seconds"] <= 3600)
    ]

    conv_pace = (
        valid_ims_tod.groupby("conversation_id")["inter_msg_seconds"]
        .median()
        .rename("median_ims")
        .reset_index()
        .merge(conv[["conversation_id", "time_of_day", "day_of_week"]], on="conversation_id")
    )

    pace_by_tod = conv_pace.groupby("time_of_day")["median_ims"].median().round(3)
    engagement_report["pace_by_time_of_day"] = clean_dict(pace_by_tod.to_dict())

    # 5c. Pace by position
    user_valid = valid_ims[valid_ims["role"] == "user"].copy()
    pace_by_pos = (
        user_valid.groupby(
            pd.cut(user_valid["position_in_conversation"], bins=10)
        )["inter_msg_seconds"]
        .median()
        .round(3)
    )
    engagement_report["pace_by_position"] = clean_dict(
        {str(k): v for k, v in pace_by_pos.items()}
    )

    print(f"  Filtered >1hr: {filtered_gt1hr:,}  negative: {neg_count:,}")
    print(f"  User median response: {engagement_report['user_response_time_stats']['median']:.1f}s")
    print("  Step 5 complete.")
except Exception as e:
    report["warnings"].append(f"Step 5 error: {traceback.format_exc()}")
    print(f"  Step 5 ERROR: {e}")
    conv_pace = pd.DataFrame()

# ==============================================================================
# STEP 6 — Dominance shift detection
# ==============================================================================
print("\n-- Step 6: Dominance shift detection ----------------------------------")

dominance_report = {}
try:
    # Only conversations with >=10 user messages
    user_msg_counts = user_msgs.groupby("conversation_id").size()
    eligible_ids = user_msg_counts[user_msg_counts >= 10].index

    excluded_too_short = len(conv) - len(eligible_ids)

    def compute_rolling_dominance(conv_msgs, window=5):
        conv_msgs = conv_msgs.sort_values("msg_index").copy()
        conv_msgs["is_user"] = conv_msgs["role"] == "user"
        conv_msgs["user_tok_w"] = (
            conv_msgs["token_count"].where(conv_msgs["is_user"], 0)
            .rolling(window=window, min_periods=3).sum()
        )
        conv_msgs["total_tok_w"] = (
            conv_msgs["token_count"]
            .rolling(window=window, min_periods=3).sum()
        )
        conv_msgs["rolling_ratio"] = (
            conv_msgs["user_tok_w"] / conv_msgs["total_tok_w"].replace(0, np.nan)
        )
        return conv_msgs

    # Process eligible conversations
    rolling_results = []
    elig_msgs = msgs[msgs["conversation_id"].isin(eligible_ids)].copy()

    for cid, grp in elig_msgs.groupby("conversation_id"):
        try:
            rd = compute_rolling_dominance(grp)
            rr = rd["rolling_ratio"].dropna()
            if len(rr) < 4:
                continue
            mid = len(rr) // 2
            first_half_mean = rr.iloc[:mid].mean()
            second_half_mean = rr.iloc[mid:].mean()
            mean_rr = rr.mean()
            std_rr  = rr.std()

            if std_rr < 0.15:
                if mean_rr > 0.5:
                    pattern = "stable_user"
                elif mean_rr < 0.5:
                    pattern = "stable_assistant"
                else:
                    pattern = "stable_balanced"
            elif (first_half_mean - second_half_mean) >= 0.15:
                pattern = "user_to_assistant"
            elif (second_half_mean - first_half_mean) >= 0.15:
                pattern = "assistant_to_user"
            else:
                pattern = "volatile"

            rolling_results.append({
                "conversation_id": cid,
                "pattern": pattern,
                "mean_rolling_ratio": mean_rr,
                "std_rolling_ratio":  std_rr,
            })
        except Exception:
            pass

    rolling_df = pd.DataFrame(rolling_results)
    conv_with_rolling = len(rolling_df)

    pattern_dist = rolling_df["pattern"].value_counts().to_dict() if len(rolling_df) else {}
    # Ensure all categories present
    for p in ["stable_user", "stable_assistant", "stable_balanced",
              "user_to_assistant", "assistant_to_user", "volatile"]:
        pattern_dist.setdefault(p, 0)

    dominance_report["pattern_distribution"] = {k: int(v) for k, v in pattern_dist.items()}
    dominance_report["conversations_with_rolling_ratio"] = conv_with_rolling
    dominance_report["excluded_too_short"] = excluded_too_short

    # 6c. Monthly pattern distribution
    if len(rolling_df) > 0:
        rolling_df = rolling_df.merge(
            conv[["conversation_id", "year_month"]], on="conversation_id", how="left"
        )
        monthly_patterns = (
            rolling_df.groupby(["year_month", "pattern"])
            .size()
            .unstack(fill_value=0)
        )
        # Only months with ≥10 conversations
        monthly_enough = monthly_patterns[monthly_patterns.sum(axis=1) >= 10]
        monthly_pct = monthly_enough.div(monthly_enough.sum(axis=1), axis=0).round(3)
        dominance_report["monthly_pattern_distribution"] = clean_dict(
            monthly_pct.to_dict(orient="index")
        )
    else:
        dominance_report["monthly_pattern_distribution"] = {}

    print(f"  Eligible (≥10 user msgs): {len(eligible_ids):,}  "
          f"With rolling data: {conv_with_rolling:,}")
    print(f"  Patterns: {pattern_dist}")
    print("  Step 6 complete.")
except Exception as e:
    rolling_df = pd.DataFrame()
    report["warnings"].append(f"Step 6 error: {traceback.format_exc()}")
    print(f"  Step 6 ERROR: {e}")

# ==============================================================================
# STEP 7 — Cognitive signature
# ==============================================================================
print("\n-- Step 7: Cognitive signature ----------------------------------------")

signature = {}
try:
    ratio_valid = conv["user_token_ratio"].dropna()
    med_ratio = ratio_valid.median()

    # primary_mode
    if med_ratio > 0.5:
        primary_mode = "user_dominated"
    elif med_ratio < 0.5:
        primary_mode = "assistant_dominated"
    else:
        primary_mode = "balanced"

    # consistency: 1 - (std / range), clamped 0–1
    r_std = ratio_valid.std()
    r_range = ratio_valid.max() - ratio_valid.min()
    consistency = float(np.clip(1.0 - (r_std / r_range) if r_range > 0 else 1.0, 0, 1))

    # depth_preference
    med_turns = conv["turns"].median()
    if med_turns < 5:
        depth_pref = "brief"
    elif med_turns <= 15:
        depth_pref = "moderate"
    else:
        depth_pref = "deep"

    # pace_style — from user response median (already filtered)
    try:
        user_med_resp = engagement_report["user_response_time_stats"]["median"]
        if user_med_resp is None:
            raise ValueError
        if user_med_resp < 30:
            pace_style = "rapid"
        elif user_med_resp <= 300:
            pace_style = "measured"
        else:
            pace_style = "reflective"
    except Exception:
        pace_style = "unknown"

    # adaptation_tendency
    if len(rolling_df) > 0:
        adapt = rolling_df["pattern"].value_counts().idxmax()
    else:
        adapt = "unknown"

    # temporal_sensitivity (ANOVA ratio ~ time_of_day)
    try:
        tod_groups = [g["user_token_ratio"].dropna() for _, g in conv.groupby("time_of_day")]
        tod_groups = [g for g in tod_groups if len(g) >= 5]
        if len(tod_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*tod_groups)
            temporal_sensitivity = bool(p_value < 0.05)
        else:
            temporal_sensitivity = False
    except Exception:
        temporal_sensitivity = False

    # opening_style
    opening_style = str(conv["first_user_message_type"].value_counts().idxmax())

    # escalation_pattern (from Step 4)
    escalation_pattern = positional_report.get("opening_vs_closing", {}).get("pattern", "unknown")

    # era_sensitivity (eta-squared from ANOVA ratio ~ model_era)
    try:
        era_groups = [g["user_token_ratio"].dropna() for _, g in conv.groupby("model_era")]
        era_groups = [g for g in era_groups if len(g) >= 5]
        if len(era_groups) >= 2:
            f_stat_era, _ = stats.f_oneway(*era_groups)
            grand_mean = ratio_valid.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in era_groups)
            ss_total   = ((ratio_valid - grand_mean) ** 2).sum()
            era_sensitivity = float(ss_between / ss_total) if ss_total > 0 else 0.0
        else:
            era_sensitivity = 0.0
            report["warnings"].append("era_sensitivity: only one era in data; eta-squared set to 0.")
    except Exception as e:
        era_sensitivity = 0.0
        report["warnings"].append(f"era_sensitivity error: {e}")

    signature = {
        "primary_mode":        primary_mode,
        "consistency":         clean(consistency),
        "depth_preference":    depth_pref,
        "pace_style":          pace_style,
        "adaptation_tendency": adapt,
        "temporal_sensitivity": temporal_sensitivity,
        "opening_style":       opening_style,
        "escalation_pattern":  escalation_pattern,
        "era_sensitivity":     clean(era_sensitivity),
    }

    print(f"  Signature: mode={primary_mode}  depth={depth_pref}  pace={pace_style}")
    print(f"             adapt={adapt}  temporal_sens={temporal_sensitivity}")
    print("  Step 7 complete.")
except Exception as e:
    report["warnings"].append(f"Step 7 error: {traceback.format_exc()}")
    print(f"  Step 7 ERROR: {e}")

# ==============================================================================
# FIGURES
# ==============================================================================
print("\n-- Figures ------------------------------------------------------------")

def save_fig(fig, fname):
    path = figpath(fname)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures_generated.append(f"outputs/figures/turn_dynamics/{fname}")
    print(f"  Saved: {fname}")

# -- Figure 01: Token ratio distribution -------------------------------------
try:
    ratio = conv["user_token_ratio"].dropna()
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    ax.hist(ratio, bins=60, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white", linewidth=0.4,
            density=True, label="Histogram")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ratio)
    xg  = np.linspace(ratio.min(), ratio.max(), 300)
    ax.plot(xg, kde(xg), color=COLOR_ACCENT, linewidth=2, label="KDE")

    ax.axvline(0.5,           color="black",          linestyle="--", linewidth=1.5, label="Balanced (0.5)")
    ax.axvline(ratio.median(), color=COLOR_SECONDARY,  linestyle=":",  linewidth=1.5, label=f"Median ({ratio.median():.3f})")

    ax.set_xlabel("User Token Ratio  (user tokens / total tokens)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.set_title("Distribution of User Token Ratio", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)

    pct_user = (ratio > 0.5).mean() * 100
    pct_asst = (ratio < 0.5).mean() * 100
    textstr = (f"Mean: {ratio.mean():.3f}\nMedian: {ratio.median():.3f}\n"
               f"User-dominated: {pct_user:.1f}%\nAsst-dominated: {pct_asst:.1f}%")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=TICK_SIZE,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray"))
    ax.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax)
    save_fig(fig, "01_token_ratio_distribution.png")
except Exception as e:
    report["warnings"].append(f"Fig 01 error: {e}")
    print(f"  Fig 01 ERROR: {e}")

# -- Figure 02: Token ratio by conversation type ------------------------------
try:
    # Compute medians for sorting
    type_order = (
        conv.groupby("conversation_type")["user_token_ratio"]
        .median()
        .sort_values()
        .index.tolist()
    )
    conv_plot = conv[conv["conversation_type"].isin(type_order)].copy()
    conv_plot["conversation_type"] = pd.Categorical(
        conv_plot["conversation_type"], categories=type_order, ordered=True
    )
    type_counts = conv_plot["conversation_type"].value_counts()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    parts = ax.violinplot(
        [conv_plot[conv_plot["conversation_type"] == t]["user_token_ratio"].dropna()
         for t in type_order],
        positions=range(len(type_order)),
        showmedians=True, showextrema=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR_PRIMARY)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color(COLOR_ACCENT)
    parts["cmedians"].set_linewidth(2.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Balanced (0.5)")
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(
        [f"{t}\n(n={type_counts.get(t, 0):,})" for t in type_order],
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel("User Token Ratio", fontsize=LABEL_SIZE)
    ax.set_title("User Token Ratio by Conversation Type", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax)
    save_fig(fig, "02_token_ratio_by_type.png")
except Exception as e:
    report["warnings"].append(f"Fig 02 error: {e}")
    print(f"  Fig 02 ERROR: {e}")

# -- Figure 03: Token ratio trend (interactive HTML) --------------------------
try:
    monthly_base = conv.groupby("year_month")["user_token_ratio"].agg(
        ["mean", "median", "std", "count"]
    )
    monthly_base.columns = ["mean", "median", "std", "count"]
    monthly_base["p25"] = conv.groupby("year_month")["user_token_ratio"].quantile(0.25)
    monthly_base["p75"] = conv.groupby("year_month")["user_token_ratio"].quantile(0.75)
    monthly_g = monthly_base.reset_index().sort_values("year_month")

    conv_s = conv.sort_values("created_at").copy()
    conv_s["ratio_roll"] = (
        conv_s["user_token_ratio"].rolling(window=90, min_periods=30, center=True).mean()
    )
    conv_s = conv_s.dropna(subset=["ratio_roll"])

    fig3 = go.Figure()

    # Shaded band p25–p75
    fig3.add_trace(go.Scatter(
        x=list(monthly_g["year_month"]) + list(monthly_g["year_month"])[::-1],
        y=list(monthly_g["p75"]) + list(monthly_g["p25"])[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th–75th pct", showlegend=True, hoverinfo="skip",
    ))
    fig3.add_trace(go.Scatter(
        x=monthly_g["year_month"], y=monthly_g["mean"],
        mode="lines+markers", name="Monthly mean",
        line=dict(color=COLOR_PRIMARY, width=2),
        customdata=monthly_g[["count", "std"]].values,
        hovertemplate="Month: %{x}<br>Mean: %{y:.3f}<br>N: %{customdata[0]}<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=monthly_g["year_month"], y=monthly_g["median"],
        mode="lines+markers", name="Monthly median",
        line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
        hovertemplate="Month: %{x}<br>Median: %{y:.3f}<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=conv_s["created_at"].dt.strftime("%Y-%m-%d"), y=conv_s["ratio_roll"],
        mode="lines", name="90-day rolling mean",
        line=dict(color="black", width=2.5),
        hovertemplate="Date: %{x}<br>Rolling mean: %{y:.3f}<extra></extra>",
    ))

    # Era boundary lines — only add if the era date falls within the x-axis range
    ym_min = monthly_g["year_month"].min()
    ym_max = monthly_g["year_month"].max()
    for era_name, era_date in era_boundaries.items():
        era_ym = era_date[:7]  # "YYYY-MM"
        if ym_min <= era_ym <= ym_max:
            try:
                fig3.add_vline(
                    x=era_ym, line_dash="dot", line_color="gray",
                    annotation_text=era_name.replace("_", " "),
                    annotation_position="top left",
                    annotation_font_size=9,
                )
            except Exception:
                pass

    fig3.add_hline(y=0.5, line_dash="dash", line_color="lightgray",
                   annotation_text="Balanced", annotation_position="right")

    fig3.update_layout(
        title="User Token Ratio Over Time",
        xaxis_title="Month",
        yaxis_title="User Token Ratio",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig3.write_html(figpath("03_token_ratio_trend.html"), include_plotlyjs=True)
    figures_generated.append("outputs/figures/turn_dynamics/03_token_ratio_trend.html")
    print("  Saved: 03_token_ratio_trend.html")
except Exception as e:
    report["warnings"].append(f"Fig 03 error: {e}")
    print(f"  Fig 03 ERROR: {e}")

# -- Figure 04: Token ratio by model era -------------------------------------
try:
    era_g = conv.groupby("model_era")["user_token_ratio"].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    eras = era_g["model_era"].astype(str).tolist()
    x = np.arange(len(eras))
    w = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars_m = ax.bar(x - w/2, era_g["mean"],   width=w, color=COLOR_PRIMARY,   label="Mean",
                    yerr=era_g["std"], capsize=4, alpha=0.85)
    bars_md = ax.bar(x + w/2, era_g["median"], width=w, color=COLOR_ASSISTANT, label="Median",
                     alpha=0.85)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Balanced (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{e}\n(n={c:,})" for e, c in zip(eras, era_g["count"].tolist())],
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel("User Token Ratio", fontsize=LABEL_SIZE)
    ax.set_title("User Token Ratio by Model Era", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax)
    save_fig(fig, "04_token_ratio_by_era.png")
except Exception as e:
    report["warnings"].append(f"Fig 04 error: {e}")
    print(f"  Fig 04 ERROR: {e}")

# -- Figure 05: Turn depth distribution --------------------------------------
try:
    turns_all = conv["turns"].dropna()
    p99 = turns_all.quantile(0.99)
    turns_clip = turns_all[turns_all <= p99]

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.hist(turns_clip, bins=60, color=COLOR_PRIMARY, alpha=0.8,
            edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(turns_all.median(), color=COLOR_ACCENT, linestyle="--", linewidth=1.8,
               label=f"Median ({turns_all.median():.0f})")
    ax.axvline(turns_all.quantile(0.90), color=COLOR_SECONDARY, linestyle=":",
               linewidth=1.5, label=f"P90 ({turns_all.quantile(0.90):.0f})")
    ax.axvline(turns_all.quantile(0.99), color="darkred", linestyle=":",
               linewidth=1.5, label=f"P99 ({p99:.0f})")

    ax.set_xlabel("Turns (role switches)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count (log scale)", fontsize=LABEL_SIZE)
    ax.set_title("Turn Depth Distribution", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax)
    save_fig(fig, "05_turn_depth_distribution.png")
except Exception as e:
    report["warnings"].append(f"Fig 05 error: {e}")
    print(f"  Fig 05 ERROR: {e}")

# -- Figure 06: Turn depth vs. token ratio ------------------------------------
try:
    scatter_df = conv[["turns", "user_token_ratio", "conversation_type"]].dropna()
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(n=5000, random_state=42)

    type_palette = {
        "standard":     COLOR_PRIMARY,
        "tool_assisted": COLOR_ACCENT,
        "multimodal":   "#A9D18E",
        "single_turn":  "#666666",
        "empty":        "#CCCCCC",
    }
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    for ctype, grp in scatter_df.groupby("conversation_type"):
        ax.scatter(
            grp["turns"], grp["user_token_ratio"],
            alpha=0.35, s=12,
            color=type_palette.get(str(ctype), "#999999"),
            label=str(ctype),
        )

    # Binned means overlay
    bins = conv[["turns", "user_token_ratio"]].dropna().copy()
    bins["turn_bin_num"] = pd.cut(bins["turns"], bins=20)
    bin_means = bins.groupby("turn_bin_num", observed=True).agg(
        turns_mid=("turns", "median"),
        ratio_mean=("user_token_ratio", "mean"),
    ).dropna()
    ax.plot(bin_means["turns_mid"], bin_means["ratio_mean"],
            color="black", linewidth=2, zorder=5, label="Binned mean")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Turns (role switches)", fontsize=LABEL_SIZE)
    ax.set_ylabel("User Token Ratio", fontsize=LABEL_SIZE)
    ax.set_title("Turn Depth vs. User Token Ratio", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE - 1, markerscale=2)
    ax.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax)
    save_fig(fig, "06_turn_depth_vs_ratio.png")
except Exception as e:
    report["warnings"].append(f"Fig 06 error: {e}")
    print(f"  Fig 06 ERROR: {e}")

# -- Figure 07: Contribution asymmetry ----------------------------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: asymmetry score distribution
    asym = asymm_df["asymmetry_score"].dropna() if len(asymm_df) > 0 else pd.Series([], dtype=float)
    if len(asym) > 0:
        ax1.hist(asym, bins=50, color=COLOR_PRIMARY, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax1.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Balanced (0)")
        ax1.axvline(asym.median(), color=COLOR_ACCENT, linestyle=":", linewidth=1.5,
                    label=f"Median ({asym.median():.3f})")
        ax1.set_xlabel("Asymmetry Score", fontsize=LABEL_SIZE)
        ax1.set_ylabel("Count", fontsize=LABEL_SIZE)
        ax1.set_title("Message-Level Asymmetry\n(+1=user dominates, -1=asst dominates)",
                       fontsize=LABEL_SIZE, fontweight="bold")
        ax1.legend(fontsize=TICK_SIZE)
    sns.despine(ax=ax1)

    # Right: user CV vs. mean user tokens
    if len(asymm_df) > 0 and "user_cv" in asymm_df.columns:
        cv_df = asymm_df[["user_cv", "user_msg_mean", "conversation_type"]].dropna()
        if len(cv_df) > 5000:
            cv_df = cv_df.sample(5000, random_state=42)
        type_palette = {
            "standard":     COLOR_PRIMARY,
            "tool_assisted": COLOR_ACCENT,
            "multimodal":   "#A9D18E",
            "single_turn":  "#666666",
            "empty":        "#CCCCCC",
        }
        for ctype, grp in cv_df.groupby("conversation_type"):
            ax2.scatter(grp["user_msg_mean"], grp["user_cv"],
                        alpha=0.4, s=14,
                        color=type_palette.get(str(ctype), "#999999"),
                        label=str(ctype))

        xmed = cv_df["user_msg_mean"].median()
        ymed = cv_df["user_cv"].median()
        ax2.axvline(xmed, color="gray", linestyle="--", linewidth=0.8)
        ax2.axhline(ymed, color="gray", linestyle="--", linewidth=0.8)

        # Quadrant annotations
        ax2.text(0.02, 0.98, "Short &\nconsistent", transform=ax2.transAxes,
                 ha="left", va="top", fontsize=8, color=COLOR_SECONDARY)
        ax2.text(0.98, 0.98, "Long &\nconsistent", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=8, color=COLOR_SECONDARY)
        ax2.text(0.02, 0.02, "Short &\nvariable", transform=ax2.transAxes,
                 ha="left", va="bottom", fontsize=8, color=COLOR_SECONDARY)
        ax2.text(0.98, 0.02, "Long &\nvariable", transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=8, color=COLOR_SECONDARY)

        ax2.set_xlabel("Mean User Message Tokens", fontsize=LABEL_SIZE)
        ax2.set_ylabel("User Message Length CV", fontsize=LABEL_SIZE)
        ax2.set_title("User Message Style:\nLength vs. Variability", fontsize=LABEL_SIZE, fontweight="bold")
        ax2.legend(fontsize=TICK_SIZE - 1, markerscale=2)
    sns.despine(ax=ax2)

    plt.suptitle("Contribution Asymmetry Analysis", fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "07_contribution_asymmetry.png")
except Exception as e:
    report["warnings"].append(f"Fig 07 error: {e}")
    print(f"  Fig 07 ERROR: {e}")

# -- Figure 08: Message length by position (HTML) -----------------------------
try:
    msgs_pos = msgs.copy()
    msgs_pos["position_bin_num"] = pd.cut(
        msgs_pos["position_in_conversation"], bins=10,
        labels=[f"{i*10+5}%" for i in range(10)],
    )
    pos_agg = msgs_pos.groupby(["position_bin_num", "role"], observed=True)["token_count"].agg(
        mean="mean", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75)
    ).reset_index()

    user_pos = pos_agg[pos_agg["role"] == "user"].sort_values("position_bin_num")
    asst_pos = pos_agg[pos_agg["role"] == "assistant"].sort_values("position_bin_num")

    fig8 = go.Figure()
    # Assistant band
    fig8.add_trace(go.Scatter(
        x=list(asst_pos["position_bin_num"]) + list(asst_pos["position_bin_num"])[::-1],
        y=list(asst_pos["p75"]) + list(asst_pos["p25"])[::-1],
        fill="toself", fillcolor="rgba(169,209,142,0.2)",
        line=dict(color="rgba(255,255,255,0)"), name="Asst 25–75%", hoverinfo="skip",
    ))
    # User band
    fig8.add_trace(go.Scatter(
        x=list(user_pos["position_bin_num"]) + list(user_pos["position_bin_num"])[::-1],
        y=list(user_pos["p75"]) + list(user_pos["p25"])[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.2)",
        line=dict(color="rgba(255,255,255,0)"), name="User 25–75%", hoverinfo="skip",
    ))
    fig8.add_trace(go.Scatter(
        x=user_pos["position_bin_num"], y=user_pos["mean"],
        mode="lines+markers", name="User mean tokens",
        line=dict(color=COLOR_USER, width=2.5),
        hovertemplate="Position: %{x}<br>User mean: %{y:.0f} tokens<extra></extra>",
    ))
    fig8.add_trace(go.Scatter(
        x=asst_pos["position_bin_num"], y=asst_pos["mean"],
        mode="lines+markers", name="Assistant mean tokens",
        line=dict(color="#52973E", width=2.5),
        hovertemplate="Position: %{x}<br>Asst mean: %{y:.0f} tokens<extra></extra>",
    ))
    fig8.update_layout(
        title="Message Length by Position in Conversation",
        xaxis_title="Position in Conversation",
        yaxis_title="Mean Token Count",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig8.write_html(figpath("08_msg_length_by_position.html"), include_plotlyjs=True)
    figures_generated.append("outputs/figures/turn_dynamics/08_msg_length_by_position.html")
    print("  Saved: 08_msg_length_by_position.html")
except Exception as e:
    report["warnings"].append(f"Fig 08 error: {e}")
    print(f"  Fig 08 ERROR: {e}")

# -- Figure 09: Engagement heatmap --------------------------------------------
try:
    msgs_conv = msgs.merge(
        conv[["conversation_id", "time_of_day", "day_of_week"]], on="conversation_id", how="left"
    )
    valid_pace = msgs_conv[
        msgs_conv["inter_msg_seconds"].notna() &
        (msgs_conv["inter_msg_seconds"] > 0) &
        (msgs_conv["inter_msg_seconds"] <= 3600)
    ].copy()

    # Conv-level pace merged with time info
    pace_conv = valid_pace.groupby("conversation_id")["inter_msg_seconds"].median().reset_index()
    pace_conv.columns = ["conversation_id", "median_ims"]
    pace_conv = pace_conv.merge(
        conv[["conversation_id", "time_of_day", "day_of_week"]], on="conversation_id"
    )
    pace_conv["day_label"] = pace_conv["day_of_week"].map(DOW_LABELS)

    pivot = pace_conv.pivot_table(
        index="day_label", columns="time_of_day", values="median_ims", aggfunc="median"
    )
    # Reorder
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tod_order_present = [t for t in TOD_ORDER if t in pivot.columns]
    pivot = pivot.reindex(index=[d for d in dow_order if d in pivot.index],
                          columns=tod_order_present)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", annot=True, fmt=".0f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Median inter-msg seconds"},
        vmin=0,
    )
    ax.set_title("Engagement Intensity by Day & Time of Day\n(lower = faster / more engaged)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Time of Day", fontsize=LABEL_SIZE)
    ax.set_ylabel("Day of Week", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    save_fig(fig, "09_engagement_heatmap.png")
except Exception as e:
    report["warnings"].append(f"Fig 09 error: {e}")
    print(f"  Fig 09 ERROR: {e}")

# -- Figure 10: Dominance shift timeline (HTML) -------------------------------
try:
    if len(rolling_df) > 0 and "year_month" in rolling_df.columns:
        monthly_pcts = (
            rolling_df.groupby(["year_month", "pattern"])
            .size()
            .unstack(fill_value=0)
        )
        monthly_count = monthly_pcts.sum(axis=1)
        monthly_pcts = monthly_pcts[monthly_count >= 10]
        if len(monthly_pcts) > 0:
            monthly_pcts_norm = monthly_pcts.div(monthly_pcts.sum(axis=1), axis=0) * 100

            pattern_colors = {
                "stable_user":      COLOR_USER,
                "stable_assistant": "#A9D18E",
                "stable_balanced":  COLOR_BALANCED,
                "assistant_to_user":"#C55A11",
                "user_to_assistant":"#F4B183",
                "volatile":         "#808080",
            }

            fig10 = go.Figure()
            for pat in ["stable_assistant", "stable_balanced", "stable_user",
                        "user_to_assistant", "assistant_to_user", "volatile"]:
                if pat in monthly_pcts_norm.columns:
                    fig10.add_trace(go.Scatter(
                        x=monthly_pcts_norm.index,
                        y=monthly_pcts_norm[pat],
                        mode="lines",
                        stackgroup="one",
                        name=pat.replace("_", " "),
                        fillcolor=pattern_colors.get(pat, "#999999"),
                        line=dict(color=pattern_colors.get(pat, "#999999")),
                        hovertemplate=f"{pat}: %{{y:.1f}}%<extra></extra>",
                    ))
            fig10.update_layout(
                title="Dominance Pattern Distribution Over Time",
                xaxis_title="Month",
                yaxis_title="% of Conversations",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
            )
            fig10.write_html(figpath("10_dominance_shift_timeline.html"), include_plotlyjs=True)
            figures_generated.append("outputs/figures/turn_dynamics/10_dominance_shift_timeline.html")
            print("  Saved: 10_dominance_shift_timeline.html")
        else:
            report["warnings"].append("Fig 10: no months with ≥10 conversations with rolling data.")
            print("  Fig 10 SKIP: insufficient monthly data.")
    else:
        report["warnings"].append("Fig 10: no rolling dominance data.")
        print("  Fig 10 SKIP: no rolling data.")
except Exception as e:
    report["warnings"].append(f"Fig 10 error: {e}")
    print(f"  Fig 10 ERROR: {e}")

# -- Figure 11: Temporal patterns ---------------------------------------------
try:
    conv["dow_label"] = pd.Categorical(
        conv["day_of_week"].map(DOW_LABELS),
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ordered=True
    )
    conv["time_of_day_cat"] = pd.Categorical(
        conv["time_of_day"].astype(str),
        categories=TOD_ORDER, ordered=True
    )

    def bar_with_sem(ax, group_col, value_col, df, palette=None, **kw):
        agg = df.groupby(group_col, observed=True)[value_col].agg(["mean", "sem", "count"])
        agg = agg.dropna()
        bars = ax.bar(
            range(len(agg)), agg["mean"],
            color=palette or COLOR_PRIMARY,
            alpha=0.85,
            yerr=agg["sem"], capsize=4, **kw
        )
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index.astype(str), fontsize=TICK_SIZE)
        return agg

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)

    bar_with_sem(axes[0, 0], "time_of_day_cat", "user_token_ratio", conv)
    axes[0, 0].set_title("Token Ratio by Time of Day", fontsize=LABEL_SIZE, fontweight="bold")
    axes[0, 0].set_ylabel("User Token Ratio", fontsize=TICK_SIZE)
    axes[0, 0].axhline(conv["user_token_ratio"].mean(), color=COLOR_ACCENT,
                       linestyle="--", linewidth=1, label="Overall mean")
    axes[0, 0].legend(fontsize=8)
    sns.despine(ax=axes[0, 0])

    bar_with_sem(axes[0, 1], "dow_label", "user_token_ratio", conv)
    axes[0, 1].set_title("Token Ratio by Day of Week", fontsize=LABEL_SIZE, fontweight="bold")
    axes[0, 1].set_ylabel("User Token Ratio", fontsize=TICK_SIZE)
    axes[0, 1].axhline(conv["user_token_ratio"].mean(), color=COLOR_ACCENT,
                       linestyle="--", linewidth=1)
    sns.despine(ax=axes[0, 1])

    bar_with_sem(axes[1, 0], "time_of_day_cat", "turns", conv)
    axes[1, 0].set_title("Turn Depth by Time of Day", fontsize=LABEL_SIZE, fontweight="bold")
    axes[1, 0].set_ylabel("Mean Turns", fontsize=TICK_SIZE)
    axes[1, 0].axhline(conv["turns"].mean(), color=COLOR_ACCENT, linestyle="--", linewidth=1)
    sns.despine(ax=axes[1, 0])

    bar_with_sem(axes[1, 1], "dow_label", "turns", conv)
    axes[1, 1].set_title("Turn Depth by Day of Week", fontsize=LABEL_SIZE, fontweight="bold")
    axes[1, 1].set_ylabel("Mean Turns", fontsize=TICK_SIZE)
    axes[1, 1].axhline(conv["turns"].mean(), color=COLOR_ACCENT, linestyle="--", linewidth=1)
    sns.despine(ax=axes[1, 1])

    # Mark ANOVA significance
    try:
        tod_g = [g["user_token_ratio"].dropna() for _, g in conv.groupby("time_of_day_cat", observed=True)]
        tod_g = [g for g in tod_g if len(g) >= 5]
        if len(tod_g) >= 2:
            _, p = stats.f_oneway(*tod_g)
            if p < 0.05:
                axes[0, 0].set_title(f"Token Ratio by Time of Day *",
                                     fontsize=LABEL_SIZE, fontweight="bold")
    except Exception:
        pass

    plt.suptitle("Temporal Patterns: Token Ratio & Turn Depth", fontsize=TITLE_SIZE,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "11_temporal_patterns.png")
except Exception as e:
    report["warnings"].append(f"Fig 11 error: {e}")
    print(f"  Fig 11 ERROR: {e}")

# -- Figure 12: Signature summary ---------------------------------------------
try:
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=FIGSIZE_TALL)
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.4)
    ax_text   = fig.add_subplot(gs[:2, 0])  # signature text card
    ax_radar  = fig.add_subplot(gs[:2, 1], polar=True)  # radar
    ax_spark  = fig.add_subplot(gs[2, :])   # sparkline

    # -- Text card -------------------------------------------------------------
    ax_text.axis("off")
    ax_text.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96, transform=ax_text.transAxes,
        boxstyle="round,pad=0.02", facecolor="#F0F4F8", edgecolor="#BDD7EE", linewidth=2,
    ))

    sig_lines = [
        ("COGNITIVE SIGNATURE", True),
        ("", False),
        (f"Primary mode:     {signature.get('primary_mode', '—')}", False),
        (f"Consistency:      {signature.get('consistency', 0):.3f}", False),
        (f"Depth preference: {signature.get('depth_preference', '—')}", False),
        (f"Pace style:       {signature.get('pace_style', '—')}", False),
        (f"Adaptation:       {signature.get('adaptation_tendency', '—')}", False),
        (f"Opening style:    {signature.get('opening_style', '—')}", False),
        (f"Escalation:       {signature.get('escalation_pattern', '—')}", False),
        (f"Temporal sens.:   {'Yes' if signature.get('temporal_sensitivity') else 'No'}", False),
        (f"Era sensitivity:  {signature.get('era_sensitivity', 0):.3f}", False),
    ]
    y0 = 0.93
    for line, bold in sig_lines:
        if bold:
            ax_text.text(0.08, y0, line, transform=ax_text.transAxes,
                         fontsize=11, fontweight="bold", color=COLOR_PRIMARY, va="top")
        else:
            ax_text.text(0.08, y0, line, transform=ax_text.transAxes,
                         fontsize=9.5, color="#333333", va="top", family="monospace")
        y0 -= 0.085

    # -- Radar chart ----------------------------------------------------------
    radar_labels = ["Token\nRatio", "Consistency", "Depth\n(norm)", "Pace\n(inv)", "Era\nSens"]

    # Normalize each dimension 0→1
    ratio_norm = float(np.clip(signature.get("consistency", 0.5), 0, 1))
    raw_ratio  = float(np.clip(conv["user_token_ratio"].dropna().median() * 2, 0, 1))
    depth_map  = {"brief": 0.2, "moderate": 0.5, "deep": 1.0}
    depth_norm = depth_map.get(signature.get("depth_preference", "moderate"), 0.5)
    pace_map   = {"rapid": 1.0, "measured": 0.5, "reflective": 0.15, "unknown": 0.5}
    pace_norm  = pace_map.get(signature.get("pace_style", "measured"), 0.5)
    era_norm   = float(np.clip(signature.get("era_sensitivity", 0) * 10, 0, 1))

    values = [raw_ratio, ratio_norm, depth_norm, pace_norm, era_norm]
    N = len(values)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.plot(angles_plot, values_plot, color=COLOR_PRIMARY, linewidth=2)
    ax_radar.fill(angles_plot, values_plot, color=COLOR_PRIMARY, alpha=0.25)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(radar_labels, fontsize=8.5)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax_radar.set_title("Signature Profile", fontsize=LABEL_SIZE, fontweight="bold", pad=14)
    ax_radar.grid(color="lightgray", linewidth=0.7)

    # -- Sparkline of monthly median ratio ------------------------------------
    monthly_spark = (
        conv.groupby("year_month")["user_token_ratio"]
        .median()
        .reset_index()
        .sort_values("year_month")
    )
    ax_spark.plot(monthly_spark["year_month"], monthly_spark["user_token_ratio"],
                  color=COLOR_PRIMARY, linewidth=2, marker="o", markersize=4)
    ax_spark.fill_between(range(len(monthly_spark)),
                          monthly_spark["user_token_ratio"],
                          alpha=0.15, color=COLOR_PRIMARY)
    ax_spark.set_xticks(range(len(monthly_spark)))
    ax_spark.set_xticklabels(monthly_spark["year_month"], rotation=45,
                              ha="right", fontsize=8)
    ax_spark.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_spark.set_title("Monthly Median User Token Ratio", fontsize=LABEL_SIZE, fontweight="bold")
    ax_spark.set_ylabel("Median Ratio", fontsize=TICK_SIZE)
    ax_spark.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax_spark)

    plt.suptitle("Turn Dynamics — Cognitive Signature Summary",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)
    save_fig(fig, "12_signature_summary.png")
except Exception as e:
    report["warnings"].append(f"Fig 12 error: {e}")
    print(f"  Fig 12 ERROR: {e}")

# ==============================================================================
# BUILD & WRITE REPORT
# ==============================================================================
print("\n-- Writing report -----------------------------------------------------")

report.update({
    "token_ratio":            token_ratio_report,
    "turn_depth":             turn_depth_report,
    "contribution_asymmetry": asymmetry_report,
    "positional_dynamics":    positional_report,
    "engagement_intensity":   engagement_report,
    "dominance_shifts":       dominance_report,
    "cognitive_signature":    signature,
    "figures_generated":      figures_generated,
})

clean_report = clean_dict(report)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_report, f, indent=2, ensure_ascii=False)
print(f"  Report written to: {REPORT_PATH}")

# ==============================================================================
# VALIDATION CHECKLIST
# ==============================================================================
print("\n== VALIDATION CHECKLIST ==============================================")

checks = {}

# 1. Report exists with all keys
required_keys = {"input_data", "token_ratio", "turn_depth", "contribution_asymmetry",
                 "positional_dynamics", "engagement_intensity", "dominance_shifts",
                 "cognitive_signature", "figures_generated"}
checks["report_exists_all_keys"] = os.path.isfile(REPORT_PATH) and required_keys.issubset(clean_report.keys())

# 2. All 12 figures exist
expected_figs = [
    "01_token_ratio_distribution.png",
    "02_token_ratio_by_type.png",
    "03_token_ratio_trend.html",
    "04_token_ratio_by_era.png",
    "05_turn_depth_distribution.png",
    "06_turn_depth_vs_ratio.png",
    "07_contribution_asymmetry.png",
    "08_msg_length_by_position.html",
    "09_engagement_heatmap.png",
    "10_dominance_shift_timeline.html",
    "11_temporal_patterns.png",
    "12_signature_summary.png",
]
all_figs_exist = all(os.path.isfile(figpath(f)) for f in expected_figs)
checks["all_12_figures_exist"] = all_figs_exist
if not all_figs_exist:
    for f in expected_figs:
        if not os.path.isfile(figpath(f)):
            print(f"  MISSING: {f}")

# 3. PNGs ≥ 10KB
png_figs = [f for f in expected_figs if f.endswith(".png")]
all_pngs_ok = all(
    os.path.isfile(figpath(f)) and os.path.getsize(figpath(f)) >= 10_240
    for f in png_figs
)
checks["all_pngs_gte_10kb"] = all_pngs_ok
if not all_pngs_ok:
    for f in png_figs:
        path = figpath(f)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        if size < 10_240:
            print(f"  SMALL PNG ({size} bytes): {f}")

# 4. HTML files exist and have content
html_figs = [f for f in expected_figs if f.endswith(".html")]
all_htmls_ok = all(
    os.path.isfile(figpath(f)) and os.path.getsize(figpath(f)) > 1000
    for f in html_figs
)
checks["html_figures_have_content"] = all_htmls_ok

# 5. No NaN/Inf in JSON (the clean_dict should catch these)
def check_no_nan(obj, path=""):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return False, path
    if isinstance(obj, dict):
        for k, v in obj.items():
            ok, p = check_no_nan(v, f"{path}.{k}")
            if not ok:
                return False, p
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            ok, p = check_no_nan(v, f"{path}[{i}]")
            if not ok:
                return False, p
    return True, ""

no_nan, nan_path = check_no_nan(clean_report)
checks["no_nan_inf_in_report"] = no_nan
if not no_nan:
    print(f"  NaN/Inf found at: {nan_path}")

# 6. Cognitive signature fully populated
sig_keys = {"primary_mode", "consistency", "depth_preference", "pace_style",
            "adaptation_tendency", "temporal_sensitivity", "opening_style",
            "escalation_pattern", "era_sensitivity"}
checks["cognitive_signature_complete"] = sig_keys.issubset(clean_report.get("cognitive_signature", {}).keys())

# 7. Pattern distribution sums to conversations_with_rolling_ratio
ds = clean_report.get("dominance_shifts", {})
pattern_sum = sum(ds.get("pattern_distribution", {}).values())
conv_with_rr = ds.get("conversations_with_rolling_ratio", -1)
checks["pattern_dist_consistent"] = (pattern_sum == conv_with_rr)
if pattern_sum != conv_with_rr:
    print(f"  Pattern sum={pattern_sum}  conversations_with_rolling_ratio={conv_with_rr}")

# 8. Signature summary figure renders (size check)
checks["signature_summary_readable"] = (
    os.path.isfile(figpath("12_signature_summary.png")) and
    os.path.getsize(figpath("12_signature_summary.png")) >= 30_000
)

print()
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")

all_pass = all(checks.values())
print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
if report["warnings"]:
    print(f"\nWarnings ({len(report['warnings'])}):")
    for w in report["warnings"]:
        print(f"  • {w[:120]}")
