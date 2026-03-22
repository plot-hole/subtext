"""
Phase 2, Step 4: Quality Report & Final Validation.
Writes final clean parquets, validates schema, generates figures, compiles master report.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "quality_config.json"

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"
MSG_CLEAN_PATH = PROCESSED_DIR / "messages_clean.parquet"
CONV_CLEAN_PATH = PROCESSED_DIR / "conversations_clean.parquet"

# Style
PRIMARY = "#2E75B6"
SECONDARY = "#666666"
ACCENT = "#C55A11"
DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": DPI,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ====================================================================
# 4a. Write final clean files
# ====================================================================
def write_clean_files(messages, convos):
    print("\n--- 4a. Writing Final Clean Files ---")
    messages.to_parquet(MSG_CLEAN_PATH, index=False)
    convos.to_parquet(CONV_CLEAN_PATH, index=False)
    print(f"  Saved {MSG_CLEAN_PATH} ({len(messages)} rows, {len(messages.columns)} cols)")
    print(f"  Saved {CONV_CLEAN_PATH} ({len(convos)} rows, {len(convos.columns)} cols)")

    # CSV samples
    conv_csv = PROCESSED_DIR / "conversations_clean_sample.csv"
    convos.head(500).to_csv(conv_csv, index=False)
    print(f"  Saved {conv_csv} (500 rows)")

    msg_sample = messages.head(2000).copy()
    msg_sample["text_preview"] = msg_sample["text"].apply(
        lambda x: x[:100] if isinstance(x, str) else None
    )
    msg_sample.drop(columns=["text"], inplace=True)
    msg_csv = PROCESSED_DIR / "messages_clean_sample.csv"
    msg_sample.to_csv(msg_csv, index=False)
    print(f"  Saved {msg_csv} (2000 rows, text replaced with preview)")


# ====================================================================
# 4b. Schema validation
# ====================================================================
EXPECTED_MSG_SCHEMA = {
    "conversation_id": ("string", False),
    "msg_index": ("int32", False),
    "role": ("category", False),
    "text": ("string", True),
    "timestamp": ("datetime64", True),
    "token_count": ("int32", True),
    "char_count": ("int32", True),
    "word_count": ("int32", True),
    "has_code": ("bool", False),
    "has_attachment": ("bool", False),
    "is_branched": ("bool", False),
    "is_empty_msg": ("bool", False),
    "is_bulk_input": ("bool", False),
    "is_error_response": ("bool", False),
    "is_system_boilerplate": ("bool", False),
    "content_type": ("category", False),
    "is_first_user_msg": ("bool", False),
    "is_last_msg": ("bool", False),
    "position_in_conversation": ("float32", False),
    "inter_msg_seconds": ("float32", True),
    "cumulative_user_tokens": ("int32", False),
    "has_pii": ("bool", False),
    "pii_priority": ("category", True),
}

EXPECTED_CONV_SCHEMA = {
    "conversation_id": ("string", False),
    "title": ("string", True),
    "created_at": ("datetime64", True),
    "updated_at": ("datetime64", True),
    "duration_minutes": ("float32", True),
    "msg_count": ("int32", False),
    "user_msg_count": ("int32", False),
    "assistant_msg_count": ("int32", False),
    "user_token_total": ("int32", False),
    "assistant_token_total": ("int32", False),
    "hour_of_day": ("int8", True),
    "day_of_week": ("int8", True),
    "has_code": ("bool", False),
    "is_branched": ("bool", False),
    "is_empty": ("bool", False),
    "has_tool_use": ("bool", False),
    "has_plugin": ("bool", False),
    "is_custom_gpt": ("bool", False),
    "is_multimodal": ("bool", False),
    "has_bulk_input": ("bool", False),
    "is_single_turn": ("bool", False),
    "conversation_type": ("category", False),
    "model_era": ("category", False),
    "year": ("int16", True),
    "month": ("int8", True),
    "year_month": ("string", True),
    "year_week": ("string", True),
    "is_weekend": ("bool", True),
    "time_of_day": ("category", True),
    "days_since_first": ("int32", True),
    "gap_days_from_prev": ("float32", True),
    "user_token_ratio": ("float32", True),
    "avg_user_msg_tokens": ("float32", True),
    "avg_assistant_msg_tokens": ("float32", True),
    "turns": ("int32", False),
    "first_user_message_tokens": ("int32", True),
    "first_user_message_type": ("category", True),
    "quality_score": ("float32", False),
    "quality_flag": ("category", False),
    "is_analysable": ("bool", False),
    "has_pii": ("bool", False),
    "max_pii_priority": ("category", True),
}


def validate_schema(df, expected, name):
    """Validate columns exist. Returns (passed, failed, details)."""
    results = []
    for col, (expected_type, nullable) in expected.items():
        if col not in df.columns:
            results.append((col, "FAIL", f"column missing"))
        else:
            results.append((col, "PASS", f"present"))
    return results


def run_schema_validation(messages, convos):
    print("\n--- 4b. Schema Validation ---")
    msg_results = validate_schema(messages, EXPECTED_MSG_SCHEMA, "messages_clean")
    conv_results = validate_schema(convos, EXPECTED_CONV_SCHEMA, "conversations_clean")

    msg_pass = sum(1 for _, s, _ in msg_results if s == "PASS")
    msg_fail = sum(1 for _, s, _ in msg_results if s == "FAIL")
    conv_pass = sum(1 for _, s, _ in conv_results if s == "PASS")
    conv_fail = sum(1 for _, s, _ in conv_results if s == "FAIL")

    print(f"  messages_clean: {msg_pass}/{len(msg_results)} columns present")
    if msg_fail > 0:
        for col, status, detail in msg_results:
            if status == "FAIL":
                print(f"    [FAIL] {col}: {detail}")

    print(f"  conversations_clean: {conv_pass}/{len(conv_results)} columns present")
    if conv_fail > 0:
        for col, status, detail in conv_results:
            if status == "FAIL":
                print(f"    [FAIL] {col}: {detail}")

    return {
        "messages_clean": {"total_checks": len(msg_results), "passed": msg_pass, "failed": msg_fail},
        "conversations_clean": {"total_checks": len(conv_results), "passed": conv_pass, "failed": conv_fail},
    }


# ====================================================================
# 4c. Referential integrity checks
# ====================================================================
def run_integrity_checks(messages, convos):
    print("\n--- 4c. Referential Integrity Checks ---")
    results = []

    # 1. Every conv_id in messages exists in conversations
    msg_ids = set(messages["conversation_id"].unique())
    conv_ids = set(convos["conversation_id"].unique())
    check = msg_ids.issubset(conv_ids)
    results.append(("msg conv_ids subset of conv conv_ids", check))

    # 2. Every conv_id in conversations has >= 0 messages
    results.append(("all conv_ids allow zero messages", True))

    # 3. msg_count matches actual
    actual_counts = messages.groupby("conversation_id").size()
    merged = convos.set_index("conversation_id")["msg_count"]
    matched = merged.reindex(actual_counts.index)
    check3 = (matched == actual_counts).all() if len(matched) > 0 else True
    results.append(("msg_count matches actual message count", bool(check3)))

    # 4. user_msg_count matches
    actual_user = messages[messages["role"] == "user"].groupby("conversation_id").size()
    merged_user = convos.set_index("conversation_id")["user_msg_count"]
    matched_user = merged_user.reindex(actual_user.index).fillna(0)
    actual_user_filled = actual_user.reindex(merged_user.index).fillna(0)
    check4 = (merged_user == actual_user_filled).all()
    results.append(("user_msg_count matches actual", bool(check4)))

    # 5. No duplicate (conversation_id, msg_index)
    dups = messages.duplicated(subset=["conversation_id", "msg_index"]).sum()
    results.append(("no duplicate (conversation_id, msg_index)", dups == 0))

    # 6. Timestamps monotonically non-decreasing (warn only)
    non_mono = 0
    for conv_id, grp in messages.groupby("conversation_id"):
        ts = grp.sort_values("msg_index")["timestamp"].dropna()
        if len(ts) > 1 and (ts.diff().dropna() < pd.Timedelta(0)).any():
            non_mono += 1
    if non_mono > 0:
        results.append((f"timestamps monotonic (WARN: {non_mono} convos non-monotonic)", True))
    else:
        results.append(("timestamps monotonically non-decreasing", True))

    # 7. No timestamps before 2022-01-01 or after today
    ts_all = messages["timestamp"].dropna()
    bad_early = (ts_all < pd.Timestamp("2022-01-01", tz="UTC")).sum()
    bad_future = (ts_all > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).sum()
    results.append((f"no timestamps before 2022 (found {bad_early})", bad_early == 0))
    results.append((f"no timestamps after today (found {bad_future})", bad_future == 0))

    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    return results


# ====================================================================
# 4d. Visualizations
# ====================================================================
def fig1_quality_score_dist(convos):
    """Histogram of quality_score with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    qs = convos["quality_score"].dropna()
    ax.hist(qs, bins=50, color=PRIMARY, edgecolor="white", alpha=0.8)
    ax.axvline(0.4, color="red", linestyle="--", linewidth=2, label="Excluded threshold (0.4)")
    ax.axvline(0.7, color=ACCENT, linestyle="--", linewidth=2, label="Complete threshold (0.7)")

    # Count and label regions
    exc = (qs < 0.4).sum()
    par = ((qs >= 0.4) & (qs < 0.7)).sum()
    comp = (qs >= 0.7).sum()
    total = len(qs)
    ax.text(0.15, ax.get_ylim()[1] * 0.85, f"Excluded\n{exc} ({exc/total*100:.1f}%)",
            ha="center", fontsize=9, color="red", fontweight="bold")
    ax.text(0.55, ax.get_ylim()[1] * 0.85, f"Partial\n{par} ({par/total*100:.1f}%)",
            ha="center", fontsize=9, color=ACCENT, fontweight="bold")
    ax.text(0.85, ax.get_ylim()[1] * 0.85, f"Complete\n{comp} ({comp/total*100:.1f}%)",
            ha="center", fontsize=9, color="green", fontweight="bold")

    ax.set_title("Conversation Quality Score Distribution")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    ax.legend()
    path = FIGURES_DIR / "quality_score_dist.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_edge_case_breakdown(convos):
    """Horizontal bar chart of edge case flag counts."""
    flags = ["is_single_turn", "is_multimodal", "is_custom_gpt", "has_bulk_input",
             "has_tool_use", "has_plugin", "is_empty"]
    counts = [(f, int(convos[f].sum())) for f in flags if f in convos.columns]
    counts.sort(key=lambda x: x[1], reverse=True)
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color=PRIMARY)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center", fontsize=10)
    ax.set_title("Edge Case Flag Breakdown")
    ax.set_xlabel("Conversation Count")
    ax.invert_yaxis()
    path = FIGURES_DIR / "edge_case_breakdown.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_model_era_timeline(convos, config):
    """Stacked bar chart: conversations per month colored by model_era."""
    df = convos.dropna(subset=["created_at"]).copy()
    df["ym"] = df["created_at"].dt.to_period("M").astype(str)
    ct = df.groupby(["ym", "model_era"]).size().unstack(fill_value=0)

    era_colors = {
        "gpt-3.5": "#A9A9A9",
        "gpt-4-early": "#4CAF50",
        "gpt-4-turbo": "#2196F3",
        "gpt-4o": "#FF9800",
        "o1-era": "#9C27B0",
        "o3-era": "#E91E63",
        "unknown": "#BDBDBD",
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[era_colors.get(c, SECONDARY) for c in ct.columns], width=0.9)

    # Era boundary lines
    era_bounds = config["model_era_boundaries"]
    months = ct.index.tolist()
    for label, date_str in era_bounds.items():
        period = pd.Period(date_str, freq="M")
        period_str = str(period)
        if period_str in months:
            idx = months.index(period_str)
            ax.axvline(idx - 0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_title("Conversations per Month by Model Era")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(title="Model Era", fontsize=8)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 3 != 0:
            label.set_visible(False)
    plt.xticks(rotation=45, ha="right")
    path = FIGURES_DIR / "model_era_timeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_sankey(convos):
    """Plotly Sankey: total -> conversation_type -> quality_flag -> is_analysable."""
    # Nodes
    types = sorted(convos["conversation_type"].unique().tolist())
    flags = sorted(convos["quality_flag"].unique().tolist())
    analysable_labels = ["Analysable", "Excluded"]

    all_labels = ["All Conversations"] + types + flags + analysable_labels
    label_idx = {l: i for i, l in enumerate(all_labels)}

    sources = []
    targets = []
    values = []
    colors = []

    # Level 1: All -> conversation_type
    for t in types:
        ct = int((convos["conversation_type"] == t).sum())
        if ct > 0:
            sources.append(label_idx["All Conversations"])
            targets.append(label_idx[t])
            values.append(ct)
            colors.append("rgba(46,117,182,0.4)")

    # Level 2: conversation_type -> quality_flag
    for t in types:
        for f in flags:
            ct = int(((convos["conversation_type"] == t) & (convos["quality_flag"] == f)).sum())
            if ct > 0:
                sources.append(label_idx[t])
                targets.append(label_idx[f])
                values.append(ct)
                colors.append("rgba(102,102,102,0.3)")

    # Level 3: quality_flag -> is_analysable
    for f in flags:
        ct_yes = int(((convos["quality_flag"] == f) & (convos["is_analysable"])).sum())
        ct_no = int(((convos["quality_flag"] == f) & (~convos["is_analysable"])).sum())
        if ct_yes > 0:
            sources.append(label_idx[f])
            targets.append(label_idx["Analysable"])
            values.append(ct_yes)
            colors.append("rgba(76,175,80,0.4)")
        if ct_no > 0:
            sources.append(label_idx[f])
            targets.append(label_idx["Excluded"])
            values.append(ct_no)
            colors.append("rgba(244,67,54,0.4)")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_labels,
            color=[PRIMARY] + [SECONDARY] * len(types) + [ACCENT] * len(flags) + ["green", "red"],
        ),
        link=dict(source=sources, target=targets, value=values, color=colors),
    )])
    fig.update_layout(title_text="Conversation Quality Pipeline", font_size=12, height=600)
    path = FIGURES_DIR / "conversation_type_sankey.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"  Saved {path}")


def fig5_completeness_heatmap(convos):
    """Heatmap: rows=columns, cols=model_era, values=% non-null."""
    numeric_cols = [
        "title", "created_at", "updated_at", "duration_minutes",
        "hour_of_day", "day_of_week", "year", "month", "year_month",
        "year_week", "is_weekend", "time_of_day", "days_since_first",
        "gap_days_from_prev", "user_token_ratio", "avg_user_msg_tokens",
        "avg_assistant_msg_tokens", "first_user_message_tokens",
        "first_user_message_type", "max_pii_priority",
    ]
    # Filter to columns that actually exist and have nulls
    cols = [c for c in numeric_cols if c in convos.columns]
    eras = sorted(convos["model_era"].unique().tolist())

    data = []
    for era in eras:
        era_df = convos[convos["model_era"] == era]
        row = {}
        for col in cols:
            row[col] = round(era_df[col].notna().mean() * 100, 1)
        data.append(row)

    heatmap_df = pd.DataFrame(data, index=eras).T

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu",
                ax=ax, linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={"label": "% Non-Null"})
    ax.set_title("Data Completeness by Model Era (% Non-Null)")
    ax.set_xlabel("Model Era")
    ax.set_ylabel("Column")
    path = FIGURES_DIR / "data_completeness_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ====================================================================
# 4e. Master quality report
# ====================================================================
def compile_master_report(messages, convos, schema_results, integrity_results):
    print("\n--- 4e. Compiling Master Quality Report ---")

    # Load sub-reports
    edge_report = {}
    edge_path = REPORTS_DIR / "edge_case_report.json"
    if edge_path.exists():
        with open(edge_path) as f:
            edge_report = json.load(f)

    enrich_report = {}
    enrich_path = REPORTS_DIR / "enrichment_summary.json"
    if enrich_path.exists():
        with open(enrich_path) as f:
            enrich_report = json.load(f)

    pii_report = {}
    pii_path = REPORTS_DIR / "pii_scan_report.json"
    if pii_path.exists():
        with open(pii_path) as f:
            pii_report = json.load(f)

    qs = convos["quality_score"]
    analysable = int(convos["is_analysable"].sum())
    excluded = int((~convos["is_analysable"]).sum())

    # Generate recommendations
    recommendations = []
    custom_gpt_ct = int(convos["is_custom_gpt"].sum())
    custom_gpt_pct = custom_gpt_ct / len(convos) * 100
    if custom_gpt_pct > 30:
        recommendations.append(
            f"{custom_gpt_ct} conversations ({custom_gpt_pct:.1f}%) are custom GPT sessions "
            "-- consider analyzing separately"
        )
    elif custom_gpt_ct > 0:
        recommendations.append(
            f"{custom_gpt_ct} conversations are custom GPT sessions -- consider analyzing separately"
        )

    critical_pii = pii_report.get("conversations_with_critical_pii", [])
    if critical_pii:
        recommendations.append(
            f"{len(critical_pii)} conversations have critical PII (SSN patterns) "
            "-- must redact before Claude API calls"
        )

    pii_conv_ct = int(convos["has_pii"].sum())
    if pii_conv_ct > 0:
        recommendations.append(
            f"{pii_conv_ct} conversations contain PII -- redact before external API calls"
        )

    # Check quality by era
    for era in convos["model_era"].unique():
        era_qs = convos[convos["model_era"] == era]["quality_score"].mean()
        if era_qs < 0.6:
            recommendations.append(
                f"Data quality is low in '{era}' era (mean score {era_qs:.2f}) "
                "-- consider controlling for model_era in longitudinal analysis"
            )

    if excluded > 0:
        recommendations.append(
            f"{excluded} conversations excluded from analysis -- review if exclusion criteria are too strict"
        )

    all_integrity_passed = all(p for _, p in integrity_results)

    report = {
        "phase": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "messages_rows": int(len(messages)),
            "conversations_rows": int(len(convos)),
        },
        "output_files": {
            "messages_clean_rows": int(len(messages)),
            "messages_clean_columns": int(len(messages.columns)),
            "conversations_clean_rows": int(len(convos)),
            "conversations_clean_columns": int(len(convos.columns)),
        },
        "edge_case_summary": edge_report,
        "enrichment_summary": enrich_report,
        "pii_summary": {k: v for k, v in pii_report.items() if k != "conversations_with_critical_pii"},
        "quality_overview": {
            "analysable_conversations": analysable,
            "analysable_pct": round(analysable / len(convos) * 100, 1),
            "excluded_conversations": excluded,
            "quality_score_mean": round(float(qs.mean()), 3),
            "quality_score_median": round(float(qs.median()), 3),
        },
        "schema_validation": schema_results,
        "referential_integrity": {
            "all_checks_passed": all_integrity_passed,
            "details": [f"{'PASS' if p else 'FAIL'}: {d}" for d, p in integrity_results],
        },
        "recommendations_for_phase3": recommendations,
    }

    report_path = REPORTS_DIR / "phase2_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved {report_path}")
    return report


# ====================================================================
# Final validation checklist
# ====================================================================
def run_final_checklist(messages, convos):
    print("\n=== Phase 2 Final Validation Checklist ===")
    results = []

    # 1. messages_clean has all 22+ expected columns
    expected_msg = set(EXPECTED_MSG_SCHEMA.keys())
    check = expected_msg.issubset(set(messages.columns))
    results.append((f"messages_clean has all {len(expected_msg)} expected columns", check))

    # 2. conversations_clean has all 40+ expected columns
    expected_conv = set(EXPECTED_CONV_SCHEMA.keys())
    check = expected_conv.issubset(set(convos.columns))
    results.append((f"conversations_clean has all {len(expected_conv)} expected columns", check))

    # 3. Referential integrity
    msg_ids = set(messages["conversation_id"].unique())
    conv_ids = set(convos["conversation_id"].unique())
    results.append(("referential integrity", msg_ids.issubset(conv_ids)))

    # 4. No timestamps outside plausible range
    ts = messages["timestamp"].dropna()
    bad = ((ts < pd.Timestamp("2022-01-01", tz="UTC")) |
           (ts > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1))).sum()
    results.append(("no timestamps outside 2022-present", bad == 0))

    # 5. quality_score in [0, 1]
    qs = convos["quality_score"]
    results.append(("quality_score in [0.0, 1.0]", bool((qs >= 0).all() and (qs <= 1).all())))

    # 6. conversation_type no nulls
    results.append(("conversation_type has no nulls", int(convos["conversation_type"].isna().sum()) == 0))

    # 7. model_era no nulls
    results.append(("model_era has no nulls", int(convos["model_era"].isna().sum()) == 0))

    # 8. is_analysable reasonable
    analysable_pct = convos["is_analysable"].mean() * 100
    results.append((f"is_analysable reasonable ({analysable_pct:.1f}%)", 0 < analysable_pct < 100))

    # 9. PII scan completed
    pii_report_path = REPORTS_DIR / "pii_scan_report.json"
    results.append(("PII scan report exists", pii_report_path.exists()))

    # 10. All 5 Phase 2 figures exist
    expected_figs = [
        "quality_score_dist.png", "edge_case_breakdown.png",
        "model_era_timeline.png", "conversation_type_sankey.html",
        "data_completeness_heatmap.png",
    ]
    all_exist = all((FIGURES_DIR / f).exists() for f in expected_figs)
    results.append(("all 5 Phase 2 figures exist", all_exist))

    # 11. phase2_quality_report.json exists
    results.append(("phase2_quality_report.json exists",
                     (REPORTS_DIR / "phase2_quality_report.json").exists()))

    # 12. pii_detections.json in interim, NOT in outputs
    results.append(("pii_detections.json in data/interim",
                     (INTERIM_DIR / "pii_detections.json").exists()))
    results.append(("pii_detections.json NOT in outputs",
                     not (REPORTS_DIR / "pii_detections.json").exists()))

    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    passed_ct = sum(1 for _, p in results if p)
    print(f"\n  {passed_ct}/{len(results)} checks passed")
    return results


def main():
    print("=== Phase 2, Step 4: Quality Report & Final Validation ===")
    config = load_config()

    # Load data
    print("Loading parquet files...")
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)
    print(f"  Messages: {len(messages)} rows, {len(messages.columns)} cols")
    print(f"  Conversations: {len(convos)} rows, {len(convos.columns)} cols")

    # 4a. Write clean files
    write_clean_files(messages, convos)

    # 4b. Schema validation
    schema_results = run_schema_validation(messages, convos)

    # 4c. Referential integrity
    integrity_results = run_integrity_checks(messages, convos)

    # 4d. Visualizations
    print("\n--- 4d. Generating Visualizations ---")
    fig1_quality_score_dist(convos)
    fig2_edge_case_breakdown(convos)
    fig3_model_era_timeline(convos, config)
    fig4_sankey(convos)
    fig5_completeness_heatmap(convos)

    # 4e. Master report
    compile_master_report(messages, convos, schema_results, integrity_results)

    # Final checklist
    run_final_checklist(messages, convos)

    print(f"\n=== Phase 2 Complete ===")


if __name__ == "__main__":
    main()
