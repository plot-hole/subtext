"""
Step 4: Phase 1 Data Inventory Report — Corpus statistics + 10 visualizations + validation checklist.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"

# Style
PRIMARY = "#2E75B6"
SECONDARY = "#666666"
ACCENT = "#C55A11"
PALETTE = [PRIMARY, ACCENT, SECONDARY, "#4CAF50"]
DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10

LOCAL_TZ = "America/Chicago"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": DPI,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
})


def load_data():
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)
    return messages, convos


def compute_inventory(messages, convos):
    """Compute all Phase 1 inventory statistics."""
    user_msgs = messages[messages["role"] == "user"]
    asst_msgs = messages[messages["role"] == "assistant"]

    date_min = convos["created_at"].min()
    date_max = convos["created_at"].max()
    date_range_days = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0

    # Conversations per month
    convos_with_date = convos[convos["created_at"].notna()].copy()
    convos_with_date["month"] = convos_with_date["created_at"].dt.to_period("M").astype(str)
    cpm = convos_with_date.groupby("month").size().to_dict()

    quality_dist = convos["quality_flag"].value_counts().to_dict() if "quality_flag" in convos.columns else {}

    inventory = {
        "total_conversations": int(len(convos)),
        "total_messages": int(len(messages)),
        "total_user_messages": int(len(user_msgs)),
        "total_assistant_messages": int(len(asst_msgs)),
        "total_user_tokens": int(user_msgs["token_count"].sum()),
        "total_assistant_tokens": int(asst_msgs["token_count"].sum()),
        "date_range_start": date_min.isoformat() if pd.notna(date_min) else None,
        "date_range_end": date_max.isoformat() if pd.notna(date_max) else None,
        "date_range_days": int(date_range_days),
        "avg_messages_per_conversation": round(convos["msg_count"].mean(), 1),
        "median_messages_per_conversation": float(convos["msg_count"].median()),
        "avg_user_tokens_per_message": round(user_msgs["token_count"].mean(), 1) if len(user_msgs) > 0 else 0,
        "avg_assistant_tokens_per_message": round(asst_msgs["token_count"].mean(), 1) if len(asst_msgs) > 0 else 0,
        "conversations_with_code_pct": round(convos["has_code"].mean() * 100, 1),
        "branched_conversations_pct": round(convos["is_branched"].mean() * 100, 1),
        "quality_flag_distribution": quality_dist,
        "conversations_per_month": cpm,
    }
    return inventory


# ======== Visualization functions ========

def fig1_temporal_heatmap(convos):
    """Heatmap: day-of-week x hour-of-day."""
    df = convos.dropna(subset=["hour_of_day", "day_of_week"]).copy()
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    pivot = df.groupby(["day_of_week", "hour_of_day"]).size().unstack(fill_value=0)
    # Ensure all hours/days present
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt="d",
                xticklabels=range(24), yticklabels=day_labels,
                linewidths=0.5, cbar_kws={"label": "Conversations"})
    ax.set_title("Conversation Start Times: Day of Week × Hour of Day (US/Central)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    path = FIGURES_DIR / "temporal_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_daily_volume(convos):
    """Plotly: daily conversation count + 7-day rolling avg + cumulative."""
    df = convos.dropna(subset=["created_at"]).copy()
    df["date"] = df["created_at"].dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["rolling_7d"] = daily["count"].rolling(7, min_periods=1).mean()
    daily["cumulative"] = daily["count"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=daily["date"], y=daily["count"], name="Daily Count",
               marker_color=PRIMARY, opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=daily["date"], y=daily["rolling_7d"], name="7-Day Avg",
                   line=dict(color=ACCENT, width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=daily["date"], y=daily["cumulative"], name="Cumulative",
                   line=dict(color=SECONDARY, width=2, dash="dot")),
        secondary_y=True,
    )
    fig.update_layout(
        title="Daily Conversation Volume",
        xaxis_title="Date",
        height=500,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Daily Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Total", secondary_y=True)
    path = FIGURES_DIR / "daily_volume.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"  Saved {path}")


def fig3_monthly_volume(convos):
    """Bar chart: conversations per month, colored by year."""
    df = convos.dropna(subset=["created_at"]).copy()
    df["year"] = df["created_at"].dt.year.astype(str)
    df["month"] = df["created_at"].dt.to_period("M")
    monthly = df.groupby(["month", "year"]).size().reset_index(name="count")
    monthly["month_str"] = monthly["month"].astype(str)
    monthly = monthly.sort_values("month")

    years = sorted(monthly["year"].unique())
    year_colors = {y: c for y, c in zip(years, PALETTE * 5)}

    fig, ax = plt.subplots(figsize=(14, 5))
    for year in years:
        subset = monthly[monthly["year"] == year]
        ax.bar(subset["month_str"], subset["count"], color=year_colors[year], label=year)
    ax.set_title("Conversations per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(title="Year")
    plt.xticks(rotation=45, ha="right")
    # Show every Nth label to avoid crowding
    labels = ax.get_xticklabels()
    if len(labels) > 24:
        for i, label in enumerate(labels):
            if i % 3 != 0:
                label.set_visible(False)
    path = FIGURES_DIR / "monthly_volume.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_conversation_length_dist(convos, messages):
    """Histograms: messages per conversation + user tokens per conversation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: messages per conversation
    mc = convos["msg_count"]
    clip = mc.quantile(0.99)
    mc_clipped = mc[mc <= clip]
    ax1.hist(mc_clipped, bins=50, color=PRIMARY, edgecolor="white", alpha=0.8)
    ax1.set_title("Messages per Conversation")
    ax1.set_xlabel("Message Count")
    ax1.set_ylabel("Frequency")
    if mc.skew() > 2:
        ax1.set_yscale("log")

    # Right: user tokens per conversation
    ut = convos["user_token_total"]
    clip2 = ut.quantile(0.99)
    ut_clipped = ut[ut <= clip2]
    ax2.hist(ut_clipped, bins=50, color=ACCENT, edgecolor="white", alpha=0.8)
    ax2.set_title("User Tokens per Conversation")
    ax2.set_xlabel("Token Count")
    ax2.set_ylabel("Frequency")
    if ut.skew() > 2:
        ax2.set_yscale("log")

    fig.suptitle("Conversation Length Distributions (clipped at 99th percentile)", fontsize=TITLE_SIZE)
    path = FIGURES_DIR / "conversation_length_dist.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_message_length_dist(messages):
    """Histograms: user vs assistant message token counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    user_tok = messages.loc[messages["role"] == "user", "token_count"].dropna()
    asst_tok = messages.loc[messages["role"] == "assistant", "token_count"].dropna()

    for ax, data, color, label in [
        (ax1, user_tok, PRIMARY, "User"),
        (ax2, asst_tok, ACCENT, "Assistant"),
    ]:
        clip = data.quantile(0.99) if len(data) > 0 else 1
        clipped = data[data <= clip]
        ax.hist(clipped, bins=50, color=color, edgecolor="white", alpha=0.8)
        mean_val = data.mean()
        med_val = data.median()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.0f}")
        ax.axvline(med_val, color="green", linestyle="-.", label=f"Median: {med_val:.0f}")
        ax.set_title(f"{label} Message Token Counts")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle("Message Length Distributions", fontsize=TITLE_SIZE)
    path = FIGURES_DIR / "message_length_dist.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig6_role_volume_over_time(messages):
    """Stacked area: monthly tokens by role."""
    df = messages[messages["role"].isin(["user", "assistant"])].copy()
    df = df.dropna(subset=["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby(["month", "role"])["token_count"].sum().unstack(fill_value=0)
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    if "user" in monthly.columns and "assistant" in monthly.columns:
        ax.fill_between(range(len(monthly)), monthly["user"], label="User", color=PRIMARY, alpha=0.7)
        ax.fill_between(range(len(monthly)), monthly["user"],
                        monthly["user"] + monthly["assistant"],
                        label="Assistant", color=ACCENT, alpha=0.7)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")
    if len(monthly) > 24:
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 3 != 0:
                label.set_visible(False)
    ax.set_title("Monthly Token Volume by Role")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Tokens")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    path = FIGURES_DIR / "role_volume_over_time.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig7_hourly_pattern(convos):
    """Bar chart: conversations by hour of day."""
    df = convos.dropna(subset=["hour_of_day"])
    hourly = df.groupby("hour_of_day").size()
    hourly = hourly.reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(hourly.index, hourly.values, color=PRIMARY, edgecolor="white")
    peak = hourly.idxmax()
    bars[peak].set_color(ACCENT)
    ax.annotate(f"Peak: {peak}:00 ({hourly[peak]} convos)",
                xy=(peak, hourly[peak]), xytext=(peak + 2, hourly[peak] * 1.1),
                arrowprops=dict(arrowstyle="->", color=ACCENT),
                fontsize=10, color=ACCENT)
    ax.set_title("Conversations by Hour of Day (US/Central)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.set_xticks(range(24))
    path = FIGURES_DIR / "hourly_pattern.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig8_weekly_pattern(convos):
    """Bar chart: conversations by day of week."""
    df = convos.dropna(subset=["day_of_week"])
    weekly = df.groupby("day_of_week").size()
    weekly = weekly.reindex(range(7), fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(7), weekly.values, color=PRIMARY, edgecolor="white", tick_label=day_labels)
    ax.set_title("Conversations by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    path = FIGURES_DIR / "weekly_pattern.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig9_duration_dist(convos):
    """Histogram of conversation duration in minutes."""
    dur = convos["duration_minutes"].dropna()
    dur = dur[dur >= 0]
    clip = dur.quantile(0.99) if len(dur) > 0 else 60
    dur_clipped = dur[dur <= clip]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dur_clipped, bins=50, color=PRIMARY, edgecolor="white", alpha=0.8)
    med = dur.median()
    ax.axvline(med, color=ACCENT, linestyle="--", linewidth=2, label=f"Median: {med:.1f} min")
    ax.set_title("Conversation Duration Distribution (clipped at 99th percentile)")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Frequency")
    ax.legend()
    path = FIGURES_DIR / "conversation_duration_dist.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig10_code_over_time(convos):
    """Monthly bar chart: % of conversations containing code."""
    df = convos.dropna(subset=["created_at"]).copy()
    df["month"] = df["created_at"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        total=("has_code", "size"),
        code_count=("has_code", "sum"),
    )
    monthly["pct"] = (monthly["code_count"] / monthly["total"] * 100).round(1)
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(monthly)), monthly["pct"], color=PRIMARY, edgecolor="white")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")
    if len(monthly) > 24:
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 3 != 0:
                label.set_visible(False)
    ax.set_title("Percentage of Conversations Containing Code Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("% with Code")
    path = FIGURES_DIR / "code_conversations_over_time.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ======== Validation Checklist ========

def run_validation(messages, convos):
    """Run all validation checks and print PASS/FAIL."""
    print("\n=== Validation Checklist ===")
    results = []

    # 1. messages.parquet loads with expected columns
    expected_msg_cols = {"conversation_id", "msg_index", "role", "text", "timestamp",
                         "token_count", "char_count", "word_count", "has_code",
                         "has_attachment", "is_branched"}
    check = expected_msg_cols.issubset(set(messages.columns))
    results.append(("messages.parquet has expected columns", check))

    # 2. conversations.parquet loads with expected columns
    expected_conv_cols = {"conversation_id", "title", "created_at", "updated_at",
                          "duration_minutes", "msg_count", "user_msg_count",
                          "assistant_msg_count", "user_token_total", "assistant_token_total",
                          "hour_of_day", "day_of_week", "has_code", "is_branched"}
    check = expected_conv_cols.issubset(set(convos.columns))
    results.append(("conversations.parquet has expected columns", check))

    # 3. Referential integrity
    msg_conv_ids = set(messages["conversation_id"].unique())
    conv_ids = set(convos["conversation_id"].unique())
    check = msg_conv_ids.issubset(conv_ids)
    results.append(("Referential integrity (msg conv_ids subset of conv conv_ids)", check))

    # 4. No duplicate (conversation_id, msg_index)
    dup_count = messages.duplicated(subset=["conversation_id", "msg_index"]).sum()
    results.append(("No duplicate (conversation_id, msg_index)", dup_count == 0))

    # 5. msg_count matches actual
    actual_counts = messages.groupby("conversation_id").size()
    merged = convos.set_index("conversation_id")["msg_count"]
    matched = merged.reindex(actual_counts.index)
    check = (matched == actual_counts).all() if len(matched) > 0 else True
    results.append(("msg_count matches actual message count", check))

    # 6. Date range plausible
    if convos["created_at"].notna().any():
        min_date = convos["created_at"].min()
        max_date = convos["created_at"].max()
        check = (min_date.year >= 2020 and max_date.year <= 2027)
        results.append((f"Date range plausible ({min_date.date()} to {max_date.date()})", check))
    else:
        results.append(("Date range plausible", False))

    # 7. All 10 figures exist
    expected_figs = [
        "temporal_heatmap.png", "daily_volume.html", "monthly_volume.png",
        "conversation_length_dist.png", "message_length_dist.png",
        "role_volume_over_time.png", "hourly_pattern.png", "weekly_pattern.png",
        "conversation_duration_dist.png", "code_conversations_over_time.png",
    ]
    all_exist = all((FIGURES_DIR / f).exists() for f in expected_figs)
    results.append(("All 10 figures exist", all_exist))

    # 8. phase1_inventory.json exists
    check = (REPORTS_DIR / "phase1_inventory.json").exists()
    results.append(("phase1_inventory.json exists", check))

    # 9. parse_errors.json exists
    check = (REPORTS_DIR / "parse_errors.json").exists()
    results.append(("parse_errors.json exists", check))

    # 10. cleaning_summary.json exists
    check = (REPORTS_DIR / "cleaning_summary.json").exists()
    results.append(("cleaning_summary.json exists", check))

    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{len(results)} checks passed")
    return results


def main():
    print("=== Step 4: Phase 1 Data Inventory Report ===")

    messages, convos = load_data()
    print(f"Loaded {len(messages)} messages, {len(convos)} conversations")

    # 4a. Compute inventory
    print("\n--- 4a. Corpus Overview Statistics ---")
    inventory = compute_inventory(messages, convos)
    inv_path = REPORTS_DIR / "phase1_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"  Saved {inv_path}")

    # Print key stats
    print(f"\n  Total conversations:             {inventory['total_conversations']:,}")
    print(f"  Total messages:                  {inventory['total_messages']:,}")
    print(f"  Total user messages:             {inventory['total_user_messages']:,}")
    print(f"  Total assistant messages:        {inventory['total_assistant_messages']:,}")
    print(f"  Total user tokens:               {inventory['total_user_tokens']:,}")
    print(f"  Total assistant tokens:          {inventory['total_assistant_tokens']:,}")
    print(f"  Date range:                      {inventory['date_range_start']} to {inventory['date_range_end']}")
    print(f"  Date range (days):               {inventory['date_range_days']}")
    print(f"  Avg messages/conversation:       {inventory['avg_messages_per_conversation']}")
    print(f"  Median messages/conversation:    {inventory['median_messages_per_conversation']}")
    print(f"  Avg user tokens/message:         {inventory['avg_user_tokens_per_message']}")
    print(f"  Avg assistant tokens/message:    {inventory['avg_assistant_tokens_per_message']}")
    print(f"  Conversations with code:         {inventory['conversations_with_code_pct']}%")
    print(f"  Branched conversations:          {inventory['branched_conversations_pct']}%")
    print(f"  Quality flags:                   {inventory['quality_flag_distribution']}")

    # 4b. Visualizations
    print("\n--- 4b. Generating Visualizations ---")
    fig1_temporal_heatmap(convos)
    fig2_daily_volume(convos)
    fig3_monthly_volume(convos)
    fig4_conversation_length_dist(convos, messages)
    fig5_message_length_dist(messages)
    fig6_role_volume_over_time(messages)
    fig7_hourly_pattern(convos)
    fig8_weekly_pattern(convos)
    fig9_duration_dist(convos)
    fig10_code_over_time(convos)

    # Validation
    run_validation(messages, convos)

    print(f"\n=== Step 4 Complete ===")


if __name__ == "__main__":
    main()
