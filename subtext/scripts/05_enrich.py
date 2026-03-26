"""
Phase 2, Step 2: Metadata Enrichment.
Adds derived columns for temporal analysis, model eras, quality scores, etc.
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "quality_config.json"

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2d. First message opening classification
# ============================================================
IMPERATIVE_VERBS = {
    "write", "create", "make", "generate", "explain", "tell", "show",
    "list", "give", "find", "help", "summarize", "translate", "analyze",
    "compare", "design", "build", "fix", "debug", "convert", "calculate",
    "describe", "provide", "suggest", "recommend", "outline", "draft",
    "rewrite", "edit", "revise", "update", "add", "remove", "check",
    "define", "implement", "develop", "plan", "organize", "simplify",
    "elaborate", "clarify", "format", "sort", "extract", "review",
}

QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how",
    "is", "are", "can", "do", "does", "will", "would", "could", "should",
    "did", "has", "have", "was", "were",
}

GREETING_STARTERS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


def classify_first_message(text):
    if not text or pd.isna(text):
        return "fragment"
    text = text.strip()
    if not text:
        return "fragment"

    # Check code first
    if text.startswith("```"):
        return "code"
    from scripts_helpers import detect_code_fraction
    # Inline code fraction check
    import re
    code_matches = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(m) for m in code_matches)
    if len(text) > 0 and code_chars / len(text) > 0.5:
        return "code"

    # Question
    if text.rstrip().endswith("?"):
        return "question"
    first_word = text.split()[0].lower().rstrip("?.,!:;") if text.split() else ""
    if first_word in QUESTION_STARTERS:
        return "question"

    # Command
    if first_word in IMPERATIVE_VERBS:
        return "command"

    # Greeting
    text_lower_start = text[:20].lower()
    for g in GREETING_STARTERS:
        if text_lower_start.startswith(g):
            return "greeting"

    # Statement
    if len(text.split()) > 10:
        return "statement"

    return "fragment"


# Remove the import dependency - inline the code fraction function
def _detect_code_fraction(text):
    import re
    if not text:
        return 0.0
    matches = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(m) for m in matches)
    return code_chars / len(text) if len(text) > 0 else 0.0


def classify_first_message_v2(text):
    """Classify the first user message opening type."""
    if not text or pd.isna(text):
        return "fragment"
    text = text.strip()
    if not text:
        return "fragment"

    import re

    # Code
    if text.startswith("```"):
        return "code"
    code_frac = _detect_code_fraction(text)
    if code_frac > 0.5:
        return "code"

    # Question
    if text.rstrip().endswith("?"):
        return "question"
    words = text.split()
    first_word = words[0].lower().rstrip("?.,!:;") if words else ""
    if first_word in QUESTION_STARTERS:
        return "question"

    # Command
    if first_word in IMPERATIVE_VERBS:
        return "command"

    # Greeting
    text_lower_start = text[:20].lower()
    for g in GREETING_STARTERS:
        if text_lower_start.startswith(g):
            return "greeting"

    # Statement
    if len(words) > 10:
        return "statement"

    return "fragment"


def main():
    print("=== Phase 2, Step 2: Metadata Enrichment ===")

    config = load_config()
    tz = config["timezone"]
    era_bounds = config["model_era_boundaries"]

    # Load data
    print("Loading parquet files...")
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)
    print(f"  Messages: {len(messages)}, Conversations: {len(convos)}")

    # ================================================================
    # 2a. Model era tagging
    # ================================================================
    print("\n--- 2a. Model Era Tagging ---")
    print("  NOTE: Model era is a temporal proxy, not a true model identifier.")
    print("  The ChatGPT export does not reliably include which model was used.")

    era_dates = {
        "gpt4_launch": pd.Timestamp(era_bounds["gpt4_launch"], tz="UTC"),
        "gpt4_turbo": pd.Timestamp(era_bounds["gpt4_turbo_launch"], tz="UTC"),
        "gpt4o": pd.Timestamp(era_bounds["gpt4o_launch"], tz="UTC"),
        "o1": pd.Timestamp(era_bounds["o1_launch"], tz="UTC"),
        "o3_mini": pd.Timestamp(era_bounds["o3_mini_launch"], tz="UTC"),
        "gpt4_1": pd.Timestamp(era_bounds["gpt4_1_launch"], tz="UTC"),
        "gpt5": pd.Timestamp(era_bounds["gpt5_launch"], tz="UTC"),
        "gpt5_2": pd.Timestamp(era_bounds["gpt5_2_launch"], tz="UTC"),
        "gpt5_4": pd.Timestamp(era_bounds["gpt5_4_launch"], tz="UTC"),
    }

    def assign_era(ts):
        if pd.isna(ts):
            return "unknown"
        if ts < era_dates["gpt4_launch"]:
            return "gpt-3.5"
        elif ts < era_dates["gpt4_turbo"]:
            return "gpt-4-early"
        elif ts < era_dates["gpt4o"]:
            return "gpt-4-turbo"
        elif ts < era_dates["o1"]:
            return "gpt-4o"
        elif ts < era_dates["o3_mini"]:
            return "o1-era"
        elif ts < era_dates["gpt4_1"]:
            return "gpt-4o"
        elif ts < era_dates["gpt5"]:
            return "gpt-4.1"
        elif ts < era_dates["gpt5_2"]:
            return "gpt-5"
        elif ts < era_dates["gpt5_4"]:
            return "gpt-5.2"
        else:
            return "gpt-5.4"

    convos["model_era"] = convos["created_at"].apply(assign_era).astype("category")
    print(f"  Model era distribution: {convos['model_era'].value_counts().to_dict()}")

    # ================================================================
    # 2b. Temporal enrichment — conversations
    # ================================================================
    print("\n--- 2b. Temporal Enrichment ---")

    convos["year"] = convos["created_at"].dt.year.astype("Int16")
    convos["month"] = convos["created_at"].dt.month.astype("Int8")
    convos["year_month"] = convos["created_at"].dt.strftime("%Y-%m")
    convos["year_week"] = convos["created_at"].dt.strftime("%Y-W%V")
    convos["is_weekend"] = convos["day_of_week"].apply(
        lambda x: x >= 5 if pd.notna(x) else pd.NA
    ).astype("boolean")

    def time_of_day(h):
        if pd.isna(h):
            return pd.NA
        h = int(h)
        if h < 6:
            return "night"
        elif h < 12:
            return "morning"
        elif h < 18:
            return "afternoon"
        else:
            return "evening"

    convos["time_of_day"] = convos["hour_of_day"].apply(time_of_day).astype("category")

    # days_since_first
    earliest = convos["created_at"].min()
    convos["days_since_first"] = convos["created_at"].apply(
        lambda x: (x - earliest).days if pd.notna(x) and pd.notna(earliest) else pd.NA
    ).astype("Int32")

    # gap_days_from_prev
    convos_sorted = convos.sort_values("created_at").reset_index(drop=True)
    convos_sorted["gap_days_from_prev"] = (
        convos_sorted["created_at"].diff().dt.total_seconds() / 86400.0
    ).astype("float32")
    # Map back to original index
    convos = convos_sorted

    print(f"  Added: year, month, year_month, year_week, is_weekend, time_of_day, days_since_first, gap_days_from_prev")

    # ================================================================
    # 2b. Temporal enrichment — messages
    # ================================================================
    print("\n--- 2b. Message-level Temporal Enrichment ---")
    messages = messages.sort_values(["conversation_id", "msg_index"]).reset_index(drop=True)

    # is_first_user_msg
    user_msgs = messages[messages["role"] == "user"]
    first_user_idx = user_msgs.groupby("conversation_id")["msg_index"].min()
    first_user_set = set(
        zip(first_user_idx.index, first_user_idx.values)
    )
    messages["is_first_user_msg"] = messages.apply(
        lambda r: (r["conversation_id"], r["msg_index"]) in first_user_set, axis=1
    )

    # is_last_msg
    last_idx = messages.groupby("conversation_id")["msg_index"].max()
    last_set = set(zip(last_idx.index, last_idx.values))
    messages["is_last_msg"] = messages.apply(
        lambda r: (r["conversation_id"], r["msg_index"]) in last_set, axis=1
    )

    # position_in_conversation
    max_idx = messages.groupby("conversation_id")["msg_index"].transform("max")
    messages["position_in_conversation"] = np.where(
        max_idx > 0,
        (messages["msg_index"] / max_idx).astype("float32"),
        0.0
    ).astype("float32")

    # inter_msg_seconds
    messages["_prev_ts"] = messages.groupby("conversation_id")["timestamp"].shift(1)
    messages["inter_msg_seconds"] = (
        (messages["timestamp"] - messages["_prev_ts"]).dt.total_seconds()
    ).astype("float32")
    messages.drop(columns=["_prev_ts"], inplace=True)

    # cumulative_user_tokens
    messages["_is_user"] = messages["role"] == "user"
    messages["_user_tok"] = np.where(messages["_is_user"], messages["token_count"].fillna(0), 0)
    messages["cumulative_user_tokens"] = messages.groupby("conversation_id")["_user_tok"].cumsum().astype("int32")
    messages.drop(columns=["_is_user", "_user_tok"], inplace=True)

    print(f"  Added: is_first_user_msg, is_last_msg, position_in_conversation, inter_msg_seconds, cumulative_user_tokens")

    # ================================================================
    # 2c. Conversation-level derived metrics
    # ================================================================
    print("\n--- 2c. Conversation-level Derived Metrics ---")

    # user_token_ratio
    total_tok = convos["user_token_total"] + convos["assistant_token_total"]
    convos["user_token_ratio"] = np.where(
        total_tok > 0,
        convos["user_token_total"] / total_tok,
        np.nan
    ).astype("float32")

    # avg_user_msg_tokens
    convos["avg_user_msg_tokens"] = np.where(
        convos["user_msg_count"] > 0,
        convos["user_token_total"] / convos["user_msg_count"],
        np.nan
    ).astype("float32")

    # avg_assistant_msg_tokens
    convos["avg_assistant_msg_tokens"] = np.where(
        convos["assistant_msg_count"] > 0,
        convos["assistant_token_total"] / convos["assistant_msg_count"],
        np.nan
    ).astype("float32")

    # turns: count role switches
    def count_turns(conv_id):
        grp = messages[messages["conversation_id"] == conv_id].sort_values("msg_index")
        roles = grp[grp["role"].isin(["user", "assistant"])]["role"].values
        if len(roles) < 2:
            return 0
        return sum(1 for i in range(1, len(roles)) if roles[i] != roles[i-1])

    print("  Computing turn counts...")
    turn_counts = {}
    for conv_id in tqdm(convos["conversation_id"].values, desc="  Counting turns"):
        turn_counts[conv_id] = count_turns(conv_id)
    convos["turns"] = convos["conversation_id"].map(turn_counts).astype("int32")

    # first_user_message_tokens
    first_user_msgs = messages[messages["is_first_user_msg"]].set_index("conversation_id")
    convos["first_user_message_tokens"] = convos["conversation_id"].map(
        first_user_msgs["token_count"]
    ).astype("Int32")

    # ================================================================
    # 2d. First message opening classification
    # ================================================================
    print("\n--- 2d. First Message Opening Classification ---")

    first_user_texts = first_user_msgs["text"]
    convos["first_user_message_type"] = convos["conversation_id"].map(
        first_user_texts.apply(classify_first_message_v2)
    ).fillna("fragment").astype("category")

    fmt_dist = convos["first_user_message_type"].value_counts().to_dict()
    print(f"  First message type distribution: {fmt_dist}")

    # ================================================================
    # 2e. Quality score
    # ================================================================
    print("\n--- 2e. Quality Score ---")

    weights = config["quality_score_weights"]
    min_msgs = config["min_messages_for_analysis"]

    # Pre-compute per-conversation quality metrics from messages
    msg_quality = messages.groupby("conversation_id").agg(
        timestamp_completeness=("timestamp", lambda s: float(s.notna().mean())),
        text_completeness=("text", lambda s: float((s.notna() & (s.str.strip() != "")).mean())),
    ).reset_index()

    # first/last msg has timestamp
    first_msgs = messages.groupby("conversation_id").first()
    last_msgs = messages.groupby("conversation_id").last()
    msg_quality["first_msg_has_ts"] = msg_quality["conversation_id"].map(
        first_msgs["timestamp"].notna()
    ).fillna(False)
    msg_quality["last_msg_has_ts"] = msg_quality["conversation_id"].map(
        last_msgs["timestamp"].notna()
    ).fillna(False)

    # Merge into convos
    convos = convos.merge(msg_quality, on="conversation_id", how="left")

    # Compute score
    def compute_quality_score(row):
        score = 0.0
        score += weights["has_timestamps"] * (row.get("timestamp_completeness", 0.0) or 0.0)
        score += weights["has_text"] * (row.get("text_completeness", 0.0) or 0.0)
        score += weights["not_empty"] * (0.0 if row.get("is_empty", True) else 1.0)
        score += weights["reasonable_length"] * (
            1.0 if row.get("msg_count", 0) >= min_msgs else 0.0
        )
        fts = row.get("first_msg_has_ts", False)
        lts = row.get("last_msg_has_ts", False)
        score += weights["not_truncated"] * (1.0 if (fts and lts) else 0.0)
        return round(score, 3)

    convos["quality_score"] = convos.apply(compute_quality_score, axis=1).astype("float32")

    # Drop helper columns
    convos.drop(columns=["timestamp_completeness", "text_completeness",
                          "first_msg_has_ts", "last_msg_has_ts"], inplace=True)

    qs = convos["quality_score"]
    print(f"  Quality score stats: mean={qs.mean():.3f}, median={qs.median():.3f}, "
          f"std={qs.std():.3f}, min={qs.min():.3f}, max={qs.max():.3f}")

    # Update quality_flag
    def assign_quality_flag(row):
        if row["is_empty"]:
            return "empty"
        if row["quality_score"] < 0.4:
            return "low_quality"
        if row["quality_score"] < 0.7:
            return "partial"
        return "complete"

    convos["quality_flag"] = convos.apply(assign_quality_flag, axis=1).astype("category")
    qf_dist = convos["quality_flag"].value_counts().to_dict()
    print(f"  Quality flag distribution: {qf_dist}")

    # ================================================================
    # 2f. Analysability flag
    # ================================================================
    print("\n--- 2f. Analysability Flag ---")

    convos["is_analysable"] = (
        (~convos["is_empty"]) &
        (convos["quality_score"] >= 0.4) &
        (convos["msg_count"] >= min_msgs)
    )

    analysable = convos["is_analysable"].sum()
    excluded = len(convos) - analysable
    print(f"  Analysable: {analysable} ({analysable/len(convos)*100:.1f}%)")
    print(f"  Excluded:   {excluded} ({excluded/len(convos)*100:.1f}%)")

    # Exclusion reasons
    empty_ct = int(convos["is_empty"].sum())
    low_q = int(((~convos["is_empty"]) & (convos["quality_score"] < 0.4)).sum())
    too_few_mask = (
        (~convos["is_empty"]) &
        (convos["quality_score"] >= 0.4) &
        (convos["msg_count"] < min_msgs)
    )
    too_few = int(too_few_mask.sum())
    print(f"  Exclusion reasons: empty={empty_ct}, low_quality={low_q}, too_few_messages={too_few}")

    # ================================================================
    # Save
    # ================================================================
    print("\n--- Saving ---")
    messages.to_parquet(MSG_PATH, index=False)
    convos.to_parquet(CONV_PATH, index=False)
    print(f"  Saved {MSG_PATH}")
    print(f"  Saved {CONV_PATH}")

    # Enrichment summary
    summary = {
        "columns_added": {
            "messages": [
                "is_first_user_msg", "is_last_msg", "position_in_conversation",
                "inter_msg_seconds", "cumulative_user_tokens"
            ],
            "conversations": [
                "model_era", "year", "month", "year_month", "year_week",
                "is_weekend", "time_of_day", "days_since_first", "gap_days_from_prev",
                "user_token_ratio", "avg_user_msg_tokens", "avg_assistant_msg_tokens",
                "turns", "first_user_message_tokens", "first_user_message_type",
                "quality_score", "quality_flag", "is_analysable"
            ],
        },
        "model_era_distribution": {k: int(v) for k, v in convos["model_era"].value_counts().to_dict().items()},
        "first_message_type_distribution": {k: int(v) for k, v in fmt_dist.items()},
        "quality_score_stats": {
            "mean": round(float(qs.mean()), 3),
            "median": round(float(qs.median()), 3),
            "std": round(float(qs.std()), 3),
            "min": round(float(qs.min()), 3),
            "max": round(float(qs.max()), 3),
        },
        "quality_flag_distribution": {k: int(v) for k, v in qf_dist.items()},
        "analysable_conversations": int(analysable),
        "excluded_conversations": int(excluded),
        "exclusion_reasons": {
            "empty": empty_ct,
            "low_quality_score": low_q,
            "too_few_messages": too_few,
        },
    }
    summary_path = REPORTS_DIR / "enrichment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    print(f"\n=== Step 2 Complete ===")


if __name__ == "__main__":
    main()
