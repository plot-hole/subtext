"""
Module 3.2f: Vocabulary Transfer
Script: 18_vocab_transfer.py

Identifies words and phrases that originate in AI responses and subsequently
appear in the user's own messages — with a temporal gap. This measures
permanent AI influence on user cognition: not whether the user adopted the
AI's frame in the moment (Module 3.2e), but whether the AI permanently
changed how the user thinks and speaks.

No API calls required — pure corpus-level NLP using spaCy and wordfreq.

Usage:
    python scripts/18_vocab_transfer.py

    # Adjust gap threshold (default 24 hours):
    python scripts/18_vocab_transfer.py --gap-hours 48

    # Adjust minimum post-transfer count:
    python scripts/18_vocab_transfer.py --min-count 3

    # Adjust common word filter (top N words excluded):
    python scripts/18_vocab_transfer.py --freq-threshold 5000

    # Dry run (compute stats, don't save):
    python scripts/18_vocab_transfer.py --dry-run
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
import plotly.graph_objects as go
from tqdm import tqdm

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH       = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")

OUT_PARQUET     = os.path.join(BASE, "data", "processed", "vocab_transfer.parquet")
OUT_TIMELINE    = os.path.join(BASE, "data", "processed", "vocab_user_timeline.parquet")
OUT_REPORT      = os.path.join(BASE, "outputs", "reports", "vocab_transfer_report.json")
FIG_DIR         = os.path.join(BASE, "outputs", "figures", "vocab_transfer")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "processed"), exist_ok=True)

# -- Style constants (match existing scripts) --------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
DPI = 150

STICKINESS_COLORS = {
    "one_time": "#BAB0AC",
    "repeated": "#F28E2B",
    "sticky":   "#E15759",
}

FUNC_COLORS = {
    "interpersonal_analysis": "#4E79A7", "emotional_processing": "#F28E2B",
    "creative_expression":    "#E15759", "career_strategy":      "#76B7B2",
    "self_modeling":          "#59A14F", "practical":             "#EDC948",
    "learning":               "#B07AA1", "problem_solving":      "#FF9DA7",
    "coding":                 "#9C755F", "social_rehearsal":     "#BAB0AC",
    "work_professional":      "#86BCB6", "planning":             "#F1CE63",
}

EMOTION_COLORS = {
    "analytical":  "#4E79A7", "curious":     "#A0CBE8", "reflective":  "#76B7B2",
    "energized":   "#F28E2B", "playful":     "#FFBE7D", "anxious":     "#E15759",
    "frustrated":  "#FF9DA7", "grieving":    "#B07AA1", "vulnerable":  "#D4A6C8",
    "numb":        "#BAB0AC", "strategic":   "#59A14F", "determined":  "#8CD17D",
}


# -- Helper functions --------------------------------------------------------
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
    return os.path.join(FIG_DIR, name)


# ============================================================================
# TOKENISATION
# ============================================================================
def load_spacy():
    """Load spaCy model."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # Increase max length for large texts
    nlp.max_length = 2_000_000
    return nlp


def load_common_words(freq_threshold):
    """Load top-N most common English words for filtering."""
    try:
        from wordfreq import top_n_list
        common = set(top_n_list("en", freq_threshold))
        print(f"  Loaded {len(common):,} common words from wordfreq")
        return common
    except ImportError:
        pass

    # Fallback: spaCy stopwords + manual list
    import spacy
    nlp = spacy.load("en_core_web_sm")
    common = set(nlp.Defaults.stop_words)
    # Add common analytical words that aren't stopwords
    extras = {
        "important", "actually", "basically", "generally", "specific",
        "different", "similar", "example", "think", "know", "want",
        "need", "like", "just", "really", "thing", "things", "way",
        "good", "bad", "new", "old", "time", "people", "person",
        "make", "made", "right", "sure", "going", "come", "take",
        "look", "point", "even", "also", "well", "lot", "much",
        "many", "part", "kind", "sort", "mean", "say", "said",
        "tell", "told", "ask", "asked", "yes", "yeah", "ok", "okay",
        "great", "thank", "thanks", "please", "sorry", "hello", "hi",
        "hey", "wow", "oh", "ah", "hmm", "huh", "interesting",
        "absolutely", "definitely", "exactly", "probably", "maybe",
        "perhaps", "certainly", "clearly", "obviously", "honestly",
        "pretty", "quite", "real", "true", "false", "fact", "idea",
        "question", "answer", "problem", "issue", "situation", "case",
        "work", "working", "help", "feel", "feeling", "sense",
        "understand", "understanding", "experience", "life", "day",
        "start", "end", "long", "big", "small", "high", "low",
    }
    common.update(extras)
    print(f"  Loaded {len(common):,} common words (spaCy stopwords + extras)")
    return common


def tokenize_message(doc, common_words):
    """
    Extract unigrams and bigrams from a spaCy Doc.
    Returns sets of (token_str, token_type).
    """
    tokens = set()
    # Unigrams: lemmatised, lowercase, alpha-only, not in common words
    lemmas = []
    for tok in doc:
        if (tok.is_alpha and len(tok.text) > 1
                and not tok.is_stop and not tok.is_punct
                and not tok.like_num):
            lemma = tok.lemma_.lower()
            if lemma not in common_words and len(lemma) > 1:
                tokens.add((lemma, "unigram"))
                lemmas.append((lemma, tok.i))
            else:
                lemmas.append((None, tok.i))
        else:
            lemmas.append((None, tok.i))

    # Bigrams from consecutive valid lemmas (within 1 token distance)
    valid_lemmas = [(lem, idx) for lem, idx in lemmas if lem is not None]
    for i in range(len(valid_lemmas) - 1):
        lem_a, idx_a = valid_lemmas[i]
        lem_b, idx_b = valid_lemmas[i + 1]
        if idx_b - idx_a <= 2:  # Allow one intervening token (stopword etc)
            bigram = f"{lem_a}_{lem_b}"
            tokens.add((bigram, "bigram"))

    return tokens


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Module 3.2f: Vocabulary Transfer")
    parser.add_argument("--gap-hours", type=float, default=24.0,
                        help="Minimum hours between AI intro and user adoption (default: 24)")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Minimum post-transfer user occurrences (default: 2)")
    parser.add_argument("--freq-threshold", type=int, default=10000,
                        help="Top N common English words to exclude (default: 10000)")
    parser.add_argument("--min-ai-uses", type=int, default=7,
                        help="Min AI uses before user's first use to count as transfer (default: 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute stats but don't save outputs")
    args = parser.parse_args()

    report = {
        "module": "vocab_transfer",
        "module_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figures_generated": [],
        "data_outputs": [],
        "warnings": [],
    }

    print("=" * 80)
    print("Module 3.2f: Vocabulary Transfer")
    print("=" * 80)
    print(f"  Gap threshold:      {args.gap_hours} hours")
    print(f"  Min post-transfer:  {args.min_count} uses")
    print(f"  Common word filter:  top {args.freq_threshold:,}")
    print(f"  Min AI uses:        {args.min_ai_uses}")
    print(f"  Dry run:            {args.dry_run}")
    print()

    # -----------------------------------------------------------------------
    # Step 0: Load data
    # -----------------------------------------------------------------------
    print("=== Step 0: Loading data ===")
    conv = pd.read_parquet(CONV_PATH)
    msgs = pd.read_parquet(MSGS_PATH)

    analysable_ids = set(conv.loc[conv["is_analysable"], "conversation_id"])
    msgs = msgs[msgs["conversation_id"].isin(analysable_ids)].copy()
    msgs = msgs[msgs["role"].isin(["user", "assistant"])].copy()
    msgs = msgs.dropna(subset=["text"])
    msgs = msgs[msgs["text"].str.strip().str.len() > 0].copy()
    msgs = msgs.sort_values(["conversation_id", "msg_index"]).reset_index(drop=True)

    # Load classifications for cross-analysis
    func_df = None
    emot_df = None
    if os.path.exists(FUNC_CLASS_PATH):
        func_df = pd.read_parquet(FUNC_CLASS_PATH)[["conversation_id", "function_primary"]]
        func_df = func_df.rename(columns={"function_primary": "functional_category"})
        print(f"  Loaded functional classifications: {len(func_df):,} conversations")
    if os.path.exists(EMOT_PATH):
        emot_df = pd.read_parquet(EMOT_PATH)[["conversation_id", "emotion_primary"]]
        emot_df = emot_df.rename(columns={"emotion_primary": "emotional_state"})
        print(f"  Loaded emotional states: {len(emot_df):,} conversations")

    user_msgs = msgs[msgs["role"] == "user"].copy()
    asst_msgs = msgs[msgs["role"] == "assistant"].copy()

    print(f"  Analysable conversations: {len(analysable_ids):,}")
    print(f"  User messages: {len(user_msgs):,}")
    print(f"  Assistant messages: {len(asst_msgs):,}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Load NLP tools
    # -----------------------------------------------------------------------
    print("=== Step 1: Loading NLP tools ===")
    nlp = load_spacy()
    common_words = load_common_words(args.freq_threshold)
    print()

    # -----------------------------------------------------------------------
    # Step 2: Build vocabulary timelines
    # -----------------------------------------------------------------------
    print("=== Step 2: Building vocabulary timelines ===")

    # For each token, track first use by role and all uses
    # user_first_use[token] = (timestamp, conversation_id)
    # assistant_first_use[token] = (timestamp, conversation_id)
    # user_all_uses[token] = [(timestamp, conversation_id), ...]
    # assistant_all_uses[token] = [(timestamp, conversation_id), ...]
    user_first_use = {}      # token -> (timestamp, conv_id)
    assistant_first_use = {} # token -> (timestamp, conv_id)
    user_all_uses = defaultdict(list)       # token -> [(timestamp, conv_id), ...]
    assistant_all_uses = defaultdict(list)   # token -> [(timestamp, conv_id), ...]

    # Process assistant messages
    print("  Processing assistant messages...")
    asst_texts = asst_msgs[["text", "timestamp", "conversation_id"]].values.tolist()
    for doc, (text, ts, cid) in tqdm(
        zip(nlp.pipe([t[0] for t in asst_texts], batch_size=500, n_process=1),
            asst_texts),
        total=len(asst_texts),
        desc="  Assistant msgs"
    ):
        tokens = tokenize_message(doc, common_words)
        for token_str, token_type in tokens:
            if token_str not in assistant_first_use or ts < assistant_first_use[token_str][0]:
                assistant_first_use[token_str] = (ts, cid)
            assistant_all_uses[token_str].append((ts, cid))

    print(f"  Unique assistant tokens: {len(assistant_first_use):,}")

    # Process user messages
    print("  Processing user messages...")
    user_texts = user_msgs[["text", "timestamp", "conversation_id"]].values.tolist()
    for doc, (text, ts, cid) in tqdm(
        zip(nlp.pipe([t[0] for t in user_texts], batch_size=500, n_process=1),
            user_texts),
        total=len(user_texts),
        desc="  User msgs"
    ):
        tokens = tokenize_message(doc, common_words)
        for token_str, token_type in tokens:
            if token_str not in user_first_use or ts < user_first_use[token_str][0]:
                user_first_use[token_str] = (ts, cid)
            user_all_uses[token_str].append((ts, cid))

    print(f"  Unique user tokens: {len(user_first_use):,}")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Identify transfer candidates (with reinforcement filter)
    # -----------------------------------------------------------------------
    print("=== Step 3: Identifying transfer candidates ===")
    print(f"  Reinforcement threshold: AI must use word >= {args.min_ai_uses}x before user's first use")
    gap_threshold = pd.Timedelta(hours=args.gap_hours)

    candidates = []
    skipped_reinforcement = 0
    for token_str, (asst_ts, asst_cid) in tqdm(
        assistant_first_use.items(), desc="  Scanning transfers"
    ):
        if token_str not in user_first_use:
            continue

        user_ts, user_cid = user_first_use[token_str]

        # AI must have used it before the user
        if user_ts <= asst_ts:
            continue

        # Reinforcement filter: count AI uses strictly before user's first use
        ai_uses_before = sum(1 for t, c in assistant_all_uses[token_str] if t < user_ts)
        if ai_uses_before < args.min_ai_uses:
            skipped_reinforcement += 1
            continue

        # Gap must exceed threshold
        gap = user_ts - asst_ts
        if gap < gap_threshold:
            continue

        # Must be in a different conversation
        if user_cid == asst_cid:
            continue

        # Determine token type
        token_type = "bigram" if "_" in token_str else "unigram"

        # Count post-transfer user uses (after first user adoption)
        all_uses = user_all_uses[token_str]
        # Count uses at or after user_ts (the adoption point)
        post_transfer_uses = [(t, c) for t, c in all_uses if t >= user_ts]
        post_transfer_count = len(post_transfer_uses)
        post_transfer_convs = len(set(c for _, c in post_transfer_uses))

        candidates.append({
            "token": token_str,
            "token_type": token_type,
            "assistant_first_use": asst_ts,
            "user_first_use": user_ts,
            "gap_hours": gap.total_seconds() / 3600,
            "source_conversation_id": asst_cid,
            "adoption_conversation_id": user_cid,
            "post_transfer_user_count": post_transfer_count,
            "post_transfer_conversations": post_transfer_convs,
            "ai_uses_before_adoption": ai_uses_before,
        })

    print(f"  Skipped (AI used < {args.min_ai_uses}x before user): {skipped_reinforcement:,}")
    print(f"  Transfer candidates (gap >= {args.gap_hours}h, reinforced, different conv): {len(candidates):,}")

    # -----------------------------------------------------------------------
    # Step 4: Filter for meaningful transfers
    # -----------------------------------------------------------------------
    print("=== Step 4: Filtering transfers ===")
    df = pd.DataFrame(candidates)

    if len(df) == 0:
        print("  WARNING: No transfer candidates found!")
        report["warnings"].append("No transfer candidates found")
        report["transfer_stats"] = {"confirmed_transfers": 0}
        if not args.dry_run:
            with open(OUT_REPORT, "w", encoding="utf-8") as f:
                json.dump(clean_dict(report), f, indent=2, ensure_ascii=False)
        return

    total_before = len(df)

    # Filter: minimum post-transfer count
    df = df[df["post_transfer_user_count"] >= args.min_count].copy()
    after_min_count = len(df)
    print(f"  After min-count filter ({args.min_count}+ uses): {after_min_count:,} (removed {total_before - after_min_count:,})")

    if len(df) == 0:
        print("  WARNING: No transfers survive min-count filter!")
        report["warnings"].append("No transfers after min-count filter")
        report["transfer_stats"] = {"confirmed_transfers": 0, "total_candidates": total_before}
        if not args.dry_run:
            with open(OUT_REPORT, "w", encoding="utf-8") as f:
                json.dump(clean_dict(report), f, indent=2, ensure_ascii=False)
        return

    # -----------------------------------------------------------------------
    # Step 5: Classify stickiness
    # -----------------------------------------------------------------------
    print("=== Step 5: Classifying stickiness ===")

    def classify_stickiness(row):
        if row["post_transfer_user_count"] >= 5 or row["post_transfer_conversations"] >= 3:
            return "sticky"
        elif row["post_transfer_user_count"] >= 2:
            return "repeated"
        return "one_time"

    df["stickiness"] = df.apply(classify_stickiness, axis=1)

    sticky_counts = df["stickiness"].value_counts()
    for cat in ["one_time", "repeated", "sticky"]:
        n = sticky_counts.get(cat, 0)
        print(f"  {cat:12s}: {n:5,}  ({n / len(df) * 100:.1f}%)")
    print()

    # -----------------------------------------------------------------------
    # Step 6: Add conversation context (functional + emotional labels)
    # -----------------------------------------------------------------------
    print("=== Step 6: Adding conversation context ===")

    if func_df is not None:
        func_map = func_df.set_index("conversation_id")["functional_category"].to_dict()
        df["source_function"] = df["source_conversation_id"].map(func_map)
        df["adoption_function"] = df["adoption_conversation_id"].map(func_map)
        print(f"  Functional labels mapped: {df['source_function'].notna().sum():,} source, {df['adoption_function'].notna().sum():,} adoption")
    else:
        df["source_function"] = np.nan
        df["adoption_function"] = np.nan

    if emot_df is not None:
        emot_map = emot_df.set_index("conversation_id")["emotional_state"].to_dict()
        df["source_emotion"] = df["source_conversation_id"].map(emot_map)
        df["adoption_emotion"] = df["adoption_conversation_id"].map(emot_map)
        print(f"  Emotional labels mapped: {df['source_emotion'].notna().sum():,} source, {df['adoption_emotion'].notna().sum():,} adoption")
    else:
        df["source_emotion"] = np.nan
        df["adoption_emotion"] = np.nan

    print()

    # -----------------------------------------------------------------------
    # Step 7: Build monthly vocabulary timeline
    # -----------------------------------------------------------------------
    print("=== Step 7: Building monthly vocabulary timeline ===")

    # Get full month range
    all_timestamps = msgs["timestamp"]
    min_month = all_timestamps.min().strftime("%Y-%m")
    max_month = all_timestamps.max().strftime("%Y-%m")
    all_months = pd.date_range(min_month, max_month, freq="MS").strftime("%Y-%m").tolist()

    # User's monthly vocabulary
    user_monthly_tokens = defaultdict(set)  # month -> set of unique tokens
    user_new_tokens = defaultdict(set)      # month -> tokens first used that month
    user_seen_ever = set()

    for token_str, uses in user_all_uses.items():
        for ts, cid in uses:
            month = ts.strftime("%Y-%m")
            user_monthly_tokens[month].add(token_str)
            if token_str not in user_seen_ever:
                user_new_tokens[month].add(token_str)
                user_seen_ever.add(token_str)

    # Transfers per month (by adoption date)
    transfer_months = df["user_first_use"].dt.strftime("%Y-%m").value_counts().to_dict()
    sticky_months = df[df["stickiness"] == "sticky"]["user_first_use"].dt.strftime("%Y-%m").value_counts().to_dict()

    timeline_rows = []
    cumulative_transfers = 0
    cumulative_sticky = 0

    for month in all_months:
        unique = len(user_monthly_tokens.get(month, set()))
        new = len(user_new_tokens.get(month, set()))
        transfers = transfer_months.get(month, 0)
        stickies = sticky_months.get(month, 0)
        cumulative_transfers += transfers
        cumulative_sticky += stickies
        transfer_rate = transfers / new if new > 0 else 0.0

        timeline_rows.append({
            "year_month": month,
            "unique_tokens_used": unique,
            "new_tokens_introduced": new,
            "transfers_this_month": transfers,
            "transfer_rate": round(transfer_rate, 4),
            "cumulative_transfers": cumulative_transfers,
            "sticky_transfers_cumulative": cumulative_sticky,
        })

    timeline_df = pd.DataFrame(timeline_rows)
    print(f"  Timeline spans {len(all_months)} months: {min_month} to {max_month}")
    print(f"  Cumulative transfers: {cumulative_transfers:,}")
    print(f"  Cumulative sticky: {cumulative_sticky:,}")
    print()

    # -----------------------------------------------------------------------
    # Step 8: Compute statistics for report
    # -----------------------------------------------------------------------
    print("=== Step 8: Computing statistics ===")

    confirmed = len(df)
    unigrams = (df["token_type"] == "unigram").sum()
    bigrams = (df["token_type"] == "bigram").sum()

    gap_mean = df["gap_hours"].mean()
    gap_median = df["gap_hours"].median()

    # Transfer rate trend
    tl_with_data = timeline_df[timeline_df["transfers_this_month"] > 0]
    if len(tl_with_data) >= 3:
        x = np.arange(len(timeline_df))
        y = timeline_df["transfers_this_month"].values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    else:
        slope = 0.0
        trend_direction = "insufficient_data"

    # AI vocabulary share
    total_user_unique = len(user_first_use)
    ai_vocab_share = confirmed / total_user_unique * 100 if total_user_unique > 0 else 0.0

    # Cross-tabulations
    cross_tabs = {}

    for label, col in [
        ("transfers_by_source_function", "source_function"),
        ("transfers_by_source_emotion", "source_emotion"),
        ("transfers_by_adoption_function", "adoption_function"),
        ("transfers_by_adoption_emotion", "adoption_emotion"),
    ]:
        vc = df[col].dropna().value_counts().to_dict()
        cross_tabs[label] = {str(k): int(v) for k, v in vc.items()}

    # Sticky transfers by source function
    sticky_df = df[df["stickiness"] == "sticky"]
    cross_tabs["sticky_by_source_function"] = {
        str(k): int(v) for k, v in
        sticky_df["source_function"].dropna().value_counts().to_dict().items()
    }

    # Cross-domain transfers
    cross_domain = df.dropna(subset=["source_function", "adoption_function"])
    cross_domain_counts = cross_domain.groupby(
        ["source_function", "adoption_function"]
    ).size().reset_index(name="count")
    cross_tabs["cross_domain_transfers"] = [
        {"source": row["source_function"], "target": row["adoption_function"], "count": int(row["count"])}
        for _, row in cross_domain_counts.iterrows()
    ]

    # Top transfers
    top_transfers = df.nlargest(min(50, len(df)), "post_transfer_user_count")[
        ["token", "post_transfer_user_count", "post_transfer_conversations",
         "stickiness", "gap_hours", "token_type"]
    ].to_dict("records")

    # Signature fragment
    most_productive_func = ""
    if cross_tabs.get("transfers_by_source_function"):
        most_productive_func = max(
            cross_tabs["transfers_by_source_function"],
            key=cross_tabs["transfers_by_source_function"].get
        )

    most_productive_emotion = ""
    if cross_tabs.get("transfers_by_source_emotion"):
        most_productive_emotion = max(
            cross_tabs["transfers_by_source_emotion"],
            key=cross_tabs["transfers_by_source_emotion"].get
        )

    sticky_count = (df["stickiness"] == "sticky").sum()
    top_5_sticky = df[df["stickiness"] == "sticky"].nlargest(
        min(5, sticky_count), "post_transfer_user_count"
    )["token"].tolist()

    # Conversation position analysis
    position_analysis = {
        "early_third_adoption_rate": None,
        "middle_third_adoption_rate": None,
        "late_third_adoption_rate": None,
    }

    summary = (
        f"Confirmed transfers: {confirmed:,} ({unigrams:,} unigrams, {bigrams:,} bigrams). "
        f"Sticky: {sticky_count:,} ({sticky_count / confirmed * 100:.1f}%). "
        f"Mean gap: {gap_mean:.0f}h. "
        f"AI vocabulary share: {ai_vocab_share:.1f}%. "
        f"Trend: {trend_direction} ({slope:+.2f} transfers/month). "
        f"Most productive source: {most_productive_func}."
    )

    print(f"  {summary}")
    print()

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    report.update({
        "corpus_stats": {
            "total_user_tokens_unique": int(total_user_unique),
            "total_assistant_tokens_unique": int(len(assistant_first_use)),
            "total_messages_processed": int(len(msgs)),
            "gap_threshold_hours": args.gap_hours,
            "common_word_filter": f"top_{args.freq_threshold}",
            "min_post_transfer_count": args.min_count,
            "min_ai_uses_before_adoption": args.min_ai_uses,
            "skipped_insufficient_reinforcement": skipped_reinforcement,
        },
        "transfer_stats": {
            "total_transfer_candidates": int(total_before),
            "after_min_count_filter": int(after_min_count),
            "confirmed_transfers": int(confirmed),
            "unigram_transfers": int(unigrams),
            "bigram_transfers": int(bigrams),
        },
        "stickiness": {
            cat: {
                "count": int(sticky_counts.get(cat, 0)),
                "pct": round(sticky_counts.get(cat, 0) / confirmed * 100, 1)
            }
            for cat in ["one_time", "repeated", "sticky"]
        },
        "top_transfers": clean_dict(top_transfers),
        "temporal": {
            "mean_gap_hours": round(float(gap_mean), 1),
            "median_gap_hours": round(float(gap_median), 1),
            "transfer_rate_trend": trend_direction,
            "transfer_rate_slope_per_month": round(float(slope), 3),
            "monthly_transfer_counts": {
                row["year_month"]: int(row["transfers_this_month"])
                for _, row in timeline_df.iterrows()
            },
        },
        "cross_tabulations": clean_dict(cross_tabs),
        "vocabulary_growth": {
            "monthly_unique_user_tokens": {
                row["year_month"]: int(row["unique_tokens_used"])
                for _, row in timeline_df.iterrows()
            },
            "monthly_cumulative_transfers": {
                row["year_month"]: int(row["cumulative_transfers"])
                for _, row in timeline_df.iterrows()
            },
            "ai_vocabulary_share_pct": round(ai_vocab_share, 2),
        },
        "signature_fragment": {
            "total_sticky_transfers": int(sticky_count),
            "top_5_sticky_terms": top_5_sticky,
            "most_productive_function": most_productive_func,
            "most_productive_emotion": most_productive_emotion,
            "transfer_rate_trend": trend_direction,
            "ai_vocab_share_pct": round(ai_vocab_share, 2),
            "summary": summary,
        },
        "conversation_position_analysis": position_analysis,
    })

    # -----------------------------------------------------------------------
    # Step 9: Generate figures
    # -----------------------------------------------------------------------
    print("=== Step 9: Generating figures ===")
    figures_generated = []

    # -- Figure 1: Top Transferred Terms ------------------------------------
    try:
        top_n = min(30, len(df))
        top = df.nlargest(top_n, "post_transfer_user_count")

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
        colors = [STICKINESS_COLORS.get(s, COLOR_SECONDARY) for s in top["stickiness"]]
        bars = ax.barh(range(top_n), top["post_transfer_user_count"].values, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["token"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Post-Transfer User Occurrences")
        ax.set_title("Top Transferred Terms: AI → User Vocabulary", fontsize=14, fontweight="bold")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=STICKINESS_COLORS["sticky"], label="Sticky (5+ uses or 3+ convs)"),
            Patch(facecolor=STICKINESS_COLORS["repeated"], label="Repeated (2-4 uses)"),
            Patch(facecolor=STICKINESS_COLORS["one_time"], label="One-time"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        path = figpath("fig1_top_transferred_terms.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 1: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 1 error: {e}")
        print(f"  Fig 1 ERROR: {e}")

    # -- Figure 2: Transfer Rate Over Time ----------------------------------
    try:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        months = timeline_df["year_month"]
        transfers = timeline_df["transfers_this_month"]
        rates = timeline_df["transfer_rate"]

        ax1.bar(range(len(months)), transfers, color=COLOR_PRIMARY, alpha=0.7, label="Transfers/month")
        ax1.set_ylabel("New Transfers", color=COLOR_PRIMARY)
        ax1.set_xticks(range(len(months)))
        ax1.set_xticklabels(months, rotation=45, ha="right", fontsize=8)

        ax2 = ax1.twinx()
        ax2.plot(range(len(months)), rates, color=COLOR_ACCENT, linewidth=2, marker="o",
                 markersize=4, label="Transfer rate")
        ax2.set_ylabel("Transfer Rate (transfers / new tokens)", color=COLOR_ACCENT)

        ax1.set_title("Vocabulary Transfer Rate Over Time", fontsize=14, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        plt.tight_layout()
        path = figpath("fig2_transfer_rate_over_time.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 2: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 2 error: {e}")
        print(f"  Fig 2 ERROR: {e}")

    # -- Figure 3: Stickiness Distribution ----------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        cats = ["one_time", "repeated", "sticky"]
        sizes = [sticky_counts.get(c, 0) for c in cats]
        colors = [STICKINESS_COLORS[c] for c in cats]
        labels = [f"{c.replace('_', ' ').title()}\n({s:,}, {s / confirmed * 100:.1f}%)"
                  for c, s in zip(cats, sizes)]

        wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                               textprops={"fontsize": 11})
        ax.set_title("Transfer Stickiness Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()
        path = figpath("fig3_stickiness_distribution.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 3: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 3 error: {e}")
        print(f"  Fig 3 ERROR: {e}")

    # -- Figure 4: Gap Distribution -----------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        gaps = df["gap_hours"].values
        # Log-scale bins
        log_bins = np.logspace(np.log10(max(gaps.min(), 1)), np.log10(gaps.max()), 40)
        ax.hist(gaps, bins=log_bins, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Gap Between AI Introduction and User Adoption (hours, log scale)")
        ax.set_ylabel("Number of Transfers")
        ax.set_title("Time to Vocabulary Adoption", fontsize=14, fontweight="bold")

        # Add reference lines
        for label, val in [("1 day", 24), ("1 week", 168), ("1 month", 720)]:
            if val <= gaps.max():
                ax.axvline(val, color=COLOR_ACCENT, linestyle="--", alpha=0.6)
                ax.text(val, ax.get_ylim()[1] * 0.9, f" {label}", fontsize=8, color=COLOR_ACCENT)

        plt.tight_layout()
        path = figpath("fig4_gap_distribution.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 4: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 4 error: {e}")
        print(f"  Fig 4 ERROR: {e}")

    # -- Figure 5: Transfer Source by Functional Category -------------------
    try:
        src_func = df["source_function"].dropna().value_counts()
        if len(src_func) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            cats = src_func.index.tolist()
            vals = src_func.values
            colors = [FUNC_COLORS.get(c, COLOR_SECONDARY) for c in cats]
            ax.bar(range(len(cats)), vals, color=colors)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Number of Transfers Originating")
            ax.set_title("Vocabulary Transfer Sources by Functional Category",
                         fontsize=14, fontweight="bold")

            plt.tight_layout()
            path = figpath("fig5_transfer_by_function.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            figures_generated.append(path)
            print(f"  Fig 5: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 5 error: {e}")
        print(f"  Fig 5 ERROR: {e}")

    # -- Figure 6: Transfer Source by Emotional State -----------------------
    try:
        src_emo = df["source_emotion"].dropna().value_counts()
        if len(src_emo) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            cats = src_emo.index.tolist()
            vals = src_emo.values
            colors = [EMOTION_COLORS.get(c, COLOR_SECONDARY) for c in cats]
            ax.bar(range(len(cats)), vals, color=colors)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Number of Transfers Originating")
            ax.set_title("Vocabulary Transfer Sources by Emotional State",
                         fontsize=14, fontweight="bold")

            plt.tight_layout()
            path = figpath("fig6_transfer_by_emotion.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            figures_generated.append(path)
            print(f"  Fig 6: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 6 error: {e}")
        print(f"  Fig 6 ERROR: {e}")

    # -- Figure 7: Cumulative Vocabulary Growth -----------------------------
    try:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Build cumulative user vocab per month
        cumulative_user = []
        seen = set()
        for month in all_months:
            seen.update(user_monthly_tokens.get(month, set()))
            cumulative_user.append(len(seen))

        cumulative_ai = timeline_df["cumulative_transfers"].values

        ax.plot(range(len(all_months)), cumulative_user, color=COLOR_PRIMARY,
                linewidth=2, label="Total unique user tokens")
        ax.set_ylabel("Cumulative Unique Tokens (user)", color=COLOR_PRIMARY)

        ax2 = ax.twinx()
        ax2.plot(range(len(all_months)), cumulative_ai, color=COLOR_ACCENT,
                 linewidth=2, label="AI-transferred tokens")
        ax2.set_ylabel("Cumulative AI Transfers", color=COLOR_ACCENT)

        ax.set_xticks(range(len(all_months)))
        ax.set_xticklabels(all_months, rotation=45, ha="right", fontsize=8)
        ax.set_title("Cumulative Vocabulary Growth: Total vs AI-Transferred",
                     fontsize=14, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        plt.tight_layout()
        path = figpath("fig7_cumulative_vocab_growth.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 7: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 7 error: {e}")
        print(f"  Fig 7 ERROR: {e}")

    # -- Figure 8: Transfer Network (HTML) ----------------------------------
    try:
        cd = cross_domain_counts
        if len(cd) > 0:
            # Build Sankey
            sources_list = sorted(cd["source_function"].unique())
            targets_list = sorted(cd["adoption_function"].unique())
            all_labels = sources_list + [f"{t} " for t in targets_list]  # space to disambiguate
            label_to_idx = {l: i for i, l in enumerate(all_labels)}

            sankey_source = [label_to_idx[r["source_function"]] for _, r in cd.iterrows()]
            sankey_target = [label_to_idx[f"{r['adoption_function']} "] for _, r in cd.iterrows()]
            sankey_value = cd["count"].tolist()

            node_colors = ([FUNC_COLORS.get(s, COLOR_SECONDARY) for s in sources_list] +
                           [FUNC_COLORS.get(t, COLOR_SECONDARY) for t in targets_list])

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    label=[l.strip() for l in all_labels],
                    color=node_colors,
                ),
                link=dict(
                    source=sankey_source,
                    target=sankey_target,
                    value=sankey_value,
                ),
            ))
            fig.update_layout(
                title_text="Vocabulary Transfer Flow: Source → Adoption Domain",
                font_size=11,
                width=1000, height=700,
            )

            path = figpath("fig8_transfer_network.html")
            fig.write_html(path, include_plotlyjs=True)
            figures_generated.append(path)
            print(f"  Fig 8: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 8 error: {e}")
        print(f"  Fig 8 ERROR: {e}")

    # -- Figure 9: Sticky Transfers Timeline --------------------------------
    try:
        sticky_terms = df[df["stickiness"] == "sticky"].nlargest(
            min(20, sticky_count), "post_transfer_user_count"
        )["token"].tolist()

        if len(sticky_terms) > 0:
            fig, ax = plt.subplots(figsize=(14, max(6, len(sticky_terms) * 0.4)))

            for i, term in enumerate(sticky_terms):
                uses = user_all_uses[term]
                timestamps = [t for t, c in uses]
                ax.scatter(timestamps, [i] * len(timestamps), s=15, alpha=0.6,
                           color=STICKINESS_COLORS["sticky"])
                # Mark adoption point
                adoption_ts = df.loc[df["token"] == term, "user_first_use"].iloc[0]
                ax.scatter([adoption_ts], [i], s=60, marker="D", color=COLOR_ACCENT,
                           zorder=5, edgecolors="black", linewidth=0.5)

            ax.set_yticks(range(len(sticky_terms)))
            ax.set_yticklabels(sticky_terms, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Time")
            ax.set_title("Sticky Transfer Timeline (◆ = adoption point)",
                         fontsize=14, fontweight="bold")

            plt.tight_layout()
            path = figpath("fig9_sticky_timeline.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            figures_generated.append(path)
            print(f"  Fig 9: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 9 error: {e}")
        print(f"  Fig 9 ERROR: {e}")

    # -- Figure 10: Dashboard -----------------------------------------------
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # Panel 1: Summary stats
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis("off")
        stats_text = (
            f"Confirmed Transfers: {confirmed:,}\n"
            f"Unigrams: {unigrams:,} | Bigrams: {bigrams:,}\n"
            f"Sticky: {sticky_count:,} ({sticky_count / confirmed * 100:.1f}%)\n"
            f"Mean Gap: {gap_mean:.0f}h ({gap_mean / 24:.0f} days)\n"
            f"AI Vocab Share: {ai_vocab_share:.1f}%\n"
            f"Trend: {trend_direction} ({slope:+.2f}/mo)"
        )
        ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes,
                 fontsize=11, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        ax1.set_title("Summary", fontsize=12, fontweight="bold")

        # Panel 2: Top 10 terms
        ax2 = fig.add_subplot(gs[0, 1:])
        top10 = df.nlargest(10, "post_transfer_user_count")
        colors = [STICKINESS_COLORS.get(s, COLOR_SECONDARY) for s in top10["stickiness"]]
        ax2.barh(range(10), top10["post_transfer_user_count"].values, color=colors)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(top10["token"].values, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Post-Transfer Uses")
        ax2.set_title("Top 10 Transferred Terms", fontsize=12, fontweight="bold")

        # Panel 3: Stickiness pie
        ax3 = fig.add_subplot(gs[1, 0])
        cats = ["one_time", "repeated", "sticky"]
        sizes = [sticky_counts.get(c, 0) for c in cats]
        pie_colors = [STICKINESS_COLORS[c] for c in cats]
        ax3.pie(sizes, labels=[c.replace("_", " ").title() for c in cats],
                colors=pie_colors, startangle=90, textprops={"fontsize": 9})
        ax3.set_title("Stickiness", fontsize=12, fontweight="bold")

        # Panel 4: Transfer rate over time
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.bar(range(len(all_months)), timeline_df["transfers_this_month"],
                color=COLOR_PRIMARY, alpha=0.7)
        ax4.set_xticks(range(len(all_months)))
        ax4.set_xticklabels(all_months, rotation=45, ha="right", fontsize=7)
        ax4.set_ylabel("Transfers/Month")
        ax4.set_title("Monthly Transfer Count", fontsize=12, fontweight="bold")

        # Panel 5: Top source functions
        ax5 = fig.add_subplot(gs[2, 0:2])
        if len(cross_tabs.get("transfers_by_source_function", {})) > 0:
            sf = cross_tabs["transfers_by_source_function"]
            sorted_sf = sorted(sf.items(), key=lambda x: -x[1])[:8]
            cats_sf = [x[0] for x in sorted_sf]
            vals_sf = [x[1] for x in sorted_sf]
            colors_sf = [FUNC_COLORS.get(c, COLOR_SECONDARY) for c in cats_sf]
            ax5.barh(range(len(cats_sf)), vals_sf, color=colors_sf)
            ax5.set_yticks(range(len(cats_sf)))
            ax5.set_yticklabels(cats_sf, fontsize=9)
            ax5.invert_yaxis()
            ax5.set_xlabel("Transfers")
            ax5.set_title("Top Source Functions", fontsize=12, fontweight="bold")

        # Panel 6: Key insight
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis("off")
        insight = (
            f"Top 5 Sticky Terms:\n"
            + "\n".join(f"  • {t}" for t in top_5_sticky[:5])
            + f"\n\nMost productive:\n  {most_productive_func}"
        )
        ax6.text(0.1, 0.5, insight, transform=ax6.transAxes,
                 fontsize=10, verticalalignment="center",
                 bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.8))
        ax6.set_title("Key Findings", fontsize=12, fontweight="bold")

        fig.suptitle("Vocabulary Transfer Dashboard", fontsize=16, fontweight="bold", y=0.98)
        path = figpath("fig10_dashboard.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        figures_generated.append(path)
        print(f"  Fig 10: {path}")
    except Exception as e:
        report["warnings"].append(f"Figure 10 error: {e}")
        print(f"  Fig 10 ERROR: {e}")

    print()
    report["figures_generated"] = [os.path.relpath(f, BASE) for f in figures_generated]

    # -----------------------------------------------------------------------
    # Step 10: Save outputs
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN — skipping saves ===")
        print(f"  Would save {len(df):,} transfers to {OUT_PARQUET}")
        print(f"  Would save {len(timeline_df):,} timeline rows to {OUT_TIMELINE}")
        print(f"  Would save report to {OUT_REPORT}")
    else:
        print("=== Step 10: Saving outputs ===")

        # Save transfer parquet
        out_df = df.copy()
        for col in ["token_type", "stickiness", "source_function", "adoption_function",
                     "source_emotion", "adoption_emotion"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype("category")
        out_df["post_transfer_user_count"] = out_df["post_transfer_user_count"].astype("int32")
        out_df["post_transfer_conversations"] = out_df["post_transfer_conversations"].astype("int32")
        out_df["gap_hours"] = out_df["gap_hours"].astype("float32")

        out_df.to_parquet(OUT_PARQUET, index=False)
        report["data_outputs"].append("data/processed/vocab_transfer.parquet")
        print(f"  Saved {len(out_df):,} transfers to {OUT_PARQUET}")

        # Save timeline parquet
        timeline_df.to_parquet(OUT_TIMELINE, index=False)
        report["data_outputs"].append("data/processed/vocab_user_timeline.parquet")
        print(f"  Saved {len(timeline_df):,} timeline rows to {OUT_TIMELINE}")

        # Save report
        with open(OUT_REPORT, "w", encoding="utf-8") as f:
            json.dump(clean_dict(report), f, indent=2, ensure_ascii=False)
        print(f"  Saved report to {OUT_REPORT}")

    # -----------------------------------------------------------------------
    # Step 11: Validation
    # -----------------------------------------------------------------------
    print()
    print("=== Validation ===")
    checks = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        checks.append((name, status))
        print(f"  [{status}] {name}")

    if not args.dry_run:
        check("vocab_transfer.parquet exists", os.path.exists(OUT_PARQUET))
        check("vocab_user_timeline.parquet exists", os.path.exists(OUT_TIMELINE))
    check(f"All gaps >= {args.gap_hours}h", (df["gap_hours"] >= args.gap_hours).all())
    check(f"All post_transfer_count >= {args.min_count}", (df["post_transfer_user_count"] >= args.min_count).all())
    check("Stickiness labels valid", set(df["stickiness"].unique()).issubset({"one_time", "repeated", "sticky"}))
    check("user_first_use > assistant_first_use", (df["user_first_use"] > df["assistant_first_use"]).all())
    check("Timeline has no month gaps", len(timeline_df) == len(all_months))
    if not args.dry_run:
        check("Report JSON exists", os.path.exists(OUT_REPORT))
    check("Signature summary non-empty", len(summary) > 0)
    check(f"Top transfers has entries", len(top_transfers) >= min(10, confirmed))

    # Check figures
    png_figs = [f for f in figures_generated if f.endswith(".png")]
    html_figs = [f for f in figures_generated if f.endswith(".html")]
    for fp in png_figs:
        size = os.path.getsize(fp) if os.path.exists(fp) else 0
        check(f"{os.path.basename(fp)} >= 10KB", size >= 10240)
    for fp in html_figs:
        check(f"{os.path.basename(fp)} is self-contained",
              os.path.exists(fp) and os.path.getsize(fp) > 1000)

    passed = sum(1 for _, s in checks if s == "PASS")
    failed = sum(1 for _, s in checks if s == "FAIL")
    print()
    print(f"  Passed: {passed}/{len(checks)}   Failed: {failed}/{len(checks)}")

    if failed > 0:
        report["warnings"].append(f"Validation: {failed} checks failed")

    print()
    print("=" * 80)
    print("Module 3.2f: Vocabulary Transfer — COMPLETE")
    print("=" * 80)
    print(f"  {summary}")


if __name__ == "__main__":
    main()
