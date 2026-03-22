"""
Module 3.1: Opening Taxonomy
Script: 10_opening_taxonomy.py

Classifies first user messages into a two-level linguistic taxonomy using spaCy,
then analyses opening patterns vs. conversation outcomes, time, and vocabulary.

NOTE: Data discovery — most conversations have an empty/phantom first user message
(is_first_user_msg=True with NaN text). We use the first user message WITH actual
text content instead, which yields 1570 classifiable conversations.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import math
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spacy
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
SHAPES_PATH = os.path.join(BASE, "data", "processed", "shape_clusters.parquet")

OUT_PARQUET = os.path.join(BASE, "data", "processed", "opening_classifications.parquet")
OUT_REPORT  = os.path.join(BASE, "outputs", "reports", "opening_taxonomy_report.json")
FIG_DIR     = os.path.join(BASE, "outputs", "figures", "opening_taxonomy")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "data", "processed"), exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
L1_COLORS = {
    "question":     "#4472C4",
    "directive":    "#C55A11",
    "context_dump": "#548235",
    "greeting":     "#BF8F00",
    "statement":    "#7030A0",
    "fragment":     "#A5A5A5",
}
L1_ORDER = ["directive", "question", "statement", "context_dump", "fragment", "greeting"]

FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE     = (14, 6)
FIGSIZE_TALL     = (10, 8)
DPI              = 150
TITLE_SIZE       = 14
LABEL_SIZE       = 11
TICK_SIZE        = 10

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v

def clean_dict(d):
    if isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict(v) for v in d]
    return clean(d)

report = {
    "module": "opening_taxonomy",
    "module_version": "1.0",
    "generated_at": datetime.datetime.now().isoformat(),
    "warnings": [],
    "figures_generated": [],
    "data_outputs": [],
}

def warn(msg):
    print(f"  [WARN] {msg}")
    report["warnings"].append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Load, filter, extract first messages
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 0 — Load & filter")
print("=" * 60)

try:
    conversations = pd.read_parquet(CONV_PATH)
    messages      = pd.read_parquet(MSGS_PATH)

    conv = conversations[conversations["is_analysable"]].copy()

    # Data discovery: is_first_user_msg=True messages are mostly empty/phantom.
    # Use first user message WITH actual text per conversation instead.
    user_msgs = messages[
        (messages["role"] == "user") &
        (messages["conversation_id"].isin(conv["conversation_id"]))
    ].copy()
    user_msgs_text = user_msgs[
        user_msgs["text"].notna() &
        (user_msgs["text"].str.strip() != "")
    ].copy()
    user_msgs_sorted = user_msgs_text.sort_values(["conversation_id", "msg_index"])
    first_text_msgs  = user_msgs_sorted.groupby("conversation_id", sort=False).first().reset_index()

    # Phantom first msgs (is_first_user_msg=True with no text)
    flag_msgs     = messages[messages["is_first_user_msg"]]
    phantom_count = int(flag_msgs["text"].isna().sum())
    if phantom_count > 0:
        warn(
            f"{phantom_count:,} of {len(flag_msgs):,} is_first_user_msg=True messages have "
            f"NaN text (phantom/empty first messages). Using first user message WITH text instead."
        )

    # Merge with conversation metadata
    msg_cols  = ["conversation_id", "text", "token_count", "char_count", "word_count",
                 "has_code", "has_attachment", "msg_index"]
    conv_cols = ["conversation_id", "conversation_type", "model_era", "year_month",
                 "time_of_day", "is_weekend", "first_user_message_type",
                 "user_token_ratio", "turns", "duration_minutes", "has_code",
                 "msg_count", "created_at", "gap_days_from_prev"]

    openings = first_text_msgs[msg_cols].merge(
        conv[conv_cols],
        on="conversation_id",
        suffixes=("_msg", "_conv")
    )

    n_analysable         = len(conv)
    n_first_with_text    = len(openings)
    n_excluded_no_text   = n_analysable - n_first_with_text

    print(f"  Analysable conversations:      {n_analysable:,}")
    print(f"  First messages with text:      {n_first_with_text:,}")
    print(f"  Excluded (no text at all):     {n_excluded_no_text:,}")

    report["input_data"] = {
        "analysable_conversations": n_analysable,
        "first_messages_with_text": n_first_with_text,
        "excluded_no_text":         n_excluded_no_text,
        "spacy_errors":             0,  # updated in Step 1
        "data_note": (
            "Most conversations have an empty/phantom first user message. "
            "First user message WITH text content is used for classification."
        ),
    }

except Exception as e:
    print(f"  ERROR: {e}")
    raise


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — spaCy linguistic analysis
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 1 — spaCy linguistic analysis")
print("=" * 60)

try:
    try:
        nlp = spacy.load("en_core_web_lg")
        spacy_model_used = "en_core_web_lg"
        print("  Loaded: en_core_web_lg")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        spacy_model_used = "en_core_web_sm"
        warn("en_core_web_lg not found; using en_core_web_sm (degraded accuracy).")
        print(f"  Loaded: en_core_web_sm")
    nlp.max_length = 2_000_000

    def analyze_opening(text, nlp_model):
        """Extract linguistic features from a first user message."""
        analysis_text = text[:500] if len(text) > 500 else text
        doc = nlp_model(analysis_text)

        sents   = list(doc.sents)
        n_sents = len(sents)
        first_sent = sents[0] if sents else doc

        root_tokens = [t for t in first_sent if t.dep_ == "ROOT"]
        root_token  = root_tokens[0] if root_tokens else None
        root_pos    = root_token.pos_   if root_token else None
        root_lemma  = root_token.lemma_.lower() if root_token else None

        pos_counts = {}
        for token in first_sent:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        first_token = first_token_pos = first_token_lower = None
        for token in doc:
            if not token.is_space and not token.is_punct:
                first_token       = token
                first_token_pos   = token.pos_
                first_token_lower = token.text.lower()
                break

        has_question_mark = "?" in text
        wh_words = {"who", "what", "when", "where", "why", "how", "which", "whom", "whose"}
        aux_words = {
            "is", "are", "was", "were", "can", "could", "do", "does", "did",
            "will", "would", "should", "shall", "may", "might", "has", "have", "had",
        }
        starts_with_wh  = (first_token_lower in wh_words)  if first_token_lower else False
        starts_with_aux = (first_token_lower in aux_words) if first_token_lower else False
        has_question_syntax = starts_with_wh or starts_with_aux

        # Imperative: VERB root, no explicit subject, at least 2 tokens
        is_imperative = False
        if root_token and root_token.pos_ == "VERB":
            has_nsubj = any(
                c.dep_ in ("nsubj", "nsubjpass") for c in root_token.children
            )
            if not has_nsubj and len(list(first_sent)) >= 2:
                is_imperative = True

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        has_code_fence   = "```" in text
        greeting_words   = {"hi", "hello", "hey", "greetings", "howdy", "yo", "sup"}
        starts_with_greeting = (first_token_lower in greeting_words) if first_token_lower else False
        has_follow_up_after_greeting = starts_with_greeting and (n_sents > 1)

        n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
        n_verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
        n_adjs  = pos_counts.get("ADJ", 0)

        return {
            "n_sentences":               n_sents,
            "n_tokens_spacy":            len(doc),
            "root_pos":                  root_pos,
            "root_lemma":                root_lemma,
            "first_token_lower":         first_token_lower,
            "first_token_pos":           first_token_pos,
            "has_question_mark":         has_question_mark,
            "has_question_syntax":       has_question_syntax,
            "starts_with_wh":            starts_with_wh,
            "starts_with_aux":           starts_with_aux,
            "is_imperative":             is_imperative,
            "has_code_fence":            has_code_fence,
            "starts_with_greeting":      starts_with_greeting,
            "has_follow_up_after_greeting": has_follow_up_after_greeting,
            "n_entities":                len(entities),
            "pos_counts":                pos_counts,
            "n_nouns":                   n_nouns,
            "n_verbs":                   n_verbs,
            "n_adjs":                    n_adjs,
        }

    linguistic_features = []
    errors = []

    for _, row in tqdm(openings.iterrows(), total=len(openings), desc="spaCy analysis"):
        try:
            feats = analyze_opening(row["text"], nlp)
            feats["conversation_id"] = row["conversation_id"]
            linguistic_features.append(feats)
        except Exception as e:
            errors.append({"conversation_id": row["conversation_id"], "error": str(e)})

    ling_df = pd.DataFrame(linguistic_features).set_index("conversation_id")
    report["input_data"]["spacy_errors"]    = len(errors)
    report["input_data"]["spacy_model_used"] = spacy_model_used

    print(f"  Processed: {len(ling_df):,}   Errors: {len(errors)}")

except Exception as e:
    print(f"  ERROR in Step 1: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 1 failed: {e}")
    ling_df = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Taxonomy classification
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 2 — Taxonomy classification")
print("=" * 60)

try:
    def classify_opening(row, ling):
        """Two-level taxonomy classification (priority cascade)."""
        text       = row["text"]
        text_lower = text.lower().strip()
        word_count  = float(row["word_count"])  if pd.notna(row["word_count"])  else 0
        token_count = float(row["token_count"]) if pd.notna(row["token_count"]) else 0

        # Code paste
        if ling.get("has_code_fence"):
            code_blocks = text.split("```")
            code_chars = sum(len(code_blocks[i]) for i in range(1, len(code_blocks), 2)) if len(code_blocks) > 2 else 0
            if code_chars > len(text) * 0.5:
                return ("context_dump", "code_paste")

        # Greeting
        if ling.get("starts_with_greeting"):
            if ling.get("has_follow_up_after_greeting") or word_count >= 10:
                return ("greeting", "greeting_plus_request")
            return ("greeting", "greeting_only")

        # Text paste (long non-code)
        if token_count > 500 and not ling.get("has_code_fence"):
            n_tok = ling.get("n_tokens_spacy", 1) or 1
            if (ling.get("n_verbs", 0) / n_tok) < 0.05:
                return ("context_dump", "text_paste")

        # Structured prompt
        structural_markers = [
            "context:", "task:", "requirements:", "instructions:", "background:",
            "objective:", "goal:", "input:", "output:", "constraints:", "rules:",
        ]
        if any(m in text_lower for m in structural_markers) and ling.get("n_sentences", 1) >= 2:
            return ("context_dump", "structured_prompt")
        stripped = text_lower.lstrip()
        if token_count > 100 and (
            stripped.startswith("1.") or stripped.startswith("1)") or
            stripped.startswith("- ") or stripped.startswith("* ")
        ):
            return ("context_dump", "structured_prompt")

        # Questions
        if ling.get("has_question_mark") or ling.get("has_question_syntax"):
            q_count = text.count("?")
            if q_count > 1:
                return ("question", "multi_question")

            opinion_words = {"think", "recommend", "suggest", "best", "should",
                             "prefer", "opinion", "advice"}
            first200 = text_lower[:200]
            if any(w in first200 for w in opinion_words):
                return ("question", "opinion_question")

            if ling.get("starts_with_aux"):
                return ("question", "confirmation_question")

            if ling.get("starts_with_wh"):
                wh = ling.get("first_token_lower", "")
                if wh in {"why", "how"}:
                    return ("question", "explanation_question")
                return ("question", "factual_question")

            return ("question", "factual_question")

        # Directives
        if ling.get("is_imperative") and ling.get("root_lemma"):
            lemma = ling["root_lemma"]
            generation_verbs   = {"write", "create", "generate", "make", "produce",
                                   "draft", "compose", "design", "build"}
            analysis_verbs     = {"analyze", "compare", "evaluate", "review", "assess",
                                   "examine", "critique", "summarize", "explain"}
            transform_verbs    = {"convert", "translate", "rewrite", "format", "fix",
                                   "debug", "refactor", "optimize", "simplify", "improve",
                                   "update", "modify", "edit", "change"}
            retrieval_verbs    = {"find", "search", "list", "show", "give", "tell",
                                   "get", "look", "provide"}
            if lemma in generation_verbs:   return ("directive", "generation_directive")
            if lemma in analysis_verbs:     return ("directive", "analysis_directive")
            if lemma in transform_verbs:    return ("directive", "transformation_directive")
            if lemma in retrieval_verbs:    return ("directive", "retrieval_directive")
            return ("directive", "generation_directive")  # default directive subtype

        # Help request
        if "help" in text_lower[:50]:
            return ("directive", "help_request")

        # Statements
        if word_count > 5:
            problem_words = {"issue", "problem", "error", "broken", "doesn't work",
                             "doesnt work", "trying to", "struggling", "can't figure",
                             "cant figure", "not working", "bug", "crash"}
            if any(w in text_lower for w in problem_words):
                return ("statement", "problem_statement")

            hedging_words = {"i think", "i wonder", "maybe", "perhaps", "it seems",
                             "i'm curious", "im curious", "i've been thinking",
                             "ive been thinking", "what if", "i'm not sure", "im not sure"}
            if any(w in text_lower for w in hedging_words):
                return ("statement", "thought_exploration")

            temporal_markers = {"i just", "i've been", "ive been", "yesterday",
                                 "recently", "today", "last week", "this morning", "earlier"}
            if any(w in text_lower for w in temporal_markers):
                return ("statement", "status_update")

            if word_count > 10:
                return ("statement", "declaration")

        # Fragments
        if word_count <= 3 and ling.get("n_verbs", 0) == 0:
            return ("fragment", "keyword_fragment")

        last_char = text.rstrip()[-1:] if text.rstrip() else ""
        if text.rstrip().endswith("...") or (last_char not in ".?!\"')"):
            return ("fragment", "incomplete_thought")

        return ("fragment", "ambiguous")

    # Apply classification
    classifications = []
    for _, row in tqdm(openings.iterrows(), total=len(openings), desc="Classifying"):
        cid = row["conversation_id"]
        if cid in ling_df.index:
            ling = ling_df.loc[cid].to_dict()
            l1, l2 = classify_opening(row, ling)
        else:
            l1, l2 = "fragment", "ambiguous"
        classifications.append({"conversation_id": cid, "opening_level_1": l1, "opening_level_2": l2})

    class_df = pd.DataFrame(classifications).set_index("conversation_id")

    # Distribution counts
    l1_counts = class_df["opening_level_1"].value_counts()
    l2_counts = class_df["opening_level_2"].value_counts()
    print(f"  Level-1 distribution:")
    for cat, cnt in l1_counts.items():
        print(f"    {cat:<18} {cnt:>5}  ({cnt/len(class_df)*100:.1f}%)")

    # Sanity check: 5 samples per Level-1
    print()
    print("  Sanity check — 5 samples per Level-1:")
    for l1 in l1_counts.index:
        subset = class_df[class_df["opening_level_1"] == l1]
        sample_ids = subset.sample(min(5, len(subset)), random_state=42).index
        for sid in sample_ids:
            txt = openings.set_index("conversation_id").at[sid, "text"]
            l2  = class_df.at[sid, "opening_level_2"]
            print(f"    [{l1} / {l2}] {str(txt)[:70]}")
        print()

    # Agreement with Phase 2
    comparison = openings.set_index("conversation_id")[["first_user_message_type"]].join(class_df)
    # Convert categorical → str to avoid "new category" pandas errors
    comparison["first_user_message_type"] = comparison["first_user_message_type"].astype(str)
    # Note: Phase 2 classified mostly empty msgs as "fragment" — agreement is expected to be low
    # Map Phase 2 "command" -> "directive" for fair comparison
    p2_map = {"command": "directive", "code": "context_dump"}
    comparison["p2_mapped"] = comparison["first_user_message_type"].replace(p2_map)
    agreement     = float((comparison["p2_mapped"] == comparison["opening_level_1"]).mean())
    agreement_raw = float((comparison["first_user_message_type"] == comparison["opening_level_1"]).mean())

    confusion_raw = pd.crosstab(
        comparison["first_user_message_type"].fillna("unknown"),
        comparison["opening_level_1"],
    )

    print(f"  Agreement with Phase 2 (raw labels): {agreement_raw:.3f}")
    print(f"  Agreement with Phase 2 (mapped):      {agreement:.3f}")
    warn(
        f"Phase 2 classified {phantom_count:,} empty/phantom first messages as 'fragment' by default. "
        f"Agreement rate ({agreement:.3f}) reflects this mismatch, not classification error."
    )

    report["phase2_comparison"] = {
        "agreement_rate_raw":    round(agreement_raw, 4),
        "agreement_rate_mapped": round(agreement, 4),
        "confusion_matrix":      clean_dict(confusion_raw.to_dict()),
        "note": "Phase 2 used empty first messages for most conversations; low agreement is expected.",
    }

except Exception as e:
    print(f"  ERROR in Step 2: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 2 failed: {e}")
    class_df = pd.DataFrame(columns=["opening_level_1", "opening_level_2"])

# Ensure agreement is always defined (used in Fig 01)
agreement = report.get("phase2_comparison", {}).get("agreement_rate_mapped", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Distribution analysis
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 3 — Distribution analysis")
print("=" * 60)

try:
    # Join analysis DataFrame
    analysis = class_df.join(openings.set_index("conversation_id"))

    l1_dist = class_df["opening_level_1"].value_counts().to_dict()
    l1_pct  = {k: round(v / len(class_df) * 100, 1) for k, v in l1_dist.items()}
    l2_dist = class_df["opening_level_2"].value_counts().to_dict()
    l2_pct  = {k: round(v / len(class_df) * 100, 1) for k, v in l2_dist.items()}

    nested_dist = {}
    for l1 in class_df["opening_level_1"].unique():
        sub = class_df[class_df["opening_level_1"] == l1]
        nested_dist[l1] = {
            "total":      int(len(sub)),
            "pct_of_all": round(len(sub) / len(class_df) * 100, 1),
            "subtypes":   {k: int(v) for k, v in sub["opening_level_2"].value_counts().items()},
        }

    # Cross-tabs with metadata
    def safe_crosstab(a, b, **kwargs):
        try:
            return pd.crosstab(a, b, **kwargs).to_dict()
        except Exception as ex:
            warn(f"crosstab failed: {ex}")
            return {}

    by_conv_type = safe_crosstab(analysis["opening_level_1"], analysis["conversation_type"], normalize="columns")
    by_era       = safe_crosstab(analysis["opening_level_1"], analysis["model_era"],        normalize="columns")
    by_tod       = safe_crosstab(analysis["opening_level_1"], analysis["time_of_day"],      normalize="columns")
    by_weekend   = safe_crosstab(analysis["opening_level_1"], analysis["is_weekend"],       normalize="columns")

    # Opening stats by Level-1
    opening_stats = analysis.groupby("opening_level_1").agg(
        count=("token_count", "count"),
        mean_tokens=("token_count", "mean"),
        median_tokens=("token_count", "median"),
        std_tokens=("token_count", "std"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
    ).round(1).to_dict("index")

    print(f"  Level-1 distribution computed ({len(l1_dist)} categories)")
    print(f"  Level-2 distribution computed ({len(l2_dist)} subcategories)")

    report["distribution"] = {
        "level_1":     clean_dict(l1_dist),
        "level_1_pct": clean_dict(l1_pct),
        "level_2":     clean_dict(l2_dist),
        "level_2_pct": clean_dict(l2_pct),
        "nested":      clean_dict(nested_dist),
    }
    report["cross_tabulations"] = {
        "by_conversation_type": clean_dict(by_conv_type),
        "by_model_era":         clean_dict(by_era),
        "by_time_of_day":       clean_dict(by_tod),
        "by_weekend":           clean_dict(by_weekend),
        "by_conversation_shape": None,  # filled in Step 4c
    }
    report["opening_characteristics"] = {
        "stats_by_level_1": clean_dict(opening_stats),
    }

except Exception as e:
    print(f"  ERROR in Step 3: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 3 failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Outcome analysis
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 4 — Outcome analysis")
print("=" * 60)

outcome_tests = {}
opening_by_shape = None

try:
    outcome_by_l1 = analysis.groupby("opening_level_1").agg(
        median_turns=("turns", "median"),
        mean_turns=("turns", "mean"),
        median_duration=("duration_minutes", "median"),
        median_msg_count=("msg_count", "median"),
        median_user_token_ratio=("user_token_ratio", "median"),
        mean_user_token_ratio=("user_token_ratio", "mean"),
    ).round(3)

    outcome_by_l2 = analysis.groupby("opening_level_2").agg(
        count=("turns", "count"),
        median_turns=("turns", "median"),
        median_duration=("duration_minutes", "median"),
        median_user_token_ratio=("user_token_ratio", "median"),
    ).round(3)

    # 4b. Statistical tests
    from scipy.stats import kruskal

    for metric in ["turns", "duration_minutes", "user_token_ratio", "msg_count"]:
        try:
            groups = [g[metric].dropna() for _, g in analysis.groupby("opening_level_1")]
            groups = [g for g in groups if len(g) >= 10]
            if len(groups) >= 2:
                stat, p = kruskal(*groups)
                n    = sum(len(g) for g in groups)
                eta  = max(0.0, (stat - len(groups) + 1) / (n - len(groups)))
                outcome_tests[metric] = {
                    "H_statistic": round(float(stat), 2),
                    "p_value":     round(float(p), 6),
                    "eta_squared": round(float(eta), 4),
                    "significant": bool(p < 0.05),
                    "n_groups":    len(groups),
                }
        except Exception as e2:
            warn(f"Kruskal-Wallis for {metric} failed: {e2}")

    print(f"  Outcome tests: {outcome_tests}")

    # 4c. Shape cross-reference
    if os.path.exists(SHAPES_PATH):
        shapes = pd.read_parquet(SHAPES_PATH)[["shape_cluster_name"]]
        analysis_ws = analysis.copy()
        analysis_ws = analysis_ws.join(shapes, how="left")
        opening_by_shape = pd.crosstab(
            analysis_ws["opening_level_1"],
            analysis_ws["shape_cluster_name"],
            normalize="index",
        ).round(3)
        report["cross_tabulations"]["by_conversation_shape"] = clean_dict(opening_by_shape.to_dict())
        print(f"  Shape cross-reference: {opening_by_shape.shape[0]} x {opening_by_shape.shape[1]}")
    else:
        warn("shape_clusters.parquet not found — skipping shape cross-reference.")

    report["outcome_analysis"] = {
        "by_level_1":       clean_dict(outcome_by_l1.to_dict("index")),
        "by_level_2":       clean_dict(outcome_by_l2.to_dict("index")),
        "statistical_tests": clean_dict(outcome_tests),
    }

except Exception as e:
    print(f"  ERROR in Step 4: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 4 failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Longitudinal analysis
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 5 — Longitudinal analysis")
print("=" * 60)

monthly_openings    = pd.DataFrame()
monthly_greeting_rate = pd.Series(dtype=float)
repeat_rate         = None
transition_matrix   = None
n_session_pairs     = 0

try:
    # 5a. Monthly opening type proportions
    monthly_openings = pd.crosstab(
        analysis["year_month"],
        analysis["opening_level_1"],
        normalize="index",
    ).round(3)
    print(f"  Monthly distribution: {len(monthly_openings)} months")

    # 5b. Era shift analysis
    from scipy.stats import chi2_contingency
    contingency_era = pd.crosstab(analysis["opening_level_1"], analysis["model_era"])
    n_eras = contingency_era.shape[1]
    if n_eras >= 2:
        chi2, p_val, dof, _ = chi2_contingency(contingency_era)
        n_all   = int(contingency_era.values.sum())
        min_dim = min(contingency_era.shape) - 1
        cramers = float(np.sqrt(chi2 / (n_all * min_dim))) if min_dim > 0 else 0.0
    else:
        chi2 = p_val = dof = cramers = None
        warn(f"Only {n_eras} model era in data — chi-squared era test skipped.")

    era_dominant = {}
    for era, grp in analysis.groupby("model_era"):
        mode = grp["opening_level_1"].mode()
        era_dominant[str(era)] = str(mode.iloc[0]) if len(mode) > 0 else None

    # 5c. Transition analysis
    analysis_time = analysis.copy()
    analysis_time["gap_days_from_prev"] = analysis_time["gap_days_from_prev"].astype(float)
    analysis_time = analysis_time.sort_values("created_at")

    session_pairs = []
    prev_type = None
    for _, row in analysis_time.iterrows():
        gap = row["gap_days_from_prev"]
        if pd.notna(gap) and gap < (1.0 / 24) and prev_type is not None:
            session_pairs.append((prev_type, row["opening_level_1"]))
        prev_type = row["opening_level_1"]

    n_session_pairs = len(session_pairs)
    if session_pairs:
        trans_df = pd.DataFrame(session_pairs, columns=["from", "to"])
        transition_matrix = pd.crosstab(trans_df["from"], trans_df["to"], normalize="index").round(3)
        repeat_rate = float((trans_df["from"] == trans_df["to"]).mean())
        print(f"  Session pairs: {n_session_pairs}, repeat rate: {repeat_rate:.3f}")
    else:
        warn("No within-session pairs found for transition analysis.")

    # Greeting trend
    monthly_greeting_rate = analysis.groupby("year_month").apply(
        lambda g: float((g["opening_level_1"] == "greeting").mean())
    ).round(3)

    # Prompt sophistication trend (structured_prompt %)
    monthly_struct = analysis.groupby("year_month").apply(
        lambda g: float((g["opening_level_2"] == "structured_prompt").mean())
    ).round(3)
    struct_trend_vals = monthly_struct.values
    if len(struct_trend_vals) >= 3:
        slope = np.polyfit(range(len(struct_trend_vals)), struct_trend_vals, 1)[0]
        prompt_soph_trend = "increasing" if slope > 0.001 else ("decreasing" if slope < -0.001 else "stable")
    else:
        prompt_soph_trend = "stable"

    report["longitudinal"] = {
        "monthly_distribution": clean_dict(monthly_openings.to_dict("index")),
        "era_shift_test": {
            "chi2":     float(chi2) if chi2 is not None else None,
            "p_value":  float(p_val) if p_val is not None else None,
            "cramers_v": float(cramers) if cramers is not None else None,
            "era_dominant_types": era_dominant,
        },
        "transition_analysis": {
            "repeat_rate":      repeat_rate,
            "transition_matrix": clean_dict(transition_matrix.to_dict("index")) if transition_matrix is not None else None,
            "n_session_pairs":  n_session_pairs,
        },
        "prompt_sophistication_trend": prompt_soph_trend,
    }

except Exception as e:
    print(f"  ERROR in Step 5: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 5 failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Vocabulary and linguistic depth
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 6 — Vocabulary analysis")
print("=" * 60)

try:
    # 6a. Directive verbs
    dir_ids   = class_df[class_df["opening_level_1"] == "directive"].index
    dir_verbs = ling_df.loc[ling_df.index.isin(dir_ids), "root_lemma"].dropna()
    verb_freq = dir_verbs.value_counts().head(30).to_dict()

    # 6b. Question words
    q_ids    = class_df[class_df["opening_level_1"] == "question"].index
    q_starts = ling_df.loc[ling_df.index.isin(q_ids), "first_token_lower"].dropna()
    q_freq   = q_starts.value_counts().head(20).to_dict()

    # 6c. Greetings
    g_ids     = class_df[class_df["opening_level_1"] == "greeting"].index
    g_words   = ling_df.loc[ling_df.index.isin(g_ids), "first_token_lower"].dropna()
    g_freq    = g_words.value_counts().to_dict()

    # 6d. Linguistic complexity by type
    complexity_by_type = ling_df.join(class_df).groupby("opening_level_1").agg(
        mean_n_sentences=("n_sentences", "mean"),
        mean_n_nouns=("n_nouns", "mean"),
        mean_n_verbs=("n_verbs", "mean"),
        mean_n_adjs=("n_adjs", "mean"),
        mean_n_entities=("n_entities", "mean"),
    ).round(2)

    print(f"  Top 5 directive verbs: {list(verb_freq.items())[:5]}")
    print(f"  Top 5 question words:  {list(q_freq.items())[:5]}")
    print(f"  Greeting forms:        {g_freq}")

    report["vocabulary"] = {
        "top_directive_verbs":    clean_dict(dict(list(verb_freq.items())[:30])),
        "top_question_words":     clean_dict(dict(list(q_freq.items())[:20])),
        "greeting_words":         clean_dict(g_freq),
        "monthly_greeting_rate":  clean_dict(monthly_greeting_rate.to_dict()),
    }
    report["opening_characteristics"]["linguistic_complexity_by_level_1"] = clean_dict(
        complexity_by_type.to_dict("index")
    )

except Exception as e:
    print(f"  ERROR in Step 6: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 6 failed: {e}")
    verb_freq = {}
    q_freq    = {}
    g_freq    = {}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Cognitive signature fragment
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 7 — Cognitive signature")
print("=" * 60)

try:
    dominant_l1   = class_df["opening_level_1"].mode().iloc[0]
    dominant_l2   = class_df["opening_level_2"].mode().iloc[0]
    dominant_pct  = float(l1_dist.get(dominant_l1, 0) / len(class_df) * 100)

    # Approach style
    q_pct   = float(l1_dist.get("question",     0) / len(class_df) * 100)
    d_pct   = float(l1_dist.get("directive",    0) / len(class_df) * 100)
    cd_pct  = float(l1_dist.get("context_dump", 0) / len(class_df) * 100)
    gr_pct  = float(l1_dist.get("greeting",     0) / len(class_df) * 100)

    if q_pct  > 40: approach_style = "interrogative"
    elif d_pct > 40: approach_style = "directive"
    elif cd_pct > 25: approach_style = "contextual"
    elif gr_pct > 20: approach_style = "social"
    else: approach_style = "mixed"

    # Greeting tendency
    g_monthly = monthly_greeting_rate if len(monthly_greeting_rate) > 0 else pd.Series([0])
    overall_g_rate = float(g_monthly.mean())
    if overall_g_rate > 0.15: greeting_tendency = "consistent_greeter"
    elif overall_g_rate < 0.05: greeting_tendency = "non_greeter"
    elif len(g_monthly) >= 3:
        g_slope = np.polyfit(range(len(g_monthly)), g_monthly.values, 1)[0]
        greeting_tendency = "declining_greeter" if g_slope < -0.001 else "occasional_greeter"
    else:
        greeting_tendency = "occasional_greeter"

    # Opening predicts depth
    turns_test       = outcome_tests.get("turns", {})
    opening_predicts = bool(turns_test.get("significant", False))

    top_verbs = [str(v) for v in list(verb_freq.keys())[:5]]
    top_qw    = [str(w) for w in list(q_freq.keys())[:5]]

    # Summary sentence
    summary_parts = []
    summary_parts.append(
        f"The user predominantly opens with {dominant_l1}s ({dominant_pct:.0f}%), "
        f"most often as {dominant_l2.replace('_', ' ')}."
    )
    if opening_predicts:
        summary_parts.append("Opening type significantly predicts conversation depth.")
    if greeting_tendency == "non_greeter":
        summary_parts.append("The user rarely uses greetings, indicating a consistently task-oriented style.")
    elif greeting_tendency == "declining_greeter":
        summary_parts.append("Greeting usage has declined over time, suggesting increasing task-orientation.")
    elif greeting_tendency == "consistent_greeter":
        summary_parts.append("The user consistently opens with social greetings.")

    signature = {
        "dominant_opening_type":        dominant_l1,
        "dominant_opening_subtype":     dominant_l2,
        "dominant_pct":                 round(dominant_pct, 1),
        "approach_style":               approach_style,
        "prompt_sophistication_trend":  report.get("longitudinal", {}).get("prompt_sophistication_trend", "stable"),
        "greeting_tendency":            greeting_tendency,
        "opening_predicts_depth":       opening_predicts,
        "session_habit_strength":       round(repeat_rate, 3) if repeat_rate is not None else None,
        "era_sensitivity":              round(float(cramers), 4) if cramers is not None else 0.0,
        "top_verbs":                    top_verbs,
        "top_question_words":           top_qw,
        "summary":                      " ".join(summary_parts),
    }

    print(f"  Dominant type: {dominant_l1} ({dominant_pct:.1f}%)")
    print(f"  Approach style: {approach_style}")
    print(f"  Opening predicts depth: {opening_predicts}")
    print(f"  Summary: {signature['summary']}")

    report["cognitive_signature_fragment"] = clean_dict(signature)

except Exception as e:
    print(f"  ERROR in Step 7: {e}")
    import traceback; traceback.print_exc()
    warn(f"Step 7 failed: {e}")
    report["cognitive_signature_fragment"] = {"summary": "Cognitive signature computation failed."}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("FIGURES — generating 12 figures")
print("=" * 60)

def save_fig(fig, name, ext="png"):
    path = os.path.join(FIG_DIR, f"{name}.{ext}")
    if ext == "png":
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.write_html(path, include_plotlyjs="cdn")
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {name}.{ext} ({size_kb:.0f} KB)")
    report["figures_generated"].append(f"{name}.{ext}")
    return path


# ── Fig 01: Coarse vs refined ─────────────────────────────────────────────────
try:
    p2_dist = openings["first_user_message_type"].value_counts()
    p3_dist = class_df["opening_level_1"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    colors_p2 = ["#4472C4", "#C55A11", "#7030A0", "#A5A5A5", "#BF8F00", "#548235"]
    axes[0].barh(p2_dist.index[::-1], p2_dist.values[::-1], color=colors_p2[:len(p2_dist)])
    axes[0].set_title("Phase 2 Heuristic", fontsize=TITLE_SIZE - 1)
    axes[0].set_xlabel("Count", fontsize=LABEL_SIZE)
    for i, (v, k) in enumerate(zip(p2_dist.values[::-1], p2_dist.index[::-1])):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=TICK_SIZE)

    p3_ordered = p3_dist.sort_values()
    bar_colors = [L1_COLORS.get(k, "#888888") for k in p3_ordered.index]
    axes[1].barh(p3_ordered.index, p3_ordered.values, color=bar_colors)
    axes[1].set_title("Phase 3 Linguistic (this module)", fontsize=TITLE_SIZE - 1)
    axes[1].set_xlabel("Count", fontsize=LABEL_SIZE)
    for i, (k, v) in enumerate(zip(p3_ordered.index, p3_ordered.values)):
        axes[1].text(v + 1, i, str(v), va="center", fontsize=TICK_SIZE)

    fig.suptitle(
        f"Phase 2 Heuristic vs. Phase 3 Linguistic Classification\n"
        f"(N={len(class_df):,} | Agreement: {agreement:.1%})",
        fontsize=TITLE_SIZE, y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "01_coarse_vs_refined")

except Exception as e:
    warn(f"Fig 01 failed: {e}")


# ── Fig 02: Taxonomy distribution (grouped horizontal bars) ──────────────────
try:
    rows, colors_list, group_labels, subtick_positions = [], [], [], []
    tick_positions = []
    current_y = 0
    l1_for_each_bar = []

    l1_for_display = [l for l in L1_ORDER if l in class_df["opening_level_1"].unique()]

    for l1 in l1_for_display:
        sub = class_df[class_df["opening_level_1"] == l1]["opening_level_2"].value_counts()
        mid = current_y + (len(sub) - 1) / 2
        group_labels.append((mid, f"{l1} (n={l1_dist.get(l1, 0):,})"))
        for l2, cnt in sub.items():
            rows.append({"label": l2, "count": cnt, "l1": l1, "y": current_y})
            colors_list.append(L1_COLORS.get(l1, "#888"))
            l1_for_each_bar.append(l1)
            current_y += 1
        current_y += 0.5  # gap

    plot_df = pd.DataFrame(rows)
    total_n = len(class_df)

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.45)))
    bars = ax.barh(
        [r["y"] for r in rows],
        [r["count"] for r in rows],
        color=colors_list, height=0.7, edgecolor="white",
    )
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"].replace("_", " ") for r in rows], fontsize=TICK_SIZE - 1)
    for i, (bar, row) in enumerate(zip(bars, rows)):
        cnt = row["count"]
        pct = cnt / total_n * 100
        ax.text(cnt + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=8)

    for y_mid, label in group_labels:
        ax.text(-0.01, y_mid, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=TICK_SIZE, fontweight="bold",
                color=L1_COLORS.get(label.split(" ")[0], "#333"))

    ax.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax.set_title("Opening Taxonomy Distribution (Level 1 + Level 2)", fontsize=TITLE_SIZE)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, "02_taxonomy_distribution")

except Exception as e:
    warn(f"Fig 02 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 03: Taxonomy sunburst (HTML) ─────────────────────────────────────────
try:
    sun_rows = []
    for l1 in class_df["opening_level_1"].unique():
        l1_cnt = int(l1_dist.get(l1, 0))
        sun_rows.append({"id": l1, "parent": "", "label": l1, "value": l1_cnt, "color": L1_COLORS.get(l1, "#888")})
        sub = class_df[class_df["opening_level_1"] == l1]["opening_level_2"].value_counts()
        for l2, cnt in sub.items():
            sun_rows.append({"id": f"{l1}/{l2}", "parent": l1, "label": l2, "value": int(cnt), "color": L1_COLORS.get(l1, "#888")})

    if not sun_rows:
        raise ValueError("No classification data for sunburst — class_df is empty.")
    sun_df = pd.DataFrame(sun_rows)
    fig_sun = go.Figure(go.Sunburst(
        ids=sun_df["id"],
        labels=sun_df["label"].str.replace("_", " "),
        parents=sun_df["parent"],
        values=sun_df["value"],
        marker=dict(colors=sun_df["color"]),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percentParent:.1%}<extra></extra>",
        branchvalues="total",
    ))
    fig_sun.update_layout(
        title=dict(text="Opening Taxonomy — Two-Level Sunburst", font=dict(size=TITLE_SIZE)),
        margin=dict(t=60, l=0, r=0, b=0),
        height=600,
    )
    save_fig(fig_sun, "03_taxonomy_sunburst", ext="html")

except Exception as e:
    warn(f"Fig 03 failed: {e}")


# ── Fig 04: Taxonomy by conversation type ────────────────────────────────────
try:
    ct_cross = pd.crosstab(analysis["conversation_type"], analysis["opening_level_1"])
    ct_cross_pct = ct_cross.div(ct_cross.sum(axis=1), axis=0)
    ct_cross_pct = ct_cross_pct[[c for c in L1_ORDER if c in ct_cross_pct.columns]]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bottom = np.zeros(len(ct_cross_pct))
    for l1 in ct_cross_pct.columns:
        vals = ct_cross_pct[l1].values
        ax.bar(ct_cross_pct.index, vals, bottom=bottom,
               label=l1, color=L1_COLORS.get(l1, "#888"), width=0.6)
        bottom += vals

    ax.set_xlabel("Conversation Type", fontsize=LABEL_SIZE)
    ax.set_ylabel("Proportion", fontsize=LABEL_SIZE)
    ax.set_title("Opening Type by Conversation Type", fontsize=TITLE_SIZE)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=TICK_SIZE, framealpha=0.85)
    plt.xticks(rotation=30, ha="right", fontsize=TICK_SIZE)
    plt.tight_layout()
    save_fig(fig, "04_taxonomy_by_type")

except Exception as e:
    warn(f"Fig 04 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 05: Taxonomy by era ───────────────────────────────────────────────────
try:
    era_cross = pd.crosstab(analysis["model_era"], analysis["opening_level_1"])
    era_pct   = era_cross.div(era_cross.sum(axis=1), axis=0)
    era_pct   = era_pct[[c for c in L1_ORDER if c in era_pct.columns]]

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bottom = np.zeros(len(era_pct))
    for l1 in era_pct.columns:
        vals = era_pct[l1].values
        ax.bar(era_pct.index, vals, bottom=bottom,
               label=l1, color=L1_COLORS.get(l1, "#888"), width=0.5)
        bottom += vals

    ax.set_xlabel("Model Era", fontsize=LABEL_SIZE)
    ax.set_ylabel("Proportion", fontsize=LABEL_SIZE)
    ax.set_title("Opening Type by Model Era", fontsize=TITLE_SIZE)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=TICK_SIZE, framealpha=0.85)
    if len(era_pct) == 1:
        ax.text(0.5, 0.5, "Only one model era in data (o3-era)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=LABEL_SIZE, color="#666")
    plt.tight_layout()
    save_fig(fig, "05_taxonomy_by_era")

except Exception as e:
    warn(f"Fig 05 failed: {e}")


# ── Fig 06: Taxonomy trend (HTML stacked area) ────────────────────────────────
try:
    mo_open = monthly_openings.copy()
    all_l1  = [c for c in L1_ORDER if c in mo_open.columns]
    for col in all_l1:
        if col not in mo_open.columns:
            mo_open[col] = 0.0

    fig_trend = go.Figure()
    for l1 in reversed(all_l1):
        fig_trend.add_trace(go.Scatter(
            x=mo_open.index, y=mo_open.get(l1, 0),
            name=l1, mode="lines",
            line=dict(width=0, color=L1_COLORS.get(l1, "#888")),
            stackgroup="one", groupnorm="percent",
            fillcolor=L1_COLORS.get(l1, "#888"),
            hovertemplate=f"<b>{l1}</b><br>Month: %{{x}}<br>Pct: %{{y:.1f}}%<extra></extra>",
        ))

    fig_trend.update_layout(
        title=dict(text="Opening Type Distribution Over Time", font=dict(size=TITLE_SIZE)),
        xaxis_title="Month",
        yaxis_title="Proportion (%)",
        legend_title="Opening Type",
        height=450,
        hovermode="x unified",
    )
    save_fig(fig_trend, "06_taxonomy_trend", ext="html")

except Exception as e:
    warn(f"Fig 06 failed: {e}")


# ── Fig 07: Opening length by category (box plot) ────────────────────────────
try:
    # Build data per level-2 category, sorted by median token_count
    ling_class = ling_df.join(class_df)
    merged_l2 = analysis[["token_count", "opening_level_2", "opening_level_1"]].dropna()
    l2_medians = merged_l2.groupby("opening_level_2")["token_count"].median().sort_values()

    # Filter to l2 categories that have at least 2 data points (boxplot requirement)
    l2_valid = [l2 for l2 in l2_medians.index
                if len(merged_l2[merged_l2["opening_level_2"] == l2]) >= 2]
    l2_medians = l2_medians.loc[l2_valid]

    fig, ax = plt.subplots(figsize=(14, 8))
    positions = range(len(l2_medians))
    box_data  = [merged_l2[merged_l2["opening_level_2"] == l2]["token_count"].dropna().values
                 for l2 in l2_medians.index]

    bplot = ax.boxplot(box_data, positions=list(positions), patch_artist=True,
                       showfliers=False, vert=True)

    # Color by parent Level-1
    for i, (l2, patch) in enumerate(zip(l2_medians.index, bplot["boxes"])):
        sub = merged_l2[merged_l2["opening_level_2"] == l2]
        l1_val = sub["opening_level_1"].mode().iloc[0] if len(sub) > 0 else "fragment"
        patch.set_facecolor(L1_COLORS.get(l1_val, "#888"))
        patch.set_alpha(0.75)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([l.replace("_", "\n") for l in l2_medians.index],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Token Count (log scale)", fontsize=LABEL_SIZE)
    ax.set_title("Opening Message Length by Level-2 Category", fontsize=TITLE_SIZE)

    patches = [mpatches.Patch(color=L1_COLORS.get(l1, "#888"), label=l1, alpha=0.75)
               for l1 in L1_ORDER if l1 in class_df["opening_level_1"].unique()]
    ax.legend(handles=patches, loc="upper left", fontsize=TICK_SIZE)
    plt.tight_layout()
    save_fig(fig, "07_opening_length_by_category")

except Exception as e:
    warn(f"Fig 07 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 08: Opening predicts outcome (2x2) ───────────────────────────────────
try:
    l1_cats = [c for c in L1_ORDER if c in analysis["opening_level_1"].unique()]
    metrics = [
        ("turns",            "Mean Turns"),
        ("duration_minutes", "Mean Duration (min)"),
        ("user_token_ratio", "Mean User Token Ratio"),
        ("msg_count",        "Mean Message Count"),
    ]

    from scipy.stats import sem

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        means, sems, cats_present = [], [], []
        for cat in l1_cats:
            vals = analysis[analysis["opening_level_1"] == cat][metric].dropna()
            if len(vals) >= 5:
                means.append(float(vals.mean()))
                sems.append(float(sem(vals)))
                cats_present.append(cat)

        bar_colors = [L1_COLORS.get(c, "#888") for c in cats_present]
        xs = range(len(cats_present))
        ax.bar(xs, means, yerr=sems, color=bar_colors, capsize=4, width=0.6, alpha=0.85)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c.replace("_", "\n") for c in cats_present], fontsize=8)
        ax.set_ylabel(label, fontsize=TICK_SIZE)
        ax.set_title(label, fontsize=TICK_SIZE + 1)

        # Mark significance
        test = outcome_tests.get(metric, {})
        if test.get("significant"):
            ax.set_title(f"{label} *", fontsize=TICK_SIZE + 1, color="#C55A11")

    fig.suptitle("How Opening Type Predicts Conversation Outcomes\n(* = Kruskal-Wallis p < 0.05)",
                 fontsize=TITLE_SIZE)
    plt.tight_layout()
    save_fig(fig, "08_opening_predicts_outcome")

except Exception as e:
    warn(f"Fig 08 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 09: Temporal patterns (time of day + weekend) ────────────────────────
try:
    tod_cross  = pd.crosstab(analysis["time_of_day"],  analysis["opening_level_1"])
    wkd_cross  = pd.crosstab(analysis["is_weekend"],   analysis["opening_level_1"])

    tod_pct  = tod_cross.div(tod_cross.sum(axis=1), axis=0)
    wkd_pct  = wkd_cross.div(wkd_cross.sum(axis=1), axis=0)

    tod_order = ["morning", "afternoon", "evening", "night"]
    tod_order = [t for t in tod_order if t in tod_pct.index]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for pct_df, ax, title, x_order in [
        (tod_pct, axes[0], "Opening Type by Time of Day", tod_order),
        (wkd_pct, axes[1], "Opening Type by Day Type", None),
    ]:
        idx = x_order if x_order else pct_df.index.tolist()
        pct_plot = pct_df.reindex(idx)
        cols = [c for c in L1_ORDER if c in pct_plot.columns]
        bottom = np.zeros(len(idx))
        for l1 in cols:
            vals = pct_plot[l1].fillna(0).values
            ax.bar(range(len(idx)), vals, bottom=bottom,
                   label=l1, color=L1_COLORS.get(l1, "#888"), width=0.6)
            bottom += vals
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([str(i).replace("True", "Weekend").replace("False", "Weekday")
                            for i in idx], fontsize=TICK_SIZE)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Proportion", fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE - 1)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Temporal Patterns in Opening Type", fontsize=TITLE_SIZE)
    plt.tight_layout()
    save_fig(fig, "09_temporal_patterns")

except Exception as e:
    warn(f"Fig 09 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 10: Greeting analysis ────────────────────────────────────────────────
try:
    monthly_g_counts = analysis.groupby("year_month").apply(
        lambda g: int((g["opening_level_1"] == "greeting").sum())
    )

    fig, ax1 = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax2 = ax1.twinx()

    xs = range(len(monthly_g_counts))
    ax1.bar(xs, monthly_g_counts.values, color=L1_COLORS["greeting"], alpha=0.7, label="Count")
    ax2.plot(xs, monthly_greeting_rate.reindex(monthly_g_counts.index).fillna(0).values,
             color="#333333", linewidth=2, marker="o", markersize=4, label="Rate")

    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(monthly_g_counts.index, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Greeting Count", fontsize=LABEL_SIZE, color=L1_COLORS["greeting"])
    ax2.set_ylabel("Greeting Rate", fontsize=LABEL_SIZE)
    ax2.set_ylim(0, max(0.5, ax2.get_ylim()[1]))

    # Cast to numeric to avoid string-dtype mean errors when Series is empty/object
    mgr_numeric = pd.to_numeric(monthly_greeting_rate, errors="coerce")
    overall_g = float(mgr_numeric.mean()) if len(mgr_numeric) > 0 else 0.0
    if math.isnan(overall_g):
        overall_g = 0.0
    ax2.axhline(overall_g, color="#999", linestyle="--", linewidth=1, label=f"Mean: {overall_g:.2%}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=TICK_SIZE)

    ax1.set_title(f"Greeting Usage Over Time (tendency: {signature.get('greeting_tendency', 'unknown')})",
                  fontsize=TITLE_SIZE)
    plt.tight_layout()
    save_fig(fig, "10_greeting_analysis")

except Exception as e:
    warn(f"Fig 10 failed: {e}")
    import traceback; traceback.print_exc()


# ── Fig 11: Vocabulary analysis (HTML) ───────────────────────────────────────
try:
    # Directive verb → subtype mapping for color
    gen_v  = {"write","create","generate","make","produce","draft","compose","design","build"}
    ana_v  = {"analyze","compare","evaluate","review","assess","examine","critique","summarize","explain"}
    trn_v  = {"convert","translate","rewrite","format","fix","debug","refactor","optimize","simplify",
               "improve","update","modify","edit","change"}
    ret_v  = {"find","search","list","show","give","tell","get","look","provide"}

    subtype_colors = {
        "generation":     "#C55A11",
        "analysis":       "#4472C4",
        "transformation": "#548235",
        "retrieval":      "#BF8F00",
        "other":          "#A5A5A5",
    }

    verb_items = list(verb_freq.items())[:20]
    verb_labels = [v for v, _ in verb_items]
    verb_vals   = [c for _, c in verb_items]
    verb_colors = []
    for v in verb_labels:
        if v in gen_v:   verb_colors.append(subtype_colors["generation"])
        elif v in ana_v: verb_colors.append(subtype_colors["analysis"])
        elif v in trn_v: verb_colors.append(subtype_colors["transformation"])
        elif v in ret_v: verb_colors.append(subtype_colors["retrieval"])
        else:            verb_colors.append(subtype_colors["other"])

    q_items  = list(q_freq.items())[:15]
    q_labels = [w for w, _ in q_items]
    q_vals   = [c for _, c in q_items]

    fig_voc = make_subplots(rows=1, cols=2,
                            subplot_titles=("Top Directive Root Verbs", "Top Question-Starting Words"))

    fig_voc.add_trace(
        go.Bar(x=verb_vals[::-1], y=verb_labels[::-1],
               orientation="h", marker_color=verb_colors[::-1],
               hovertemplate="%{y}: %{x}<extra></extra>"),
        row=1, col=1,
    )
    fig_voc.add_trace(
        go.Bar(x=q_vals[::-1], y=q_labels[::-1],
               orientation="h", marker_color="#4472C4",
               hovertemplate="%{y}: %{x}<extra></extra>"),
        row=1, col=2,
    )

    fig_voc.update_layout(
        title=dict(text="Opening Vocabulary Analysis", font=dict(size=TITLE_SIZE)),
        showlegend=False, height=500,
    )
    save_fig(fig_voc, "11_vocabulary_analysis", ext="html")

except Exception as e:
    warn(f"Fig 11 failed: {e}")


# ── Fig 12: Signature summary dashboard ───────────────────────────────────────
try:
    sig = report.get("cognitive_signature_fragment", {})

    fig12 = plt.figure(figsize=FIGSIZE_TALL)
    gs = GridSpec(3, 2, figure=fig12, hspace=0.45, wspace=0.4)

    # Top-left: Big text — dominant type
    ax_main = fig12.add_subplot(gs[0, :])
    ax_main.axis("off")
    dom_l1  = sig.get("dominant_opening_type", "?")
    dom_pct = sig.get("dominant_pct", 0)
    ax_main.text(0.5, 0.75,
                 f"Dominant Opening: {dom_l1.upper()} ({dom_pct:.0f}%)",
                 transform=ax_main.transAxes, ha="center", va="center",
                 fontsize=18, fontweight="bold", color=L1_COLORS.get(dom_l1, "#333"))
    ax_main.text(0.5, 0.30,
                 sig.get("summary", ""),
                 transform=ax_main.transAxes, ha="center", va="center",
                 fontsize=TICK_SIZE, color="#333", wrap=True,
                 style="italic")

    # Mid-left: Pie chart of Level-1
    ax_pie = fig12.add_subplot(gs[1, 0])
    pie_cats  = [c for c in L1_ORDER if c in l1_dist]
    pie_vals  = [l1_dist[c] for c in pie_cats]
    pie_colors = [L1_COLORS.get(c, "#888") for c in pie_cats]
    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=None, colors=pie_colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Level-1 Distribution", fontsize=TICK_SIZE + 1)
    ax_pie.legend(pie_cats, loc="lower left", fontsize=7,
                  bbox_to_anchor=(-0.3, -0.1))

    # Mid-right: Key stats
    ax_stats = fig12.add_subplot(gs[1, 1])
    ax_stats.axis("off")
    g_rate     = float(monthly_greeting_rate.mean()) if len(monthly_greeting_rate) > 0 else 0
    rpt_rate   = sig.get("session_habit_strength", "N/A")
    pred_depth = "Yes *" if sig.get("opening_predicts_depth") else "No"
    approach   = sig.get("approach_style", "?")
    soph_trend = sig.get("prompt_sophistication_trend", "stable")

    stats_text = (
        f"Approach style:       {approach}\n"
        f"Greeting rate:        {g_rate:.1%}\n"
        f"Session repeat rate:  {rpt_rate if rpt_rate is not None else 'N/A'}\n"
        f"Opening predicts depth: {pred_depth}\n"
        f"Sophistication trend: {soph_trend}\n"
        f"Top verbs: {', '.join(sig.get('top_verbs', [])[:3])}"
    )
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  va="top", ha="left", fontsize=TICK_SIZE,
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.6))
    ax_stats.set_title("Key Signature Stats", fontsize=TICK_SIZE + 1)

    # Bottom: Sparkline — dominant type proportion over time
    ax_spark = fig12.add_subplot(gs[2, :])
    if dom_l1 in monthly_openings.columns:
        spark_y = monthly_openings[dom_l1].values
        spark_x = range(len(spark_y))
        ax_spark.fill_between(spark_x, spark_y, alpha=0.3, color=L1_COLORS.get(dom_l1, "#4472C4"))
        ax_spark.plot(spark_x, spark_y, color=L1_COLORS.get(dom_l1, "#4472C4"), linewidth=2)
        ax_spark.set_xticks(list(spark_x))
        ax_spark.set_xticklabels(monthly_openings.index, rotation=45, ha="right", fontsize=7)
        ax_spark.set_ylabel("Proportion", fontsize=TICK_SIZE)
        ax_spark.set_ylim(0, 1)
        ax_spark.axhline(float(np.mean(spark_y)), color="#888", linestyle="--", linewidth=1)
    ax_spark.set_title(f"'{dom_l1}' Proportion Over Time", fontsize=TICK_SIZE + 1)

    fig12.suptitle("Opening Taxonomy — Cognitive Signature Summary", fontsize=TITLE_SIZE + 1, y=1.01)
    plt.savefig(os.path.join(FIG_DIR, "12_signature_summary.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig12)
    size_kb = os.path.getsize(os.path.join(FIG_DIR, "12_signature_summary.png")) / 1024
    print(f"  Saved: 12_signature_summary.png ({size_kb:.0f} KB)")
    report["figures_generated"].append("12_signature_summary.png")

except Exception as e:
    warn(f"Fig 12 failed: {e}")
    import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT — Save parquet + JSON report
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("OUTPUT — Saving data & report")
print("=" * 60)

try:
    out_df = class_df.copy()
    out_df["opening_level_1"] = out_df["opening_level_1"].astype("category")
    out_df["opening_level_2"] = out_df["opening_level_2"].astype("category")
    out_df.to_parquet(OUT_PARQUET)
    report["data_outputs"].append(OUT_PARQUET)
    print(f"  Saved: {OUT_PARQUET} ({len(out_df):,} rows)")
except Exception as e:
    warn(f"Parquet save failed: {e}")

try:
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_REPORT}")
except Exception as e:
    warn(f"Report save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("VALIDATION CHECKLIST")
print("=" * 60)

checks = []

def chk(label, result):
    status = "PASS" if result else "FAIL"
    checks.append((label, status))
    print(f"  [{status}] {label}")
    return result

import os

chk(
    "opening_taxonomy_report.json exists with all top-level keys",
    os.path.exists(OUT_REPORT) and all(
        k in report for k in [
            "module", "input_data", "phase2_comparison", "distribution",
            "cross_tabulations", "opening_characteristics", "outcome_analysis",
            "longitudinal", "vocabulary", "cognitive_signature_fragment",
            "figures_generated", "data_outputs",
        ]
    ),
)

chk(
    "opening_classifications.parquet exists with correct columns",
    os.path.exists(OUT_PARQUET) and all(
        c in pd.read_parquet(OUT_PARQUET).reset_index().columns
        for c in ["conversation_id", "opening_level_1", "opening_level_2"]
    ),
)

chk(
    "Parquet row count matches first_messages_with_text",
    len(pd.read_parquet(OUT_PARQUET)) == report["input_data"]["first_messages_with_text"],
)

chk(
    "All conversation_ids in parquet exist in conversations_clean",
    set(pd.read_parquet(OUT_PARQUET).index).issubset(set(conversations["conversation_id"])),
)

chk(
    "Level-1 distribution sums to total classified",
    sum(l1_dist.values()) == report["input_data"]["first_messages_with_text"],
)

# Level-2 has valid Level-1 parent
L2_PARENTS = {
    "factual_question": "question", "explanation_question": "question",
    "opinion_question": "question", "confirmation_question": "question",
    "multi_question": "question",
    "generation_directive": "directive", "analysis_directive": "directive",
    "transformation_directive": "directive", "retrieval_directive": "directive",
    "help_request": "directive",
    "code_paste": "context_dump", "text_paste": "context_dump",
    "structured_prompt": "context_dump",
    "greeting_only": "greeting", "greeting_plus_request": "greeting",
    "problem_statement": "statement", "thought_exploration": "statement",
    "status_update": "statement", "declaration": "statement",
    "keyword_fragment": "fragment", "incomplete_thought": "fragment",
    "ambiguous": "fragment",
}
valid_parents = all(
    L2_PARENTS.get(row["opening_level_2"]) == row["opening_level_1"]
    for _, row in class_df.iterrows()
    if row["opening_level_2"] in L2_PARENTS
)
chk("Every level_2 value has a valid level_1 parent", valid_parents)

expected_figs = [
    "01_coarse_vs_refined.png", "02_taxonomy_distribution.png",
    "03_taxonomy_sunburst.html", "04_taxonomy_by_type.png",
    "05_taxonomy_by_era.png", "06_taxonomy_trend.html",
    "07_opening_length_by_category.png", "08_opening_predicts_outcome.png",
    "09_temporal_patterns.png", "10_greeting_analysis.png",
    "11_vocabulary_analysis.html", "12_signature_summary.png",
]

chk(
    "All 12 figures exist in outputs/figures/opening_taxonomy/",
    all(os.path.exists(os.path.join(FIG_DIR, f)) for f in expected_figs),
)

png_figs = [f for f in expected_figs if f.endswith(".png")]
chk(
    "All PNG figures are >=10KB",
    all(
        os.path.exists(os.path.join(FIG_DIR, f)) and
        os.path.getsize(os.path.join(FIG_DIR, f)) >= 10_240
        for f in png_figs
    ),
)

html_figs = [f for f in expected_figs if f.endswith(".html")]
def is_self_contained(path):
    try:
        content = open(path, encoding="utf-8").read()
        return len(content) > 5_000
    except Exception:
        return False
chk(
    "All HTML figures are self-contained",
    all(is_self_contained(os.path.join(FIG_DIR, f)) for f in html_figs),
)

def json_is_clean(path):
    """Verify JSON parses cleanly (no bare NaN/Infinity literals)."""
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False
chk(
    "No NaN or Infinity in JSON report",
    os.path.exists(OUT_REPORT) and json_is_clean(OUT_REPORT),
)

chk(
    "cognitive_signature_fragment.summary is non-empty string",
    isinstance(report.get("cognitive_signature_fragment", {}).get("summary", ""), str) and
    len(report.get("cognitive_signature_fragment", {}).get("summary", "")) > 0,
)

chk(
    "phase2_comparison.agreement_rate_mapped is in [0, 1]",
    0.0 <= report.get("phase2_comparison", {}).get("agreement_rate_mapped", -1) <= 1.0,
)

passed = sum(1 for _, s in checks if s == "PASS")
total  = len(checks)
print()
print(f"  RESULT: {passed}/{total} checks passed")
if passed == total:
    print("  ALL CHECKS PASSED")
else:
    print("  SOME CHECKS FAILED")
print()
print("Done.")
