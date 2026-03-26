"""
Module 3.2g: Hypothesis Extraction
Script: 19_hypothesis_extraction.py

Identifies and extracts every prediction, intuition, assessment, and interpretation
the user expressed across the corpus. Uses a two-pass Batch API architecture:
  Pass 1 — detects hypothesis-bearing messages (binary classification on all
           substantive user messages).
  Pass 2 — extracts and resolves referents on flagged messages only.

Produces a timestamped hypothesis catalog with resolved pronouns/demonstratives
for manual scoring (confirmed / disconfirmed / partial / unresolved).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/19_hypothesis_extraction.py

    # Dry run (build Pass 1 requests, don't submit):
    python scripts/19_hypothesis_extraction.py --dry-run

    # Resume Pass 1:
    python scripts/19_hypothesis_extraction.py --resume-pass1-id msgbatch_xxxxx

    # Resume Pass 2 (assumes Pass 1 is complete):
    python scripts/19_hypothesis_extraction.py --resume-pass2-id msgbatch_xxxxx

    # Pass 1 only (don't run extraction):
    python scripts/19_hypothesis_extraction.py --detect-only

    # Adjust token threshold:
    python scripts/19_hypothesis_extraction.py --token-threshold 50
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import plotly.graph_objects as go
from tqdm import tqdm
import tiktoken

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH       = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
SUMMARIES_PATH  = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
INTERIM_DIR     = os.path.join(BASE, "data", "interim")

PASS1_BATCH_FILE    = os.path.join(INTERIM_DIR, "hypothesis_pass1_batch.jsonl")
PASS1_BATCH_ID_FILE = os.path.join(INTERIM_DIR, "hypothesis_pass1_batch_id.txt")
PASS2_BATCH_FILE    = os.path.join(INTERIM_DIR, "hypothesis_pass2_batch.jsonl")
PASS2_BATCH_ID_FILE = os.path.join(INTERIM_DIR, "hypothesis_pass2_batch_id.txt")

OUT_PARQUET  = os.path.join(BASE, "data", "processed", "hypotheses.parquet")
OUT_REPORT   = os.path.join(BASE, "outputs", "reports", "hypothesis_extraction_report.json")
OUT_CATALOG  = os.path.join(BASE, "outputs", "reports", "hypothesis_catalog.csv")
FIG_DIR      = os.path.join(BASE, "outputs", "figures", "hypothesis_extraction")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Model & Pricing ---------------------------------------------------------
MODEL                    = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS_PASS1  = 30
MAX_OUTPUT_TOKENS_PASS2  = 500
PRICE_INPUT_PER_MTOK     = 0.40   # $0.80/MTok * 0.5 batch discount
PRICE_OUTPUT_PER_MTOK    = 2.00   # $4.00/MTok * 0.5 batch discount

# -- Token limits ------------------------------------------------------------
DEFAULT_TOKEN_THRESHOLD  = 20
MAX_CONTEXT_TOKENS       = 1500
MAX_SUMMARY_TOKENS       = 200

# -- Valid labels ------------------------------------------------------------
VALID_TYPES       = ["prediction", "assessment", "intuition", "interpretation"]
VALID_CONFIDENCE  = ["high", "moderate", "low"]
VALID_TEMPORAL    = ["immediate", "near_future", "distant_future", "unspecified"]

# -- Valid function/emotion labels (for cross-tab figures) -------------------
VALID_FUNCTIONS = [
    "interpersonal_analysis", "emotional_processing", "creative_expression",
    "career_strategy", "self_modeling", "practical", "learning",
    "problem_solving", "coding", "social_rehearsal", "work_professional",
    "planning",
]

VALID_EMOTIONS = [
    "analytical", "anxious", "curious", "frustrated", "grieving", "playful",
    "reflective", "strategic", "vulnerable", "energized", "numb", "determined",
]

# -- Style constants ---------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
DPI = 150

HYPO_COLORS = {
    "prediction":     "#4E79A7",   # blue
    "assessment":     "#59A14F",   # green
    "intuition":      "#F28E2B",   # orange
    "interpretation": "#B07AA1",   # purple
}

CONFIDENCE_COLORS = {
    "high":     "#E15759",
    "moderate": "#F28E2B",
    "low":      "#76B7B2",
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

# -- System Prompts ----------------------------------------------------------
SYSTEM_PROMPT_PASS1 = "\n".join([
    "You are a research assistant identifying messages that contain predictions, assessments, intuitions, or interpretations.",
    "",
    "Given a user message from a conversation with an AI assistant, determine whether the message contains a HYPOTHESIS — a statement expressing a belief about what is true, what will happen, or what something means.",
    "",
    "A hypothesis is:",
    '- A prediction: "I think X will happen", "this is going to fail", "he\'ll probably..."',
    '- An assessment: "he doesn\'t know what it does", "she got that because of...", "this enters at step 9"',
    '- An intuition: "I sense...", "something feels off", "I don\'t trust this"',
    '- An interpretation: "this is about control not security", "he\'s avoiding me because...", "the real reason is..."',
    "",
    "A hypothesis is NOT:",
    '- A question: "what do you think?", "should I do this?"',
    '- A directive: "analyze this", "tell me about..."',
    '- A confirmation: "okay", "got it", "yes exactly"',
    '- A factual statement about oneself: "I went to the store", "I\'m tired"',
    '- A request: "can you help me with..."',
    "- A description of past events without interpretive framing",
    "",
    "Respond with ONLY a JSON object:",
    '{"has_hypothesis": true/false, "confidence": <0.0-1.0>}',
])

SYSTEM_PROMPT_PASS2 = "\n".join([
    "You are a research assistant extracting and resolving hypotheses from conversation messages.",
    "",
    "Given a user message and its surrounding conversation context, extract every prediction, assessment, intuition, or interpretation the user expressed. For each hypothesis, resolve all pronouns and demonstratives into bracketed descriptions so the hypothesis is understandable without context.",
    "",
    "Types:",
    '- prediction: Something will happen. Future-oriented. "I think [X] will..."',
    '- assessment: Something is true now. Present-oriented factual claim. "[X] doesn\'t know..."',
    '- intuition: Something feels true. Pre-verbal signal. "I sense [X]..."',
    '- interpretation: Assigning meaning to an event. Explanatory frame. "[X] is about..."',
    "",
    "Referent resolution rules:",
    "- Replace pronouns (he, she, they, him, her, them) with bracketed role descriptions: [the senior director], [the security team lead], [the colleague]. Do NOT use actual names. Derive descriptions from context.",
    "- Replace demonstratives (this, that, these, those) with bracketed descriptions of what they refer to.",
    "- Replace implied subjects with bracketed descriptions.",
    "- If a referent is ambiguous, use a bracketed description with a question mark: [the colleague?]",
    "",
    "Confidence language:",
    '- high: Direct assertions with no hedging. "He doesn\'t know." "This will fail."',
    '- moderate: Hedged with "I think", "probably", "likely", "seems like"',
    '- low: Strongly hedged with "I sense", "something feels", "I\'m not sure but", "maybe"',
    "",
    "Temporal scope (for predictions only):",
    "- immediate: Within hours/days",
    "- near_future: Within weeks/months",
    "- distant_future: Months to years out",
    "- unspecified: No clear timeframe",
    "",
    "Respond with ONLY a JSON object:",
    "{",
    '  "hypotheses": [',
    "    {",
    '      "type": "prediction|assessment|intuition|interpretation",',
    '      "raw_text": "the exact text from the user\'s message containing the hypothesis",',
    '      "resolved_text": "the hypothesis with all pronouns and demonstratives resolved into bracketed descriptions",',
    '      "topic": "brief topic label (2-5 words)",',
    '      "confidence_language": "the specific hedging words used, or \'direct assertion\'",',
    '      "confidence_level": "high|moderate|low",',
    '      "temporal_scope": "immediate|near_future|distant_future|unspecified|null"',
    "    }",
    "  ]",
    "}",
    "",
    "If the message was flagged but on closer inspection contains no extractable hypothesis, return:",
    '{"hypotheses": []}',
])


# -- Helper functions ---------------------------------------------------------
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


def _strip_json_fences(text):
    """Extract JSON from markdown code fences (Haiku often adds trailing explanation)."""
    text = text.strip()
    # Try to extract content between ```json ... ``` fences first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip leading/trailing fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    return text.strip()


# -- Step 0: Load Data -------------------------------------------------------
def load_data(token_threshold):
    print("\n== Step 0: Load data ===================================================")

    for path, label in [
        (CONV_PATH,      "conversations_clean.parquet"),
        (MSGS_PATH,      "messages_clean.parquet"),
        (SUMMARIES_PATH, "all_summaries.csv"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    conversations = pd.read_parquet(CONV_PATH)
    messages      = pd.read_parquet(MSGS_PATH)
    summaries_raw = pd.read_csv(SUMMARIES_PATH)

    conv = conversations[conversations["is_analysable"]].copy()
    analysable_ids = set(conv["conversation_id"])
    msgs = messages[messages["conversation_id"].isin(analysable_ids)].copy()

    # Filter to user messages with >= token_threshold tokens
    enc = tiktoken.get_encoding("cl100k_base")
    user_msgs = msgs[msgs["role"] == "user"].copy()

    # Use token_count column if available, otherwise estimate
    if "token_count" in user_msgs.columns:
        substantive = user_msgs[user_msgs["token_count"] >= token_threshold].copy()
    else:
        # Estimate tokens
        user_msgs["_est_tokens"] = user_msgs["text"].apply(
            lambda t: len(enc.encode(str(t))) if pd.notna(t) else 0
        )
        substantive = user_msgs[user_msgs["_est_tokens"] >= token_threshold].copy()

    print(f"  Analysable conversations : {len(conv):,}")
    print(f"  Messages in scope        : {len(msgs):,}")
    print(f"  User messages            : {len(user_msgs):,}")
    print(f"  Substantive (>={token_threshold} tokens): {len(substantive):,}")
    print(f"  Summaries available      : {len(summaries_raw):,}")

    # Build summary map
    sum_map = {}
    for _, row in summaries_raw.iterrows():
        cid = row.get("conversation_id")
        s   = row.get("summary", "")
        if pd.notna(cid) and pd.notna(s):
            sum_map[str(cid)] = str(s)

    # Load optional cross-tab data
    func_class = None
    if os.path.exists(FUNC_CLASS_PATH):
        func_class = pd.read_parquet(FUNC_CLASS_PATH)
        print(f"  Functional classifications: {len(func_class):,}")
    else:
        print(f"  Functional classifications: NOT FOUND (cross-tab figures will be skipped)")

    emot_class = None
    if os.path.exists(EMOT_PATH):
        emot_class = pd.read_parquet(EMOT_PATH)
        print(f"  Emotional states         : {len(emot_class):,}")
    else:
        print(f"  Emotional states         : NOT FOUND (cross-tab figures will be skipped)")

    return conv, msgs, substantive, config, sum_map, func_class, emot_class


# -- Build Pass 1 inputs -----------------------------------------------------
def build_pass1_inputs(substantive):
    """Build a dict of custom_id -> message info for Pass 1 detection."""
    print("\n== Building Pass 1 inputs ==============================================")
    inputs = {}
    for _, row in tqdm(substantive.iterrows(), total=len(substantive), desc="Building Pass 1"):
        cid     = row["conversation_id"]
        msg_idx = int(row["msg_index"])
        text    = str(row.get("text", "") or "")
        custom_id = f"{cid}__msg{msg_idx}"
        inputs[custom_id] = {
            "conversation_id": cid,
            "message_index":   msg_idx,
            "text":            text,
        }
    print(f"  Pass 1 inputs built      : {len(inputs):,}")
    return inputs


def _format_pass1_message(inp):
    return (
        "Does this message contain a prediction, assessment, intuition, or interpretation?\n\n"
        f"Message:\n{inp['text']}"
    )


# -- Build batch requests (generic) ------------------------------------------
def build_batch_requests(inputs, batch_file, system_prompt, max_tokens, format_fn):
    print(f"\n  Building batch: {os.path.basename(batch_file)}")
    with open(batch_file, "w", encoding="utf-8") as f:
        for custom_id, inp in inputs.items():
            request = {
                "custom_id": custom_id,
                "params": {
                    "model":      MODEL,
                    "max_tokens": max_tokens,
                    "system":     system_prompt,
                    "messages":   [{"role": "user", "content": format_fn(inp)}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    count = sum(1 for _ in open(batch_file, encoding="utf-8"))
    print(f"  Batch file               : {batch_file}")
    print(f"  Total requests           : {count:,}")
    return count


# -- Submit batch -------------------------------------------------------------
def submit_batch(client, batch_file, batch_id_file, pass_label):
    print(f"\n== Submit {pass_label} batch ============================================")
    requests = []
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            requests.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests)
    print(f"  ============================================")
    print(f"  BATCH SUBMITTED: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ============================================")
    print(f"  Save this ID to resume: --resume-{pass_label.lower().replace(' ', '-')}-id {batch.id}")

    with open(batch_id_file, "w") as f:
        f.write(batch.id)
    return batch.id


# -- Poll batch ---------------------------------------------------------------
def poll_batch(client, batch_id, pass_label):
    print(f"\n== Poll {pass_label} batch {batch_id} ==")
    print(f"  (Checking every 60 seconds. This may take 15-90 minutes.)\n")
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        total  = (
            counts.processing + counts.succeeded
            + counts.errored + counts.canceled + counts.expired
        )
        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"Status: {status.processing_status} | "
            f"Succeeded: {counts.succeeded:,} | "
            f"Errored: {counts.errored:,} | "
            f"Processing: {counts.processing:,} | "
            f"Total: {total:,}"
        )
        if status.processing_status == "ended":
            print(f"\n  {pass_label} batch complete!")
            return status
        time.sleep(60)


# -- Retrieve results ---------------------------------------------------------
def retrieve_results(client, batch_id, pass_label):
    print(f"\n== Retrieve {pass_label} results ========================================")
    results = {}
    errors  = []

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            response = result.result.message
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            results[cid] = {
                "raw":           raw_text.strip(),
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            errors.append({
                "custom_id":  cid,
                "error_type": result.result.type,
                "error":      str(result.result.error)
                              if hasattr(result.result, "error") else "unknown",
            })

    print(f"  Batch succeeded          : {len(results):,}")
    print(f"  Batch errored            : {len(errors):,}")
    return results, errors


# -- Retry errors via standard API --------------------------------------------
def retry_errors(client, errors, inputs, system_prompt, max_tokens, format_fn):
    if not errors:
        return {}
    print(f"\n  Retrying {len(errors)} failure(s) via standard API...")
    retried = {}
    for err in tqdm(errors, desc="Retrying"):
        custom_id = err["custom_id"]
        inp = inputs.get(custom_id)
        if not inp:
            continue
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": format_fn(inp)}],
            )
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            retried[custom_id] = {
                "raw":           raw_text.strip(),
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Retry failed for {custom_id}: {e}")
        time.sleep(1)
    print(f"  Retries succeeded        : {len(retried):,}")
    return retried


# -- Parse Pass 1 results ----------------------------------------------------
def parse_pass1(results):
    """Parse Pass 1 detection results. Returns dict of custom_id -> {has_hypothesis, confidence}."""
    print("\n  Parsing Pass 1 detection responses...")
    parsed       = {}
    parse_errors = 0

    for custom_id, res in tqdm(results.items(), desc="Parsing Pass 1"):
        try:
            text = _strip_json_fences(res.get("raw", ""))
            # Handle true/false (not valid JSON booleans when unquoted in this context)
            text = text.replace(": true", ": true").replace(": false", ": false")
            data = json.loads(text)
            has_hypo = bool(data.get("has_hypothesis", False))
            conf     = float(data.get("confidence", 0.0))
            conf     = max(0.0, min(1.0, conf))
            parsed[custom_id] = {
                "has_hypothesis": has_hypo,
                "confidence":     conf,
                "input_tokens":   res.get("input_tokens", 0),
                "output_tokens":  res.get("output_tokens", 0),
            }
        except Exception as e:
            parse_errors += 1
            parsed[custom_id] = {
                "has_hypothesis": False,
                "confidence":     0.0,
                "input_tokens":   res.get("input_tokens", 0),
                "output_tokens":  res.get("output_tokens", 0),
            }

    flagged = sum(1 for v in parsed.values() if v["has_hypothesis"])
    total   = len(parsed)
    rate    = flagged / max(total, 1) * 100
    print(f"  Parse errors             : {parse_errors:,}")
    print(f"  Total parsed             : {total:,}")
    print(f"  Flagged (has_hypothesis) : {flagged:,} ({rate:.1f}%)")
    return parsed, parse_errors


# -- Build Pass 2 inputs -----------------------------------------------------
def build_pass2_inputs(flagged_ids, pass1_inputs, msgs, conv, sum_map):
    """Build Pass 2 extraction inputs with context window for flagged messages."""
    print("\n== Building Pass 2 inputs ==============================================")
    enc = tiktoken.get_encoding("cl100k_base")

    # Pre-sort messages by conversation and index
    msgs_sorted = msgs.sort_values(["conversation_id", "msg_index"])
    conv_msgs = {}
    for _, row in msgs_sorted.iterrows():
        cid = row["conversation_id"]
        if cid not in conv_msgs:
            conv_msgs[cid] = []
        conv_msgs[cid].append({
            "msg_index": row["msg_index"],
            "role":      row["role"],
            "text":      str(row.get("text", "") or ""),
        })

    # Build title map
    title_map = {}
    for _, row in conv.iterrows():
        title_map[row["conversation_id"]] = str(row.get("title", "") or "")

    inputs = {}
    for custom_id in tqdm(flagged_ids, desc="Building Pass 2"):
        inp = pass1_inputs.get(custom_id)
        if not inp:
            continue

        cid     = inp["conversation_id"]
        msg_idx = inp["message_index"]
        text    = inp["text"]
        title   = title_map.get(cid, "")

        # Get summary, truncated to MAX_SUMMARY_TOKENS
        summary = sum_map.get(str(cid), "")
        if summary:
            try:
                sum_tokens = enc.encode(summary)
                if len(sum_tokens) > MAX_SUMMARY_TOKENS:
                    summary = enc.decode(sum_tokens[:MAX_SUMMARY_TOKENS])
            except Exception:
                summary = summary[:800]

        # Build context window: preceding 3-4 exchanges, truncated to MAX_CONTEXT_TOKENS
        context_lines = []
        context_tokens_used = 0
        if cid in conv_msgs:
            preceding = [m for m in conv_msgs[cid] if m["msg_index"] < msg_idx]
            # Take last 8 messages (up to 4 exchanges of 2 messages each)
            preceding = preceding[-8:]
            for m in reversed(preceding):
                m_text = m["text"]
                try:
                    m_toks = len(enc.encode(m_text))
                except Exception:
                    m_toks = len(m_text) // 4
                if context_tokens_used + m_toks > MAX_CONTEXT_TOKENS:
                    # Truncate this message to fit
                    remaining = MAX_CONTEXT_TOKENS - context_tokens_used
                    if remaining > 20:
                        try:
                            m_text = enc.decode(enc.encode(m_text)[:remaining])
                        except Exception:
                            m_text = m_text[:remaining * 4]
                        context_lines.insert(0, f"{m['role'].title()}: {m_text}")
                    break
                context_lines.insert(0, f"{m['role'].title()}: {m_text}")
                context_tokens_used += m_toks

        context_str = "\n\n".join(context_lines) if context_lines else "(no preceding context)"

        inputs[custom_id] = {
            "conversation_id": cid,
            "message_index":   msg_idx,
            "text":            text,
            "title":           title,
            "summary":         summary,
            "context":         context_str,
        }

    print(f"  Pass 2 inputs built      : {len(inputs):,}")
    return inputs


def _format_pass2_message(inp):
    return (
        "Extract and resolve hypotheses from this message.\n\n"
        f"Conversation title: {inp['title']}\n"
        f"Conversation summary: {inp['summary']}\n\n"
        f"Recent context:\n{inp['context']}\n\n"
        "---\n\n"
        f"User message to analyze:\n{inp['text']}"
    )


# -- Parse Pass 2 results ----------------------------------------------------
def parse_pass2(results, pass2_inputs, pass1_parsed):
    """Parse Pass 2 extraction results. Returns list of hypothesis rows."""
    print("\n  Parsing Pass 2 extraction responses...")
    rows              = []
    parse_errors      = 0
    empty_extractions = 0
    multi_hypo_msgs   = 0

    for custom_id, res in tqdm(results.items(), desc="Parsing Pass 2"):
        inp = pass2_inputs.get(custom_id, {})
        cid     = inp.get("conversation_id", "")
        msg_idx = inp.get("message_index", 0)
        p1      = pass1_parsed.get(custom_id, {})

        try:
            text = _strip_json_fences(res.get("raw", ""))
            data = json.loads(text)
            hypotheses = data.get("hypotheses", [])

            if not isinstance(hypotheses, list):
                hypotheses = []

            if len(hypotheses) == 0:
                empty_extractions += 1
                continue

            if len(hypotheses) > 1:
                multi_hypo_msgs += 1

            for h_idx, h in enumerate(hypotheses):
                h_type  = str(h.get("type", "")).strip()
                raw_txt = str(h.get("raw_text", "")).strip()
                res_txt = str(h.get("resolved_text", "")).strip()
                topic   = str(h.get("topic", "")).strip()
                conf_lang  = str(h.get("confidence_language", "")).strip()
                conf_level = str(h.get("confidence_level", "")).strip()
                temp_scope = h.get("temporal_scope")

                # Validate type
                if h_type not in VALID_TYPES:
                    parse_errors += 1
                    continue

                # Validate confidence level
                if conf_level not in VALID_CONFIDENCE:
                    conf_level = "moderate"  # default

                # Validate temporal scope
                if temp_scope is not None:
                    temp_scope = str(temp_scope).strip()
                    if temp_scope in ("null", "None", ""):
                        temp_scope = None
                    elif temp_scope not in VALID_TEMPORAL:
                        temp_scope = "unspecified"

                # Only predictions get temporal scope
                if h_type != "prediction":
                    temp_scope = None

                rows.append({
                    "hypothesis_id":        f"{cid}_{msg_idx}_{h_idx}",
                    "conversation_id":      cid,
                    "message_index":        msg_idx,
                    "hypothesis_index":     h_idx,
                    "type":                 h_type,
                    "raw_text":             raw_txt,
                    "resolved_text":        res_txt,
                    "topic":                topic,
                    "confidence_language":  conf_lang,
                    "confidence_level":     conf_level,
                    "temporal_scope":       temp_scope,
                    "status":               "unscored",
                    "pass1_confidence":     float(p1.get("confidence", 0.0)),
                    "pass2_input_tokens":   res.get("input_tokens", 0),
                    "pass2_output_tokens":  res.get("output_tokens", 0),
                })

        except Exception as e:
            parse_errors += 1

    print(f"  Hypotheses extracted     : {len(rows):,}")
    print(f"  Empty extractions        : {empty_extractions:,}")
    print(f"  Multi-hypothesis msgs    : {multi_hypo_msgs:,}")
    print(f"  Parse errors             : {parse_errors:,}")
    return rows, parse_errors, empty_extractions, multi_hypo_msgs


# -- Save results -------------------------------------------------------------
def save_results(rows, conv, msgs, func_class, emot_class):
    print("\n== Step 5: Save results ================================================")

    if not rows:
        print("  WARNING: No hypotheses to save!")
        df = pd.DataFrame(columns=[
            "hypothesis_id", "conversation_id", "message_index", "hypothesis_index",
            "type", "raw_text", "resolved_text", "topic", "confidence_language",
            "confidence_level", "temporal_scope", "message_timestamp",
            "conversation_function", "conversation_emotion", "status",
            "pass1_confidence", "pass2_input_tokens", "pass2_output_tokens",
        ])
        df.to_parquet(OUT_PARQUET, index=False)
        df.to_csv(OUT_CATALOG, index=False)
        return df

    df = pd.DataFrame(rows)

    # Add message timestamp
    msg_ts = msgs[msgs["role"] == "user"][["conversation_id", "msg_index", "timestamp"]].copy()
    msg_ts = msg_ts.rename(columns={"msg_index": "message_index", "timestamp": "message_timestamp"})
    if "timestamp" in msgs.columns:
        df = df.merge(msg_ts, on=["conversation_id", "message_index"], how="left")
    else:
        df["message_timestamp"] = pd.NaT

    # Add functional classification
    if func_class is not None and "function_primary" in func_class.columns:
        func_map = func_class.set_index("conversation_id")["function_primary"].to_dict()
        df["conversation_function"] = df["conversation_id"].map(func_map).fillna("unknown")
    else:
        df["conversation_function"] = "unknown"

    # Add emotional state
    if emot_class is not None and "emotion_primary" in emot_class.columns:
        emot_map = emot_class.set_index("conversation_id")["emotion_primary"].to_dict()
        df["conversation_emotion"] = df["conversation_id"].map(emot_map).fillna("unknown")
    else:
        df["conversation_emotion"] = "unknown"

    # Set dtypes
    df["type"]                  = df["type"].astype("category")
    df["confidence_level"]      = df["confidence_level"].astype("category")
    df["temporal_scope"]        = df["temporal_scope"].astype("category")
    df["conversation_function"] = df["conversation_function"].astype("category")
    df["conversation_emotion"]  = df["conversation_emotion"].astype("category")
    df["status"]                = df["status"].astype("category")
    df["pass1_confidence"]      = df["pass1_confidence"].astype("float32")
    df["pass2_input_tokens"]    = df["pass2_input_tokens"].astype("int32")
    df["pass2_output_tokens"]   = df["pass2_output_tokens"].astype("int32")
    df["message_index"]         = df["message_index"].astype("int32")
    df["hypothesis_index"]      = df["hypothesis_index"].astype("int32")

    # Sort
    df = df.sort_values(["conversation_id", "message_index", "hypothesis_index"]).reset_index(drop=True)

    # Save parquet
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Rows:  {len(df):,}")

    # Save human-reviewable catalog CSV
    catalog_cols = [
        "message_timestamp", "type", "confidence_level", "resolved_text",
        "topic", "conversation_function", "conversation_emotion", "status",
    ]
    catalog = df[[c for c in catalog_cols if c in df.columns]].copy()
    if "message_timestamp" in catalog.columns:
        catalog = catalog.rename(columns={"message_timestamp": "date"})
    catalog.to_csv(OUT_CATALOG, index=False, encoding="utf-8")
    print(f"  Saved: {OUT_CATALOG}")

    return df


# -- Figures ------------------------------------------------------------------
def make_figures(df, conv, config, func_class=None, emot_class=None):
    print("\n== Step 6: Generating figures ==========================================")

    if len(df) == 0:
        print("  No hypotheses — creating placeholder figures only")
        for name in [
            "hypothesis_type_distribution.png", "hypotheses_over_time.png",
            "confidence_level_distribution.png", "hypotheses_by_function.png",
            "hypotheses_by_emotion.png", "temporal_scope_distribution.png",
            "confidence_language_frequency.png", "hypothesis_dashboard.png",
        ]:
            _placeholder_figure(name, "No hypotheses extracted")
        return

    _fig1_type_distribution(df)
    _fig2_over_time(df, conv, config)
    _fig3_confidence_distribution(df)
    _fig4_by_function(df, func_class)
    _fig5_by_emotion(df, emot_class)
    _fig6_temporal_scope(df)
    _fig7_confidence_language(df)
    _fig8_dashboard(df, conv, func_class, emot_class)
    print("  All figures saved.")


def _fig1_type_distribution(df):
    """Horizontal bar chart of the 4 hypothesis types."""
    counts = df["type"].astype(str).value_counts()
    order  = [t for t in VALID_TYPES if t in counts.index]
    counts = counts.reindex(order).dropna()
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [HYPO_COLORS.get(t, COLOR_PRIMARY) for t in counts.index]
    bars    = ax.barh(counts.index, counts.values, color=colors)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of Hypotheses")
    ax.set_title("Hypothesis Type Distribution")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("hypothesis_type_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: hypothesis_type_distribution.png")


def _fig2_over_time(df, conv, config):
    """Line chart: hypotheses per month, separate lines per type."""
    merged = df.merge(
        conv[["conversation_id", "year_month"]],
        on="conversation_id", how="left",
    )
    if "year_month" not in merged.columns or merged["year_month"].isna().all():
        _placeholder_figure("hypotheses_over_time.png", "No year_month data")
        return

    ct = pd.crosstab(merged["year_month"].astype(str), merged["type"].astype(str))
    ct = ct.reindex(columns=[t for t in VALID_TYPES if t in ct.columns])
    ct = ct.sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for h_type in ct.columns:
        ax.plot(range(len(ct)), ct[h_type].values,
                marker="o", markersize=3, linewidth=1.5,
                label=h_type, color=HYPO_COLORS.get(h_type, COLOR_PRIMARY))

    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(ct.index, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Hypothesis Count")
    ax.set_title("Hypotheses Over Time by Type")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("hypotheses_over_time.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: hypotheses_over_time.png")


def _fig3_confidence_distribution(df):
    """Stacked bar chart: hypothesis type x confidence level."""
    ct = pd.crosstab(df["type"].astype(str), df["confidence_level"].astype(str))
    types = [t for t in VALID_TYPES if t in ct.index]
    levels = [l for l in VALID_CONFIDENCE if l in ct.columns]
    ct = ct.reindex(index=types, columns=levels, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(types))
    bottom = np.zeros(len(types))

    for level in levels:
        vals = ct[level].values
        ax.bar(x, vals, bottom=bottom,
               color=CONFIDENCE_COLORS.get(level, COLOR_PRIMARY),
               label=level, width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_xlabel("Hypothesis Type")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Level Distribution by Hypothesis Type")
    ax.legend(title="Confidence")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("confidence_level_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: confidence_level_distribution.png")


def _fig4_by_function(df, func_class):
    """Heatmap: functional classification x hypothesis type."""
    if func_class is None or "conversation_function" not in df.columns:
        _placeholder_figure("hypotheses_by_function.png", "Functional classifications not available")
        return

    valid = df[df["conversation_function"].astype(str) != "unknown"]
    if len(valid) == 0:
        _placeholder_figure("hypotheses_by_function.png", "No matching data")
        return

    ct   = pd.crosstab(valid["conversation_function"].astype(str),
                       valid["type"].astype(str))
    rows = [f for f in VALID_FUNCTIONS if f in ct.index]
    cols = [t for t in VALID_TYPES if t in ct.columns]
    ct   = ct.reindex(index=rows, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("_", " ") for r in rows], fontsize=8)

    vmax = ct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(ct.values[i, j])
            if val > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_title("Hypotheses by Functional Category")
    ax.set_xlabel("Hypothesis Type")
    ax.set_ylabel("Functional Category")
    plt.tight_layout()
    plt.savefig(figpath("hypotheses_by_function.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: hypotheses_by_function.png")


def _fig5_by_emotion(df, emot_class):
    """Heatmap: emotional state x hypothesis type."""
    if emot_class is None or "conversation_emotion" not in df.columns:
        _placeholder_figure("hypotheses_by_emotion.png", "Emotional states not available")
        return

    valid = df[df["conversation_emotion"].astype(str) != "unknown"]
    if len(valid) == 0:
        _placeholder_figure("hypotheses_by_emotion.png", "No matching data")
        return

    ct   = pd.crosstab(valid["conversation_emotion"].astype(str),
                       valid["type"].astype(str))
    rows = [e for e in VALID_EMOTIONS if e in ct.index]
    cols = [t for t in VALID_TYPES if t in ct.columns]
    ct   = ct.reindex(index=rows, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ct.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    vmax = ct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(ct.values[i, j])
            if val > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_title("Hypotheses by Emotional State")
    ax.set_xlabel("Hypothesis Type")
    ax.set_ylabel("Emotional State")
    plt.tight_layout()
    plt.savefig(figpath("hypotheses_by_emotion.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: hypotheses_by_emotion.png")


def _fig6_temporal_scope(df):
    """Pie chart: temporal scope distribution for predictions only."""
    preds = df[df["type"].astype(str) == "prediction"]
    if len(preds) == 0:
        _placeholder_figure("temporal_scope_distribution.png", "No predictions found")
        return

    scope_counts = preds["temporal_scope"].astype(str).value_counts()
    # Filter to valid scopes
    labels = [s for s in VALID_TEMPORAL if s in scope_counts.index]
    sizes  = [int(scope_counts.get(s, 0)) for s in labels]

    if not sizes or sum(sizes) == 0:
        _placeholder_figure("temporal_scope_distribution.png", "No temporal scope data")
        return

    scope_colors = {
        "immediate":      "#E15759",
        "near_future":    "#F28E2B",
        "distant_future": "#76B7B2",
        "unspecified":    "#BAB0AC",
    }
    colors = [scope_colors.get(s, COLOR_PRIMARY) for s in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[s.replace("_", " ") for s in labels],
        colors=colors, autopct="%1.1f%%", startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Temporal Scope Distribution (Predictions Only)")
    plt.tight_layout()
    plt.savefig(figpath("temporal_scope_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: temporal_scope_distribution.png")


def _fig7_confidence_language(df):
    """Bar chart of most common confidence language phrases."""
    lang_counts = df["confidence_language"].astype(str).value_counts().head(15)

    if len(lang_counts) == 0:
        _placeholder_figure("confidence_language_frequency.png", "No confidence language data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(lang_counts.index[::-1], lang_counts.values[::-1], color=COLOR_PRIMARY)

    for bar, count in zip(bars, lang_counts.values[::-1]):
        ax.text(bar.get_width() + lang_counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    ax.set_xlabel("Frequency")
    ax.set_title("Most Common Confidence Language Phrases")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("confidence_language_frequency.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: confidence_language_frequency.png")


def _fig8_dashboard(df, conv, func_class=None, emot_class=None):
    """Summary dashboard panel."""
    total = len(df)

    # Type distribution
    type_counts = df["type"].astype(str).value_counts()
    dominant_type = str(type_counts.index[0]) if len(type_counts) > 0 else "N/A"
    dominant_pct  = round(type_counts.iloc[0] / max(total, 1) * 100, 1) if len(type_counts) > 0 else 0

    # Confidence
    conf_counts = df["confidence_level"].astype(str).value_counts()
    dominant_conf = str(conf_counts.index[0]) if len(conf_counts) > 0 else "N/A"

    # Most hypothesis-dense function
    dense_func = "N/A"
    if "conversation_function" in df.columns:
        func_valid = df[df["conversation_function"].astype(str) != "unknown"]
        if len(func_valid) > 0:
            dense_func = str(func_valid["conversation_function"].astype(str).value_counts().index[0])

    # Most hypothesis-dense emotion
    dense_emot = "N/A"
    if "conversation_emotion" in df.columns:
        emot_valid = df[df["conversation_emotion"].astype(str) != "unknown"]
        if len(emot_valid) > 0:
            dense_emot = str(emot_valid["conversation_emotion"].astype(str).value_counts().index[0])

    # Monthly average
    merged = df.merge(conv[["conversation_id", "year_month"]], on="conversation_id", how="left")
    monthly = merged.groupby(merged["year_month"].astype(str)).size()
    avg_per_month = round(monthly.mean(), 1) if len(monthly) > 0 else 0

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # Panel A: total hypotheses
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.text(0.5, 0.6, f"{total:,}", ha="center", va="center",
              fontsize=28, fontweight="bold", color=COLOR_PRIMARY, transform=ax_a.transAxes)
    ax_a.text(0.5, 0.25, "total hypotheses", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_a.transAxes)
    ax_a.set_title("Total Extracted", fontsize=11)

    # Panel B: type distribution (mini pie)
    ax_b = fig.add_subplot(gs[0, 1])
    if len(type_counts) > 0:
        pie_types  = [t for t in VALID_TYPES if t in type_counts.index]
        pie_sizes  = [int(type_counts.get(t, 0)) for t in pie_types]
        pie_colors = [HYPO_COLORS.get(t, COLOR_PRIMARY) for t in pie_types]
        ax_b.pie(pie_sizes, labels=pie_types, colors=pie_colors,
                 autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    ax_b.set_title("Type Distribution", fontsize=11)

    # Panel C: dominant confidence
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.text(0.5, 0.6, dominant_conf, ha="center", va="center",
              fontsize=20, fontweight="bold",
              color=CONFIDENCE_COLORS.get(dominant_conf, COLOR_PRIMARY),
              transform=ax_c.transAxes)
    ax_c.text(0.5, 0.25, "most common\nconfidence level", ha="center", va="center",
              fontsize=9, color=COLOR_SECONDARY, transform=ax_c.transAxes)
    ax_c.set_title("Dominant Confidence", fontsize=11)

    # Panel D: per month
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.axis("off")
    ax_d.text(0.5, 0.6, f"{avg_per_month:.1f}", ha="center", va="center",
              fontsize=24, fontweight="bold", color=COLOR_PRIMARY, transform=ax_d.transAxes)
    ax_d.text(0.5, 0.25, "hypotheses/month", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_d.transAxes)
    ax_d.set_title("Monthly Average", fontsize=11)

    # Panel E: most hypothesis-dense function
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.axis("off")
    ax_e.text(0.5, 0.5, dense_func.replace("_", "\n"),
              ha="center", va="center", fontsize=9, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9"))
    ax_e.set_title("Densest Function", fontsize=10)

    # Panel F: most hypothesis-dense emotion
    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.axis("off")
    ax_f.text(0.5, 0.5, dense_emot.replace("_", "\n"),
              ha="center", va="center", fontsize=9, transform=ax_f.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0"))
    ax_f.set_title("Densest Emotion", fontsize=10)

    # Panel G: dominant type
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.axis("off")
    ax_g.text(0.5, 0.5, f"{dominant_type}\n({dominant_pct}%)",
              ha="center", va="center", fontsize=11, fontweight="bold",
              transform=ax_g.transAxes,
              color=HYPO_COLORS.get(dominant_type, COLOR_PRIMARY),
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_g.set_title("Dominant Type", fontsize=10)

    # Panel H: trend
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.axis("off")
    trend_str = "N/A"
    if len(monthly) >= 3:
        try:
            x = np.arange(len(monthly))
            slope, _, _, _, _ = stats.linregress(x, monthly.values)
            if slope > 0.1:
                trend_str = f"↑ increasing\n({slope:+.1f}/month)"
            elif slope < -0.1:
                trend_str = f"↓ decreasing\n({slope:+.1f}/month)"
            else:
                trend_str = f"→ stable\n({slope:+.1f}/month)"
        except Exception:
            pass
    ax_h.text(0.5, 0.5, trend_str, ha="center", va="center",
              fontsize=10, transform=ax_h.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_h.set_title("Hypothesis Trend", fontsize=10)

    plt.suptitle("Hypothesis Extraction Dashboard",
                 fontsize=14, fontweight="bold")
    plt.savefig(figpath("hypothesis_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: hypothesis_dashboard.png")


def _placeholder_figure(name, message):
    """Create a minimal placeholder figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=10)
    ax.set_title(name.replace(".png", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(figpath(name), dpi=DPI)
    plt.close()


# -- Report -------------------------------------------------------------------
def generate_report(df, conv, pass1_batch_id, pass2_batch_id, pass1_parsed,
                    parse_errors_p2, empty_extractions, multi_hypo_msgs,
                    warnings_list, func_class=None, emot_class=None):
    print("\n== Step 7: Generate report =============================================")

    total_hypotheses = len(df)
    total_substantive = len(pass1_parsed)
    flagged = sum(1 for v in pass1_parsed.values() if v["has_hypothesis"])
    flag_rate = round(flagged / max(total_substantive, 1) * 100, 1)

    # Type distribution
    type_dist = {}
    if total_hypotheses > 0:
        type_counts = df["type"].astype(str).value_counts()
        for t in VALID_TYPES:
            c = int(type_counts.get(t, 0))
            type_dist[t] = {"count": c, "pct": round(c / total_hypotheses * 100, 1)}
    else:
        for t in VALID_TYPES:
            type_dist[t] = {"count": 0, "pct": 0.0}

    # Confidence distribution
    conf_dist = {}
    if total_hypotheses > 0:
        conf_counts = df["confidence_level"].astype(str).value_counts()
        for c in VALID_CONFIDENCE:
            cnt = int(conf_counts.get(c, 0))
            conf_dist[c] = {"count": cnt, "pct": round(cnt / total_hypotheses * 100, 1)}
    else:
        for c in VALID_CONFIDENCE:
            conf_dist[c] = {"count": 0, "pct": 0.0}

    # Temporal scope distribution (predictions only)
    temp_dist = {}
    preds = df[df["type"].astype(str) == "prediction"] if total_hypotheses > 0 else pd.DataFrame()
    for s in VALID_TEMPORAL:
        if len(preds) > 0:
            temp_dist[s] = int((preds["temporal_scope"].astype(str) == s).sum())
        else:
            temp_dist[s] = 0

    # Referent resolution stats
    total_resolved  = 0
    ambiguous_count = 0
    if total_hypotheses > 0:
        for _, row in df.iterrows():
            rt = str(row.get("resolved_text", ""))
            brackets = re.findall(r"\[([^\]]+)\]", rt)
            total_resolved += len(brackets)
            ambiguous_count += sum(1 for b in brackets if b.endswith("?"))

    # Cross-tabulations
    cross_tabs = {}

    if func_class is not None and "conversation_function" in df.columns and total_hypotheses > 0:
        try:
            valid = df[df["conversation_function"].astype(str) != "unknown"]
            if len(valid) > 0:
                ct = pd.crosstab(valid["conversation_function"].astype(str),
                                 valid["type"].astype(str))
                cross_tabs["by_function"] = {
                    str(g): {str(t): int(v) for t, v in row.items()}
                    for g, row in ct.iterrows()
                }
        except Exception as e:
            warnings_list.append(f"Could not compute by_function cross-tab: {e}")

    if emot_class is not None and "conversation_emotion" in df.columns and total_hypotheses > 0:
        try:
            valid = df[df["conversation_emotion"].astype(str) != "unknown"]
            if len(valid) > 0:
                ct = pd.crosstab(valid["conversation_emotion"].astype(str),
                                 valid["type"].astype(str))
                cross_tabs["by_emotion"] = {
                    str(g): {str(t): int(v) for t, v in row.items()}
                    for g, row in ct.iterrows()
                }
        except Exception as e:
            warnings_list.append(f"Could not compute by_emotion cross-tab: {e}")

    if total_hypotheses > 0:
        merged = df.merge(conv[["conversation_id", "year_month"]], on="conversation_id", how="left")
        if "year_month" in merged.columns:
            ct = pd.crosstab(merged["year_month"].astype(str), merged["type"].astype(str))
            cross_tabs["by_year_month"] = {
                str(g): {str(t): int(v) for t, v in row.items()}
                for g, row in ct.iterrows()
            }

    # Confidence language frequency
    conf_lang_freq = {}
    if total_hypotheses > 0:
        lang_counts = df["confidence_language"].astype(str).value_counts().head(20)
        conf_lang_freq = {str(k): int(v) for k, v in lang_counts.items()}

    # Cost
    p1_input  = sum(v.get("input_tokens", 0) for v in pass1_parsed.values())
    p1_output = sum(v.get("output_tokens", 0) for v in pass1_parsed.values())
    p2_input  = int(df["pass2_input_tokens"].sum()) if total_hypotheses > 0 else 0
    p2_output = int(df["pass2_output_tokens"].sum()) if total_hypotheses > 0 else 0
    total_cost = round(
        (p1_input + p2_input) / 1_000_000 * PRICE_INPUT_PER_MTOK +
        (p1_output + p2_output) / 1_000_000 * PRICE_OUTPUT_PER_MTOK,
        4,
    )

    # Signature fragment
    dominant_type = ""
    dominant_conf = ""
    avg_per_month = 0.0
    dense_func = ""

    if total_hypotheses > 0:
        type_counts = df["type"].astype(str).value_counts()
        dominant_type = str(type_counts.index[0]) if len(type_counts) > 0 else ""
        conf_counts = df["confidence_level"].astype(str).value_counts()
        dominant_conf = str(conf_counts.index[0]) if len(conf_counts) > 0 else ""

        if "by_year_month" in cross_tabs:
            month_totals = {m: sum(t.values()) for m, t in cross_tabs["by_year_month"].items()}
            if month_totals:
                avg_per_month = round(sum(month_totals.values()) / len(month_totals), 1)

        if "by_function" in cross_tabs:
            func_totals = {f: sum(t.values()) for f, t in cross_tabs["by_function"].items()}
            if func_totals:
                dense_func = max(func_totals, key=func_totals.get)

    summary_text = (
        f"{total_hypotheses} hypotheses extracted from {flagged} flagged messages "
        f"({flag_rate}% flag rate). "
        f"Dominant type: {dominant_type}. "
        f"Dominant confidence: {dominant_conf}. "
        f"Average {avg_per_month} hypotheses/month."
    )

    report = {
        "module":         "hypothesis_extraction",
        "module_version": "1.0",
        "generated_at":   pd.Timestamp.now().isoformat(),
        "model":          MODEL,
        "pass1_batch_id": pass1_batch_id,
        "pass2_batch_id": pass2_batch_id,
        "input_data": {
            "total_user_messages":            0,  # filled below
            "substantive_messages_analyzed":   total_substantive,
            "pass1_flagged":                  flagged,
            "pass1_flag_rate_pct":            flag_rate,
            "pass2_hypotheses_extracted":     total_hypotheses,
            "pass2_empty_extractions":        empty_extractions,
            "messages_with_multiple_hypotheses": multi_hypo_msgs,
        },
        "type_distribution":       type_dist,
        "confidence_distribution": conf_dist,
        "temporal_scope_distribution": temp_dist,
        "referent_resolution": {
            "total_resolved_referents":  total_resolved,
            "ambiguous_referents_flagged": ambiguous_count,
            "ambiguous_pct": round(ambiguous_count / max(total_resolved, 1) * 100, 1),
        },
        "cross_tabulations":           cross_tabs,
        "confidence_language_frequency": conf_lang_freq,
        "cost": {
            "pass1_input_tokens":  p1_input,
            "pass1_output_tokens": p1_output,
            "pass2_input_tokens":  p2_input,
            "pass2_output_tokens": p2_output,
            "total_cost_usd":      total_cost,
        },
        "signature_fragment": {
            "total_hypotheses":               total_hypotheses,
            "dominant_type":                  dominant_type,
            "dominant_confidence_level":       dominant_conf,
            "hypotheses_per_month_avg":       avg_per_month,
            "most_hypothesis_dense_function": dense_func,
            "summary":                        summary_text,
        },
        "figures_generated": [
            "hypothesis_type_distribution.png",
            "hypotheses_over_time.png",
            "confidence_level_distribution.png",
            "hypotheses_by_function.png",
            "hypotheses_by_emotion.png",
            "temporal_scope_distribution.png",
            "confidence_language_frequency.png",
            "hypothesis_dashboard.png",
        ],
        "data_outputs": [
            "data/processed/hypotheses.parquet",
            "outputs/reports/hypothesis_catalog.csv",
        ],
        "warnings": warnings_list,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.4f}")
    return report


# -- Validation ---------------------------------------------------------------
def run_validation(df, conv, report):
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    checks = []

    def chk(label, result):
        status = "PASS" if result else "FAIL"
        checks.append((label, status))
        print(f"  [{status}] {label}")

    # 1. hypotheses.parquet exists
    chk("hypotheses.parquet exists", os.path.exists(OUT_PARQUET))

    # 2. hypothesis_catalog.csv exists
    chk("hypothesis_catalog.csv exists", os.path.exists(OUT_CATALOG))

    # 3. All hypothesis types valid
    if len(df) > 0:
        all_types = set(df["type"].astype(str).unique())
        invalid_types = all_types - set(VALID_TYPES)
        chk(f"All hypothesis types valid ({invalid_types if invalid_types else 'none invalid'})",
            len(invalid_types) == 0)
    else:
        chk("All hypothesis types valid (no data)", True)

    # 4. All confidence levels valid
    if len(df) > 0:
        all_conf = set(df["confidence_level"].astype(str).unique())
        invalid_conf = all_conf - set(VALID_CONFIDENCE)
        chk(f"All confidence levels valid ({invalid_conf if invalid_conf else 'none invalid'})",
            len(invalid_conf) == 0)
    else:
        chk("All confidence levels valid (no data)", True)

    # 5. All temporal scopes valid or null
    if len(df) > 0:
        preds = df[df["type"].astype(str) == "prediction"]
        if len(preds) > 0:
            all_scopes = set(preds["temporal_scope"].dropna().astype(str).unique())
            invalid_scopes = all_scopes - set(VALID_TEMPORAL)
            chk(f"All temporal scopes valid ({invalid_scopes if invalid_scopes else 'none invalid'})",
                len(invalid_scopes) == 0)
        else:
            chk("All temporal scopes valid (no predictions)", True)
    else:
        chk("All temporal scopes valid (no data)", True)

    # 6. All conversation_ids exist in conversations_clean
    if len(df) > 0:
        valid_ids = set(conv["conversation_id"])
        orphans   = set(df["conversation_id"]) - valid_ids
        chk(f"No orphan conversation_ids ({len(orphans)} orphans)", len(orphans) == 0)
    else:
        chk("No orphan conversation_ids (no data)", True)

    # 7. All resolved_text values non-empty
    if len(df) > 0:
        empty_resolved = (df["resolved_text"].astype(str).str.strip() == "").sum()
        chk(f"All resolved_text non-empty ({empty_resolved} empty)", empty_resolved == 0)
    else:
        chk("All resolved_text non-empty (no data)", True)

    # 8. No raw names in resolved_text (spot check — just verify brackets exist)
    if len(df) > 0:
        sample = df.head(50)
        has_brackets = sample["resolved_text"].astype(str).str.contains(r"\[", regex=True)
        bracket_pct = has_brackets.mean() * 100
        chk(f"Resolved text uses brackets ({bracket_pct:.0f}% of top 50 have brackets)",
            bracket_pct >= 50 or len(df) < 5)
    else:
        chk("Resolved text uses brackets (no data)", True)

    # 9. hypothesis_id values are unique
    if len(df) > 0:
        dupes = df["hypothesis_id"].duplicated().sum()
        chk(f"hypothesis_id values unique ({dupes} duplicates)", dupes == 0)
    else:
        chk("hypothesis_id values unique (no data)", True)

    # 10. Pass 1 flag rate sanity check
    input_data = report.get("input_data", {})
    flag_rate  = input_data.get("pass1_flag_rate_pct", 0)
    chk(f"Pass 1 flag rate 1-30% ({flag_rate:.1f}%)",
        1.0 <= flag_rate <= 30.0 or input_data.get("substantive_messages_analyzed", 0) == 0)

    # 11. Pass 2 empty extraction rate < 20%
    flagged = input_data.get("pass1_flagged", 0)
    empty   = input_data.get("pass2_empty_extractions", 0)
    empty_rate = empty / max(flagged, 1) * 100
    chk(f"Pass 2 empty extraction rate < 20% ({empty_rate:.1f}%)",
        empty_rate < 20 or flagged == 0)

    # 12. Report JSON exists
    chk("Report JSON exists", os.path.exists(OUT_REPORT))

    # 13. All figures exist
    expected_figs = report.get("figures_generated", [])
    missing = [f for f in expected_figs if not os.path.exists(figpath(f))]
    chk(f"All {len(expected_figs)} figures exist ({len(missing)} missing)", len(missing) == 0)

    # 14. All PNGs >= 10KB
    png_figs   = [f for f in expected_figs if f.endswith(".png")]
    small_pngs = [
        f for f in png_figs
        if os.path.exists(figpath(f)) and os.path.getsize(figpath(f)) < 10_000
    ]
    chk(f"All PNGs >= 10KB ({len(small_pngs)} too small)", len(small_pngs) == 0)

    # 15. No NaN/Infinity in report JSON
    report_str = json.dumps(clean_dict(report))
    chk("No NaN/Infinity in report JSON",
        "NaN" not in report_str and "Infinity" not in report_str)

    # 16. Signature summary non-empty
    sig = report.get("signature_fragment", {})
    chk("Signature summary non-empty", bool(sig.get("summary", "").strip()))

    # 17. Catalog has status column defaulting to unscored
    if os.path.exists(OUT_CATALOG):
        try:
            cat = pd.read_csv(OUT_CATALOG, nrows=5)
            has_status = "status" in cat.columns
            all_unscored = (cat["status"] == "unscored").all() if has_status and len(cat) > 0 else True
            chk("Catalog has status column defaulting to 'unscored'",
                has_status and all_unscored)
        except Exception:
            chk("Catalog has status column defaulting to 'unscored'", False)
    else:
        chk("Catalog has status column defaulting to 'unscored'", False)

    # Summary
    passed = sum(1 for _, s in checks if s == "PASS")
    total  = len(checks)
    print("=" * 80)
    if passed == total:
        print(f"  ALL {total} CHECKS PASSED")
    else:
        print(f"  {passed}/{total} CHECKS PASSED -- review FAIL items above")
    print("=" * 80)


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Module 3.2g: Hypothesis Extraction via Claude Haiku Batch API (two-pass)"
    )
    parser.add_argument("--resume-pass1-id", default=None,
                        help="Resume Pass 1 from a previously submitted batch ID")
    parser.add_argument("--resume-pass2-id", default=None,
                        help="Resume Pass 2 from a previously submitted batch ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build Pass 1 requests and print stats without submitting")
    parser.add_argument("--detect-only", action="store_true",
                        help="Run Pass 1 only (detection), skip extraction")
    parser.add_argument("--token-threshold", type=int, default=DEFAULT_TOKEN_THRESHOLD,
                        help=f"Minimum token count for substantive messages (default: {DEFAULT_TOKEN_THRESHOLD})")
    args = parser.parse_args()

    warnings_list = []

    # ---- Load data ----
    conv, msgs, substantive, config, sum_map, func_class, emot_class = load_data(args.token_threshold)

    # ---- Build Pass 1 inputs ----
    pass1_inputs = build_pass1_inputs(substantive)
    n_pass1 = build_batch_requests(
        pass1_inputs, PASS1_BATCH_FILE, SYSTEM_PROMPT_PASS1,
        MAX_OUTPUT_TOKENS_PASS1, _format_pass1_message,
    )

    # Estimate cost
    est_p1_cost = (n_pass1 * 245) / 1_000_000 * PRICE_INPUT_PER_MTOK + \
                  (n_pass1 * 15) / 1_000_000 * PRICE_OUTPUT_PER_MTOK
    print(f"\n  Estimated Pass 1 cost    : ${est_p1_cost:.2f}")

    if args.dry_run:
        print(f"\n== DRY RUN: {n_pass1:,} Pass 1 requests built. Not submitting. ==")
        return

    # ---- API key check ----
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    # ---- Pass 1: Detection ----
    if args.resume_pass2_id:
        # Skip Pass 1 entirely — load saved Pass 1 results
        print("\n== Skipping Pass 1 (resuming Pass 2) ==================================")
        # We need pass1_parsed; try loading from interim
        pass1_interim = os.path.join(INTERIM_DIR, "hypothesis_pass1_results.json")
        if os.path.exists(pass1_interim):
            with open(pass1_interim, "r", encoding="utf-8") as f:
                pass1_parsed = json.load(f)
            print(f"  Loaded {len(pass1_parsed):,} Pass 1 results from cache")
        else:
            print("ERROR: Cannot resume Pass 2 without Pass 1 results.")
            print(f"  Expected: {pass1_interim}")
            sys.exit(1)
        pass1_batch_id = "resumed"
    else:
        if args.resume_pass1_id:
            pass1_batch_id = args.resume_pass1_id
            print(f"\n== Resuming Pass 1 batch: {pass1_batch_id} ==")
        else:
            pass1_batch_id = submit_batch(
                client, PASS1_BATCH_FILE, PASS1_BATCH_ID_FILE, "Pass 1",
            )

        poll_batch(client, pass1_batch_id, "Pass 1")
        results_p1, errors_p1 = retrieve_results(client, pass1_batch_id, "Pass 1")

        retried_p1 = retry_errors(
            client, errors_p1, pass1_inputs,
            SYSTEM_PROMPT_PASS1, MAX_OUTPUT_TOKENS_PASS1, _format_pass1_message,
        )
        results_p1.update(retried_p1)

        pass1_parsed, p1_parse_errors = parse_pass1(results_p1)

        # Cache Pass 1 results for potential Pass 2 resume
        pass1_interim = os.path.join(INTERIM_DIR, "hypothesis_pass1_results.json")
        with open(pass1_interim, "w", encoding="utf-8") as f:
            json.dump(pass1_parsed, f)
        print(f"  Cached Pass 1 results to {pass1_interim}")

    # ---- Check detect-only mode ----
    if args.detect_only:
        flagged = sum(1 for v in pass1_parsed.values() if v["has_hypothesis"])
        total   = len(pass1_parsed)
        print(f"\n== DETECT ONLY: {flagged:,} / {total:,} messages flagged ({flagged/max(total,1)*100:.1f}%) ==")
        return

    # ---- Pass 2: Extraction ----
    flagged_ids = [k for k, v in pass1_parsed.items() if v.get("has_hypothesis")]
    print(f"\n  Messages flagged for extraction: {len(flagged_ids):,}")

    if len(flagged_ids) == 0:
        print("  No messages flagged — skipping Pass 2")
        df = save_results([], conv, msgs, func_class, emot_class)
        make_figures(df, conv, config, func_class, emot_class)
        report = generate_report(
            df, conv, pass1_batch_id, "none", pass1_parsed,
            0, 0, 0, warnings_list, func_class, emot_class,
        )
        run_validation(df, conv, report)
        print("\nDone!")
        return

    pass2_inputs = build_pass2_inputs(flagged_ids, pass1_inputs, msgs, conv, sum_map)
    n_pass2 = build_batch_requests(
        pass2_inputs, PASS2_BATCH_FILE, SYSTEM_PROMPT_PASS2,
        MAX_OUTPUT_TOKENS_PASS2, _format_pass2_message,
    )

    if args.resume_pass2_id:
        pass2_batch_id = args.resume_pass2_id
        print(f"\n== Resuming Pass 2 batch: {pass2_batch_id} ==")
    else:
        pass2_batch_id = submit_batch(
            client, PASS2_BATCH_FILE, PASS2_BATCH_ID_FILE, "Pass 2",
        )

    poll_batch(client, pass2_batch_id, "Pass 2")
    results_p2, errors_p2 = retrieve_results(client, pass2_batch_id, "Pass 2")

    retried_p2 = retry_errors(
        client, errors_p2, pass2_inputs,
        SYSTEM_PROMPT_PASS2, MAX_OUTPUT_TOKENS_PASS2, _format_pass2_message,
    )
    results_p2.update(retried_p2)

    # ---- Parse & Save ----
    rows, p2_parse_errors, empty_extractions, multi_hypo_msgs = parse_pass2(
        results_p2, pass2_inputs, pass1_parsed,
    )

    df = save_results(rows, conv, msgs, func_class, emot_class)

    # ---- Figures ----
    make_figures(df, conv, config, func_class, emot_class)

    # ---- Report ----
    report = generate_report(
        df, conv, pass1_batch_id, pass2_batch_id, pass1_parsed,
        p2_parse_errors, empty_extractions, multi_hypo_msgs,
        warnings_list, func_class, emot_class,
    )

    # Fill in total_user_messages from load_data scope
    # (We can compute it here)
    user_msg_count = len(msgs[msgs["role"] == "user"])
    report["input_data"]["total_user_messages"] = user_msg_count
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)

    # ---- Validate ----
    run_validation(df, conv, report)

    print("\nDone!")


if __name__ == "__main__":
    main()
