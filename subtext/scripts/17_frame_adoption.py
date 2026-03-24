"""
Module 3.2e: Frame Adoption Classification
Script: 17_frame_adoption.py

Classifies every user message by how it responds to the AI's preceding
output — adopt, extend, redirect, reject, ignore, or steer.
This is the first message-level classification in the pipeline; all prior
modules operated at the conversation level.

Uses Claude Haiku via standard API on message pairs (AI response + user reply).
All messages are sent to the LLM for classification — no rule-based
short-message filtering. Conversation openers (no preceding AI message)
are rule-classified as steer.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/17_frame_adoption.py

    # Resume a previously submitted batch:
    python scripts/17_frame_adoption.py --resume-batch-id msgbatch_xxxxx

    # Dry run (build requests, don't submit):
    python scripts/17_frame_adoption.py --dry-run

    # Use standard API with concurrency (if batch API is unavailable):
    python scripts/17_frame_adoption.py --standard-api --concurrency 20

    # Adjust token threshold (0 = classify everything via LLM):
    python scripts/17_frame_adoption.py --token-threshold 0
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
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
INTERIM_DIR     = os.path.join(BASE, "data", "interim")
BATCH_FILE      = os.path.join(INTERIM_DIR, "frame_adoption_batch.jsonl")
BATCH_ID_FILE   = os.path.join(INTERIM_DIR, "frame_adoption_batch_id.txt")

OUT_PARQUET = os.path.join(BASE, "data", "processed", "frame_adoption.parquet")
OUT_REPORT  = os.path.join(BASE, "outputs", "reports", "frame_adoption_report.json")
FIG_DIR     = os.path.join(BASE, "outputs", "figures", "frame_adoption")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Model & Pricing ---------------------------------------------------------
MODEL             = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 60
PRICE_INPUT_PER_MTOK  = 0.40   # $0.80/MTok * 0.5 batch discount
PRICE_OUTPUT_PER_MTOK = 2.00   # $4.00/MTok * 0.5 batch discount

# -- Token limits for message pair construction ------------------------------
DEFAULT_TOKEN_THRESHOLD    = 0
MAX_ASSISTANT_TOKENS       = 500

# -- Valid frame adoption labels ---------------------------------------------
VALID_FRAMES = [
    "adopt",
    "extend",
    "redirect",
    "reject",
    "ignore",
    "steer",
]

ALL_LABELS = VALID_FRAMES

# Valid function/emotion labels (for cross-tab figures)
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

# -- Style constants (match existing scripts) ---------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
DPI = 150

# Frame adoption color palette — semantic spectrum
FRAME_COLORS = {
    "adopt":          "#59A14F",   # green — accepted AI frame
    "extend":         "#8CD17D",   # light green — accepted + went further
    "redirect":       "#F28E2B",   # orange — changed frame
    "reject":         "#E15759",   # red — rejected AI frame
    "ignore":         "#BAB0AC",   # gray — didn't engage
    "steer":          "#4E79A7",   # blue — user driving
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

# -- System Prompt ------------------------------------------------------------
SYSTEM_PROMPT = "\n".join([
    "You are a research assistant analyzing conversational dynamics between a user and an AI assistant.",
    "",
    "Given the AI's previous response and the user's next message, classify how the user responded",
    "to the AI's frame — its interpretation, direction, or conceptual structure.",
    "",
    "Categories:",
    "",
    "- adopt: The user accepted the AI's frame. Built on it. Used the AI's language or conceptual",
    "  structure. The AI's response visibly shaped the user's next message.",
    "- extend: The user took the AI's frame but pushed it further than the AI went. Accepted the",
    "  premise but added something new the AI didn't say.",
    "- redirect: The user changed the frame. Offered a different interpretation or angle.",
    "  Not disagreement — a lateral move.",
    "- reject: The user explicitly disagreed with or corrected the AI. The AI's frame was actively",
    "  pushed away.",
    "- ignore: The user didn't engage with the AI's response at all. Introduced something",
    "  unrelated. The AI's output had no visible influence.",
    "- steer: The user issued a new question, task, or direction. Actively commanding the",
    "  conversation rather than responding to the AI's content.",
    "",
    "Key distinctions:",
    "- adopt vs extend: adopt stays within the AI's frame; extend accepts it but adds new territory",
    "- redirect vs reject: redirect offers an alternative; reject says the AI was wrong",
    "- ignore vs steer: ignore is passive non-engagement; steer is active direction-setting",
    "",
    'Respond with ONLY a JSON object:',
    '{"frame": "<category>", "confidence": <0.0-1.0>}',
    "",
    "Do not include any other text, explanation, or markdown formatting. Only the JSON object.",
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


# -- Step 0: Load Data --------------------------------------------------------
def load_data(token_threshold):
    print("\n== Step 0: Load data ===================================================")

    for path, label in [
        (CONV_PATH, "conversations_clean.parquet"),
        (MSGS_PATH, "messages_clean.parquet"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    conversations = pd.read_parquet(CONV_PATH)
    messages      = pd.read_parquet(MSGS_PATH)

    conv = conversations[conversations["is_analysable"]].copy()
    analysable_ids = set(conv["conversation_id"])
    msgs = messages[messages["conversation_id"].isin(analysable_ids)].copy()

    user_msgs = msgs[msgs["role"] == "user"].copy()

    print(f"  Analysable conversations : {len(conv):,}")
    print(f"  Messages in scope        : {len(msgs):,}")
    print(f"  User messages            : {len(user_msgs):,}")
    print(f"  Token threshold          : {token_threshold}")

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

    return conv, msgs, user_msgs, config, func_class, emot_class


# -- Build message pair inputs ------------------------------------------------
def build_message_pair_inputs(msgs, user_msgs, token_threshold):
    """
    For each user message, determine classification method:
      - rule: conversation_opener (no preceding AI message) → steer
      - llm:  all other messages get API call with preceding assistant message as context

    Returns:
      rule_rows:  list of dicts for rule-classified messages (openers only)
      llm_inputs: dict of custom_id -> {user_text, assistant_text, user_tokens, assistant_tokens}
    """
    print("\n== Building message pair inputs ========================================")
    enc = tiktoken.get_encoding("cl100k_base")

    rule_rows  = []
    llm_inputs = {}

    n_opener  = 0
    n_llm     = 0

    # Pre-sort messages by conversation and index for efficient lookup
    msgs_sorted = msgs.sort_values(["conversation_id", "msg_index"])

    # Build a lookup: for each conversation, list of (msg_index, role, text, token_count)
    conv_msgs = {}
    for _, row in msgs_sorted.iterrows():
        cid = row["conversation_id"]
        if cid not in conv_msgs:
            conv_msgs[cid] = []
        conv_msgs[cid].append({
            "msg_index":   row["msg_index"],
            "role":        row["role"],
            "text":        str(row.get("text", "") or ""),
            "token_count": int(row.get("token_count", 0)),
        })

    for _, row in tqdm(user_msgs.iterrows(), total=len(user_msgs), desc="Building pairs"):
        cid       = row["conversation_id"]
        msg_idx   = row["msg_index"]
        user_toks = int(row.get("token_count", 0))
        user_text = str(row.get("text", "") or "")

        # Check if this is a conversation opener (first user message)
        is_opener = bool(row.get("is_first_user_msg", False))

        # Find preceding assistant message
        preceding_assistant_text  = ""
        preceding_assistant_toks  = 0
        if cid in conv_msgs:
            prev_asst = None
            for m in conv_msgs[cid]:
                if m["msg_index"] < msg_idx and m["role"] == "assistant":
                    prev_asst = m
                # Messages are sorted, so keep overwriting with the latest one before msg_idx
            if prev_asst is not None:
                preceding_assistant_text = prev_asst["text"]
                preceding_assistant_toks = prev_asst["token_count"]

        # Rule 1: conversation opener → steer
        if is_opener or preceding_assistant_text.strip() == "":
            rule_rows.append({
                "conversation_id":           cid,
                "message_index":             msg_idx,
                "frame_adoption":            "steer",
                "frame_confidence":          float("nan"),
                "classification_method":     "rule",
                "is_conversation_opener":    True,
                "user_tokens":               user_toks,
                "assistant_tokens_preceding": preceding_assistant_toks,
                "input_tokens":              0,
                "output_tokens":             0,
            })
            n_opener += 1
            continue

        # LLM classification: truncate assistant message to MAX_ASSISTANT_TOKENS
        asst_truncated = preceding_assistant_text
        try:
            asst_tokens = enc.encode(preceding_assistant_text)
            if len(asst_tokens) > MAX_ASSISTANT_TOKENS:
                asst_truncated = enc.decode(asst_tokens[:MAX_ASSISTANT_TOKENS])
        except Exception:
            asst_truncated = preceding_assistant_text[:2000]

        custom_id = f"{cid}__msg{msg_idx}"
        llm_inputs[custom_id] = {
            "user_text":               user_text,
            "assistant_text":          asst_truncated,
            "conversation_id":         cid,
            "message_index":           msg_idx,
            "user_tokens":             user_toks,
            "assistant_tokens_preceding": preceding_assistant_toks,
        }
        n_llm += 1

    print(f"  Conversation openers (rule → steer)  : {n_opener:,}")
    print(f"  LLM classification needed             : {n_llm:,}")
    print(f"  Total user messages                   : {n_opener + n_llm:,}")

    return rule_rows, llm_inputs


def _format_user_message(inp):
    """Format the message pair for the classifier."""
    return (
        f"AI's previous response (truncated):\n"
        f"{inp['assistant_text']}\n\n"
        f"---\n\n"
        f"User's next message:\n"
        f"{inp['user_text']}"
    )


# -- Step 1: Build batch requests ---------------------------------------------
def build_batch_requests(llm_inputs):
    print("\n== Step 1: Build batch requests ========================================")
    with open(BATCH_FILE, "w", encoding="utf-8") as f:
        for custom_id, inp in llm_inputs.items():
            request = {
                "custom_id": custom_id,
                "params": {
                    "model":      MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "system":     SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": _format_user_message(inp)}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    count = sum(1 for _ in open(BATCH_FILE, encoding="utf-8"))
    print(f"  Batch file               : {BATCH_FILE}")
    print(f"  Total requests           : {count:,}")
    return count


# -- Step 2: Submit batch -----------------------------------------------------
def submit_batch(client):
    print("\n== Step 2: Submit batch ================================================")
    requests = []
    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            requests.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests)
    print(f"  ============================================")
    print(f"  BATCH SUBMITTED: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ============================================")
    print(f"  Save this ID to resume: --resume-batch-id {batch.id}")

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)
    return batch.id


# -- Step 3: Poll -------------------------------------------------------------
def poll_batch(client, batch_id):
    print(f"\n== Step 3: Poll batch {batch_id} ==")
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
            print(f"\n  Batch complete!")
            return status
        time.sleep(60)


# -- Step 4: Retrieve and parse results ---------------------------------------
def retrieve_results(client, batch_id):
    print(f"\n== Step 4: Retrieve results ============================================")
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


def retry_errors(client, errors, llm_inputs):
    if not errors:
        return {}
    print(f"\n  Retrying {len(errors)} failure(s) via standard API...")
    retried = {}
    for err in tqdm(errors, desc="Retrying"):
        custom_id = err["custom_id"]
        inp = llm_inputs.get(custom_id)
        if not inp:
            continue
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _format_user_message(inp)}],
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


# -- Standard API (concurrent) fallback --------------------------------------
def run_standard_api(llm_inputs, concurrency=20):
    """
    Process all requests via the standard (non-batch) API with async concurrency.
    Fallback for when the Batch API is unavailable.
    """
    import asyncio

    print(f"\n== Standard API mode ({concurrency} concurrent requests) =================")
    print(f"  Total requests: {len(llm_inputs):,}")

    import anthropic as anthropic_mod
    async_client = anthropic_mod.AsyncAnthropic()

    results = {}
    errors  = []
    sem     = asyncio.Semaphore(concurrency)

    async def classify_one(custom_id, inp, max_retries=3):
        for attempt in range(max_retries):
            async with sem:
                try:
                    response = await async_client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": _format_user_message(inp)}],
                    )
                    raw_text = "".join(
                        block.text for block in response.content if block.type == "text"
                    )
                    return custom_id, {
                        "raw":           raw_text.strip(),
                        "input_tokens":  response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # exponential backoff
                    else:
                        return custom_id, None

    async def run_all():
        tasks = [
            classify_one(cid, inp)
            for cid, inp in llm_inputs.items()
        ]
        completed = 0
        total     = len(tasks)
        batch_size = 200  # report progress every N

        for i in range(0, total, batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            for custom_id, result in batch_results:
                if result is not None:
                    results[custom_id] = result
                else:
                    errors.append({"custom_id": custom_id})
                completed += 1
            print(f"  Progress: {min(i + batch_size, total):,}/{total:,} "
                  f"({min(i + batch_size, total)/total*100:.0f}%) "
                  f"— succeeded: {len(results):,}, errors: {len(errors):,}")

    asyncio.run(run_all())
    print(f"  Completed: {len(results):,} succeeded, {len(errors):,} errors")
    return results, errors


def parse_frame(raw_text):
    """Parse JSON response. Returns (frame, confidence, error_msg)."""
    try:
        # Strip markdown code fences (Haiku wraps output despite being told not to)
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        frame  = parsed.get("frame", "").strip()
        conf   = float(parsed.get("confidence", 0.0))

        if frame not in VALID_FRAMES:
            return None, 0.0, f"invalid frame label: {frame!r}"

        conf = max(0.0, min(1.0, conf))
        return frame, float(conf), None
    except Exception as e:
        return None, 0.0, f"parse error: {e}"


def build_output_rows(results, llm_inputs):
    print("\n  Parsing frame adoption responses...")
    rows         = []
    parse_errors = 0

    for custom_id, res in tqdm(results.items(), desc="Parsing"):
        inp = llm_inputs.get(custom_id, {})
        frame, conf, err = parse_frame(res.get("raw", ""))
        if err:
            parse_errors += 1
            frame, conf = "error", 0.0
        rows.append({
            "conversation_id":           inp.get("conversation_id", ""),
            "message_index":             inp.get("message_index", 0),
            "frame_adoption":            frame,
            "frame_confidence":          conf,
            "classification_method":     "llm",
            "is_conversation_opener":    False,
            "user_tokens":               inp.get("user_tokens", 0),
            "assistant_tokens_preceding": inp.get("assistant_tokens_preceding", 0),
            "input_tokens":              res.get("input_tokens", 0),
            "output_tokens":             res.get("output_tokens", 0),
        })

    print(f"  Parse errors             : {parse_errors:,}")
    return rows, parse_errors


def fill_placeholders(rows_dict, llm_inputs):
    filled = 0
    for custom_id, inp in llm_inputs.items():
        key = (inp["conversation_id"], inp["message_index"])
        if key not in rows_dict:
            rows_dict[key] = {
                "conversation_id":           inp["conversation_id"],
                "message_index":             inp["message_index"],
                "frame_adoption":            "error",
                "frame_confidence":          0.0,
                "classification_method":     "llm",
                "is_conversation_opener":    False,
                "user_tokens":               inp.get("user_tokens", 0),
                "assistant_tokens_preceding": inp.get("assistant_tokens_preceding", 0),
                "input_tokens":              0,
                "output_tokens":             0,
            }
            filled += 1
    if filled:
        print(f"  Filled {filled} placeholder(s) for permanently failed messages")
    return filled


# -- Step 5: Save parquet -----------------------------------------------------
def save_parquet(all_rows):
    print("\n== Step 5: Save results ================================================")
    df = pd.DataFrame(all_rows)

    df["frame_adoption"]            = df["frame_adoption"].astype("category")
    df["classification_method"]     = df["classification_method"].astype("category")
    df["is_conversation_opener"]    = df["is_conversation_opener"].astype(bool)
    df["frame_confidence"]          = df["frame_confidence"].astype("float32")
    df["message_index"]             = df["message_index"].astype("int32")
    df["user_tokens"]               = df["user_tokens"].astype("int32")
    df["assistant_tokens_preceding"] = df["assistant_tokens_preceding"].astype("int32")
    df["input_tokens"]              = df["input_tokens"].astype("int32")
    df["output_tokens"]             = df["output_tokens"].astype("int32")

    # Sort for deterministic output
    df = df.sort_values(["conversation_id", "message_index"]).reset_index(drop=True)

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Rows:  {len(df):,}")
    return df


# -- Step 6: Figures ----------------------------------------------------------
def make_figures(df, conv, config, func_class=None, emot_class=None):
    print("\n== Step 6: Generating figures ==========================================")

    # Merge with conversation metadata
    merged = df.merge(
        conv[[
            "conversation_id", "conversation_type", "model_era",
            "year_month", "time_of_day", "turns", "duration_minutes",
            "msg_count",
        ]],
        on="conversation_id", how="left",
    )

    _fig1_distribution(df)
    _fig2_distribution_llm_only(df)
    _fig3_frame_by_function(df, func_class)
    _fig4_frame_by_emotion(df, emot_class)
    _fig5_over_time(merged, config)
    _fig6_influence_profile(df)
    _fig7_by_position(df, conv)
    _fig8_by_message_length(df)
    _fig9_adoption_by_depth(df, conv)
    _fig10_dashboard(df, conv, func_class, emot_class)
    print("  All figures saved.")


def _fig1_distribution(df):
    """Frame Adoption Distribution — all labels."""
    counts = df["frame_adoption"].astype(str).value_counts()
    # Order: 6 frame labels sorted by count, then any remaining (e.g. error)
    order = []
    for label in VALID_FRAMES:
        if label in counts.index:
            order.append(label)
    for label in counts.index:
        if label not in order:
            order.append(label)
    counts = counts.reindex(order).dropna()
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [FRAME_COLORS.get(l, COLOR_PRIMARY) for l in counts.index]
    bars   = ax.barh(counts.index, counts.values, color=colors)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of Messages")
    ax.set_title("Frame Adoption Distribution (All User Messages)")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("frame_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: frame_distribution.png")


def _fig2_distribution_llm_only(df):
    """Frame Adoption Distribution — LLM-classified only, excluding openers."""
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ]
    counts = llm["frame_adoption"].astype(str).value_counts()
    order  = [f for f in VALID_FRAMES if f in counts.index]
    for label in counts.index:
        if label not in order:
            order.append(label)
    counts = counts.reindex(order).dropna()
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [FRAME_COLORS.get(l, COLOR_PRIMARY) for l in counts.index]
    bars   = ax.barh(counts.index, counts.values, color=colors)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Number of Messages")
    ax.set_title("Frame Adoption Distribution (LLM-Classified, Excl. Openers)")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("frame_distribution_llm_only.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: frame_distribution_llm_only.png")


def _fig3_frame_by_function(df, func_class):
    """Heatmap: frame adoption x functional classification."""
    if func_class is None:
        print("  Skipping frame_by_function.png (functional_classifications not available)")
        _placeholder_figure("frame_by_function.png", "Functional classifications not available")
        return

    # Frame adoption is per-message; function is per-conversation. Join on conversation_id.
    llm = df[df["classification_method"].astype(str) == "llm"].copy()
    combined = llm.merge(
        func_class[["conversation_id", "function_primary"]],
        on="conversation_id", how="inner",
    )
    if len(combined) == 0:
        print("  Skipping frame_by_function.png (no matching data)")
        _placeholder_figure("frame_by_function.png", "No matching data after merge")
        return

    ct   = pd.crosstab(combined["frame_adoption"].astype(str),
                       combined["function_primary"].astype(str))
    rows = [f for f in VALID_FRAMES if f in ct.index]
    cols = [f for f in VALID_FUNCTIONS if f in ct.columns]
    ct   = ct.reindex(index=rows, columns=cols, fill_value=0)

    # Normalize to row percentages for better comparison
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(ct_pct.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Row % (frame → function)")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    vmax = ct_pct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = ct_pct.values[i, j]
            count = int(ct.values[i, j])
            if count > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title("Frame Adoption by Functional Category (row-normalized %)")
    ax.set_xlabel("Functional Category")
    ax.set_ylabel("Frame Adoption")
    plt.tight_layout()
    plt.savefig(figpath("frame_by_function.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: frame_by_function.png")


def _fig4_frame_by_emotion(df, emot_class):
    """Heatmap: frame adoption x emotional state."""
    if emot_class is None:
        print("  Skipping frame_by_emotion.png (emotional_states not available)")
        _placeholder_figure("frame_by_emotion.png", "Emotional states not available")
        return

    llm = df[df["classification_method"].astype(str) == "llm"].copy()
    combined = llm.merge(
        emot_class[["conversation_id", "emotion_primary"]],
        on="conversation_id", how="inner",
    )
    if len(combined) == 0:
        print("  Skipping frame_by_emotion.png (no matching data)")
        _placeholder_figure("frame_by_emotion.png", "No matching data after merge")
        return

    ct   = pd.crosstab(combined["frame_adoption"].astype(str),
                       combined["emotion_primary"].astype(str))
    rows = [f for f in VALID_FRAMES if f in ct.index]
    cols = [e for e in VALID_EMOTIONS if e in ct.columns]
    ct   = ct.reindex(index=rows, columns=cols, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(ct_pct.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=ax, label="Row % (frame → emotion)")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    vmax = ct_pct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = ct_pct.values[i, j]
            count = int(ct.values[i, j])
            if count > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title("Frame Adoption by Emotional State (row-normalized %)")
    ax.set_xlabel("Emotional State")
    ax.set_ylabel("Frame Adoption")
    plt.tight_layout()
    plt.savefig(figpath("frame_by_emotion.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: frame_by_emotion.png")


def _fig5_over_time(merged, config):
    """Interactive stacked area: frame adoption proportions over time."""
    llm = merged[merged["classification_method"].astype(str) == "llm"].copy()
    if "year_month" not in llm.columns or llm["year_month"].isna().all():
        print("  Skipping frame_over_time.html (no year_month data)")
        return

    ct     = pd.crosstab(llm["year_month"].astype(str),
                         llm["frame_adoption"].astype(str))
    ct     = ct.reindex(columns=[f for f in VALID_FRAMES if f in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.sort_index()

    fig = go.Figure()
    for frame in ct_pct.columns:
        fig.add_trace(go.Scatter(
            x=ct_pct.index.tolist(),
            y=ct_pct[frame].round(1).tolist(),
            mode="lines",
            stackgroup="one",
            name=frame,
            line_color=FRAME_COLORS.get(frame, "#888888"),
            hovertemplate="%{x}<br>" + frame + ": %{y:.1f}%<extra></extra>",
        ))

    era_boundaries = config.get("model_era_boundaries", {})
    for era_name, era_date in era_boundaries.items():
        era_ym = era_date[:7]
        if ct_pct.index.min() <= era_ym <= ct_pct.index.max():
            fig.add_vline(
                x=era_ym, line_dash="dash",
                line_color="rgba(100,100,100,0.5)",
                annotation_text=era_name.replace("_", " "),
                annotation_position="top left",
                annotation_font_size=9,
            )

    fig.update_layout(
        title="Frame Adoption Over Time (LLM-Classified Only)",
        xaxis_title="Month",
        yaxis_title="Proportion (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        legend_title="Frame",
        height=500,
    )
    fig.write_html(figpath("frame_over_time.html"), include_plotlyjs=True)
    print(f"  Saved: frame_over_time.html")


def _fig6_influence_profile(df):
    """Donut chart: influenced / independent / driving."""
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ]
    fa = llm["frame_adoption"].astype(str)
    total = len(fa)
    if total == 0:
        _placeholder_figure("influence_profile.png", "No LLM-classified data")
        return

    influenced  = ((fa == "adopt") | (fa == "extend")).sum()
    independent = ((fa == "redirect") | (fa == "reject")).sum()
    driving     = ((fa == "steer") | (fa == "ignore")).sum()
    other       = total - influenced - independent - driving

    sizes  = [influenced, independent, driving]
    labels = [
        f"Influenced\n(adopt+extend)\n{influenced/total*100:.1f}%",
        f"Independent\n(redirect+reject)\n{independent/total*100:.1f}%",
        f"Driving\n(steer+ignore)\n{driving/total*100:.1f}%",
    ]
    colors = ["#59A14F", "#E15759", "#4E79A7"]

    if other > 0:
        sizes.append(other)
        labels.append(f"Other\n{other/total*100:.1f}%")
        colors.append("#BAB0AC")

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for t in texts:
        t.set_fontsize(10)

    # Center annotation
    ratio = influenced / max(independent, 1)
    ax.text(0, 0, f"Influence\nRatio\n{ratio:.2f}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=COLOR_PRIMARY)

    ax.set_title("Influence Profile: How the User Responds to AI Framing", fontsize=12)
    plt.tight_layout()
    plt.savefig(figpath("influence_profile.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: influence_profile.png")


def _fig7_by_position(df, conv):
    """Frame adoption by normalized conversation position."""
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ].copy()

    # Get message count per conversation to normalize position
    msg_counts = conv.set_index("conversation_id")["msg_count"].to_dict()
    llm["msg_count"] = llm["conversation_id"].map(msg_counts)
    llm = llm[llm["msg_count"] > 0].copy()
    llm["position_pct"] = llm["message_index"] / llm["msg_count"] * 100

    # Bin into deciles
    llm["position_bin"] = pd.cut(llm["position_pct"],
                                  bins=range(0, 101, 10),
                                  labels=[f"{i}-{i+10}%" for i in range(0, 100, 10)])
    llm = llm.dropna(subset=["position_bin"])

    if len(llm) == 0:
        _placeholder_figure("frame_by_position.png", "No position data")
        return

    ct     = pd.crosstab(llm["position_bin"], llm["frame_adoption"].astype(str))
    ct     = ct.reindex(columns=[f for f in VALID_FRAMES if f in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for frame in ct_pct.columns:
        ax.plot(range(len(ct_pct)), ct_pct[frame].values,
                marker="o", markersize=4, linewidth=2,
                label=frame, color=FRAME_COLORS.get(frame, "#888888"))

    ax.set_xticks(range(len(ct_pct)))
    ax.set_xticklabels(ct_pct.index, fontsize=8)
    ax.set_xlabel("Position in Conversation (normalized)")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Frame Adoption by Conversation Position")
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("frame_by_position.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: frame_by_position.png")


def _fig8_by_message_length(df):
    """Box plot: message token count by frame adoption label."""
    llm = df[df["classification_method"].astype(str) == "llm"].copy()
    fa  = llm["frame_adoption"].astype(str)

    order = [f for f in VALID_FRAMES if f in fa.values]
    if not order:
        _placeholder_figure("frame_by_length.png", "No LLM-classified data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    data    = [llm.loc[fa == f, "user_tokens"].values for f in order]
    colors  = [FRAME_COLORS.get(f, COLOR_PRIMARY) for f in order]

    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add means as diamonds
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(order) + 1), means,
               marker="D", s=40, color="black", zorder=5, label="Mean")

    ax.set_xlabel("Frame Adoption Label")
    ax.set_ylabel("User Message Token Count")
    ax.set_title("Message Length by Frame Adoption Type")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("frame_by_length.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: frame_by_length.png")


def _fig9_adoption_by_depth(df, conv):
    """Adoption rate (adopt+extend %) by conversation depth bins."""
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ].copy()
    msg_counts = conv.set_index("conversation_id")["msg_count"].to_dict()
    llm["msg_count"] = llm["conversation_id"].map(msg_counts)

    bins   = [0, 5, 15, 30, 60, 999]
    labels = ["1-5", "6-15", "16-30", "31-60", "60+"]
    llm["depth_bin"] = pd.cut(llm["msg_count"], bins=bins, labels=labels)
    llm = llm.dropna(subset=["depth_bin"])

    if len(llm) == 0:
        _placeholder_figure("adoption_by_depth.png", "No depth data")
        return

    fa = llm["frame_adoption"].astype(str)

    results = []
    for label in labels:
        subset = fa[llm["depth_bin"] == label]
        n = len(subset)
        if n == 0:
            continue
        adopt_ext = ((subset == "adopt") | (subset == "extend")).sum()
        results.append({
            "bin":     label,
            "rate":    adopt_ext / n * 100,
            "n":       n,
        })

    if not results:
        _placeholder_figure("adoption_by_depth.png", "No depth data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(results))
    ax.bar(x, [r["rate"] for r in results],
           color=FRAME_COLORS["adopt"], alpha=0.8)

    for i, r in enumerate(results):
        ax.text(i, r["rate"] + 1, f'{r["rate"]:.1f}%\n(n={r["n"]:,})',
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([r["bin"] for r in results])
    ax.set_xlabel("Conversation Depth (total messages)")
    ax.set_ylabel("Adoption Rate (adopt + extend) %")
    ax.set_title("AI Influence by Conversation Depth")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("adoption_by_depth.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: adoption_by_depth.png")


def _fig10_dashboard(df, conv, func_class=None, emot_class=None):
    """Influence profile summary dashboard."""
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ]
    fa    = llm["frame_adoption"].astype(str)
    total = len(fa)
    if total == 0:
        _placeholder_figure("influence_dashboard.png", "No LLM-classified data")
        return

    influenced  = ((fa == "adopt") | (fa == "extend")).sum()
    independent = ((fa == "redirect") | (fa == "reject")).sum()
    driving     = ((fa == "steer") | (fa == "ignore")).sum()

    infl_pct = influenced / total * 100
    indp_pct = independent / total * 100
    driv_pct = driving / total * 100
    infl_ratio = influenced / max(independent, 1)

    # Find most influenced / independent functional category
    most_infl_func = "N/A"
    most_indp_func = "N/A"
    if func_class is not None:
        try:
            combined = llm.merge(
                func_class[["conversation_id", "function_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                func_rates = []
                for func_name, grp in combined.groupby("function_primary"):
                    grp_fa = grp["frame_adoption"].astype(str)
                    n = len(grp_fa)
                    if n < 10:
                        continue
                    adopt_ext = ((grp_fa == "adopt") | (grp_fa == "extend")).sum()
                    func_rates.append({"function": str(func_name), "rate": adopt_ext / n * 100})
                if func_rates:
                    func_rates.sort(key=lambda x: x["rate"], reverse=True)
                    most_infl_func = f"{func_rates[0]['function']} ({func_rates[0]['rate']:.0f}%)"
                    most_indp_func = f"{func_rates[-1]['function']} ({func_rates[-1]['rate']:.0f}%)"
        except Exception:
            pass

    # Most influenced emotional state
    most_infl_emot = "N/A"
    if emot_class is not None:
        try:
            combined = llm.merge(
                emot_class[["conversation_id", "emotion_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                emot_rates = []
                for emot_name, grp in combined.groupby("emotion_primary"):
                    grp_fa = grp["frame_adoption"].astype(str)
                    n = len(grp_fa)
                    if n < 10:
                        continue
                    adopt_ext = ((grp_fa == "adopt") | (grp_fa == "extend")).sum()
                    emot_rates.append({"emotion": str(emot_name), "rate": adopt_ext / n * 100})
                if emot_rates:
                    emot_rates.sort(key=lambda x: x["rate"], reverse=True)
                    most_infl_emot = f"{emot_rates[0]['emotion']} ({emot_rates[0]['rate']:.0f}%)"
        except Exception:
            pass

    # Trend direction
    trend_str = "N/A"
    merged = llm.merge(
        conv[["conversation_id", "year_month"]], on="conversation_id", how="left"
    )
    if "year_month" in merged.columns and not merged["year_month"].isna().all():
        monthly = merged.groupby("year_month").apply(
            lambda g: ((g["frame_adoption"].astype(str) == "adopt") |
                       (g["frame_adoption"].astype(str) == "extend")).sum() / len(g) * 100
        ).sort_index()
        if len(monthly) >= 3:
            try:
                x = np.arange(len(monthly))
                slope, _, _, _, _ = stats.linregress(x, monthly.values)
                trend_str = f"{'↑ increasing' if slope > 0.1 else '↓ decreasing' if slope < -0.1 else '→ stable'} ({slope:+.2f} pp/month)"
            except Exception:
                pass

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # Panel A: influence rate
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.text(0.5, 0.6, f"{infl_pct:.1f}%", ha="center", va="center",
              fontsize=24, fontweight="bold", color="#59A14F", transform=ax_a.transAxes)
    ax_a.text(0.5, 0.25, "adopt + extend", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_a.transAxes)
    ax_a.set_title("Influence Rate", fontsize=11)

    # Panel B: independence rate
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    ax_b.text(0.5, 0.6, f"{indp_pct:.1f}%", ha="center", va="center",
              fontsize=24, fontweight="bold", color="#E15759", transform=ax_b.transAxes)
    ax_b.text(0.5, 0.25, "redirect + reject", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_b.transAxes)
    ax_b.set_title("Independence Rate", fontsize=11)

    # Panel C: driving rate
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.text(0.5, 0.6, f"{driv_pct:.1f}%", ha="center", va="center",
              fontsize=24, fontweight="bold", color="#4E79A7", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.25, "steer + ignore", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_c.transAxes)
    ax_c.set_title("Driving Rate", fontsize=11)

    # Panel D: influence ratio
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.axis("off")
    ratio_color = "#59A14F" if infl_ratio > 1 else "#E15759"
    ax_d.text(0.5, 0.6, f"{infl_ratio:.2f}", ha="center", va="center",
              fontsize=24, fontweight="bold", color=ratio_color, transform=ax_d.transAxes)
    ax_d.text(0.5, 0.25, "influenced / independent\n(>1 = AI-shaped)", ha="center", va="center",
              fontsize=9, color=COLOR_SECONDARY, transform=ax_d.transAxes)
    ax_d.set_title("Influence Ratio", fontsize=11)

    # Panel E: most influenced function
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.axis("off")
    ax_e.text(0.5, 0.5, most_infl_func.replace("_", "\n"),
              ha="center", va="center", fontsize=9, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9"))
    ax_e.set_title("Most AI-Influenced\nFunction", fontsize=10)

    # Panel F: most independent function
    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.axis("off")
    ax_f.text(0.5, 0.5, most_indp_func.replace("_", "\n"),
              ha="center", va="center", fontsize=9, transform=ax_f.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE"))
    ax_f.set_title("Most Independent\nFunction", fontsize=10)

    # Panel G: most influenced emotion
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.axis("off")
    ax_g.text(0.5, 0.5, most_infl_emot.replace("_", "\n"),
              ha="center", va="center", fontsize=9, transform=ax_g.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0"))
    ax_g.set_title("Most AI-Influenced\nEmotion", fontsize=10)

    # Panel H: trend
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.axis("off")
    ax_h.text(0.5, 0.5, trend_str, ha="center", va="center",
              fontsize=10, transform=ax_h.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_h.set_title("Influence Trend", fontsize=10)

    plt.suptitle("Frame Adoption: Influence Profile Dashboard",
                 fontsize=14, fontweight="bold")
    plt.savefig(figpath("influence_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: influence_dashboard.png")


def _placeholder_figure(name, message):
    """Create a minimal placeholder figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=10)
    ax.set_title(name.replace(".png", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(figpath(name), dpi=DPI)
    plt.close()


# -- Step 7: Report -----------------------------------------------------------
def generate_report(df, conv, batch_id, parse_errors, warnings_list,
                    token_threshold, func_class=None, emot_class=None):
    print("\n== Step 7: Generate report =============================================")

    total_msgs = len(df)
    rule_msgs  = (df["classification_method"].astype(str) == "rule").sum()
    llm_msgs   = (df["classification_method"].astype(str) == "llm").sum()
    openers    = df["is_conversation_opener"].sum()
    errors     = (df["frame_adoption"].astype(str) == "error").sum()

    # -- Distribution (all messages) --
    all_counts = df["frame_adoption"].astype(str).value_counts()
    dist_all = {}
    for label in ALL_LABELS + ["error"]:
        c = int(all_counts.get(label, 0))
        dist_all[label] = {"count": c, "pct": round(c / max(total_msgs, 1) * 100, 1)}

    # -- Distribution (LLM-classified only, excluding openers) --
    llm = df[
        (df["classification_method"].astype(str) == "llm")
        & (~df["is_conversation_opener"])
    ]
    llm_counts = llm["frame_adoption"].astype(str).value_counts()
    llm_total  = len(llm)
    dist_llm = {}
    for label in VALID_FRAMES + ["error"]:
        c = int(llm_counts.get(label, 0))
        dist_llm[label] = {"count": c, "pct": round(c / max(llm_total, 1) * 100, 1)}

    # -- Influence profile --
    fa = llm["frame_adoption"].astype(str)
    influenced  = ((fa == "adopt") | (fa == "extend")).sum()
    independent = ((fa == "redirect") | (fa == "reject")).sum()
    driving     = ((fa == "steer") | (fa == "ignore")).sum()

    infl_pct   = round(influenced / max(llm_total, 1) * 100, 1)
    indp_pct   = round(independent / max(llm_total, 1) * 100, 1)
    driv_pct   = round(driving / max(llm_total, 1) * 100, 1)
    infl_ratio = round(influenced / max(independent, 1), 3)

    # -- Confidence stats --
    conf = df.loc[df["classification_method"].astype(str) == "llm", "frame_confidence"].astype(float)
    if len(conf) > 0:
        conf_stats = {
            "mean":       round(float(conf.mean()), 3),
            "median":     round(float(conf.median()), 3),
            "std":        round(float(conf.std()), 3),
            "pct_high":   round(float((conf >= 0.8).mean() * 100), 1),
            "pct_medium": round(float(((conf >= 0.5) & (conf < 0.8)).mean() * 100), 1),
            "pct_low":    round(float((conf < 0.5).mean() * 100), 1),
        }
    else:
        conf_stats = {"mean": 0, "median": 0, "std": 0,
                      "pct_high": 0, "pct_medium": 0, "pct_low": 0}

    # -- Cross-tabulations --
    cross_tabs = {}

    # by_function
    if func_class is not None:
        try:
            combined = llm.merge(
                func_class[["conversation_id", "function_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                ct = pd.crosstab(combined["function_primary"].astype(str),
                                 combined["frame_adoption"].astype(str))
                cross_tabs["by_function"] = {
                    str(g): {str(f): int(v) for f, v in row.items()}
                    for g, row in ct.iterrows()
                }
        except Exception as e:
            warnings_list.append(f"Could not compute by_function cross-tab: {e}")

    # by_emotion
    if emot_class is not None:
        try:
            combined = llm.merge(
                emot_class[["conversation_id", "emotion_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                ct = pd.crosstab(combined["emotion_primary"].astype(str),
                                 combined["frame_adoption"].astype(str))
                cross_tabs["by_emotion"] = {
                    str(g): {str(f): int(v) for f, v in row.items()}
                    for g, row in ct.iterrows()
                }
        except Exception as e:
            warnings_list.append(f"Could not compute by_emotion cross-tab: {e}")

    # by_year_month
    merged = llm.merge(
        conv[["conversation_id", "year_month", "time_of_day"]],
        on="conversation_id", how="left",
    )
    if "year_month" in merged.columns:
        ct = pd.crosstab(merged["year_month"].astype(str),
                         merged["frame_adoption"].astype(str))
        cross_tabs["by_year_month"] = {
            str(g): {str(f): int(v) for f, v in row.items()}
            for g, row in ct.iterrows()
        }
    if "time_of_day" in merged.columns:
        ct = pd.crosstab(merged["time_of_day"].astype(str),
                         merged["frame_adoption"].astype(str))
        cross_tabs["by_time_of_day"] = {
            str(g): {str(f): int(v) for f, v in row.items()}
            for g, row in ct.iterrows()
        }

    # -- Conversation position analysis --
    msg_counts = conv.set_index("conversation_id")["msg_count"].to_dict()
    llm_pos = llm.copy()
    llm_pos["msg_count"] = llm_pos["conversation_id"].map(msg_counts)
    llm_pos = llm_pos[llm_pos["msg_count"] > 0]
    llm_pos["position_pct"] = llm_pos["message_index"] / llm_pos["msg_count"]

    pos_analysis = {}
    for label, lo, hi in [("early_third", 0, 1/3), ("middle_third", 1/3, 2/3), ("late_third", 2/3, 1.01)]:
        subset = llm_pos[(llm_pos["position_pct"] >= lo) & (llm_pos["position_pct"] < hi)]
        sub_fa = subset["frame_adoption"].astype(str)
        n = len(sub_fa)
        if n > 0:
            adopt_ext = ((sub_fa == "adopt") | (sub_fa == "extend")).sum()
            pos_analysis[f"{label}_adoption_rate"] = round(adopt_ext / n * 100, 1)
        else:
            pos_analysis[f"{label}_adoption_rate"] = 0.0

    # -- Statistical tests --
    stat_tests = {}

    # frame vs function chi-square
    if func_class is not None and "by_function" in cross_tabs:
        try:
            ct = pd.DataFrame(cross_tabs["by_function"]).T.fillna(0)
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _, _ = stats.chi2_contingency(ct.values)
                stat_tests["frame_vs_function"] = {
                    "chi2": round(float(chi2), 2),
                    "p": round(float(p), 6),
                    "significant": bool(p < 0.05),
                }
        except Exception:
            pass

    # frame vs emotion chi-square
    if emot_class is not None and "by_emotion" in cross_tabs:
        try:
            ct = pd.DataFrame(cross_tabs["by_emotion"]).T.fillna(0)
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, _, _ = stats.chi2_contingency(ct.values)
                stat_tests["frame_vs_emotion"] = {
                    "chi2": round(float(chi2), 2),
                    "p": round(float(p), 6),
                    "significant": bool(p < 0.05),
                }
        except Exception:
            pass

    # frame vs position Kruskal-Wallis
    if len(llm_pos) > 0:
        try:
            groups = [
                llm_pos.loc[llm_pos["frame_adoption"].astype(str) == f, "position_pct"].dropna().values
                for f in VALID_FRAMES
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                H, p = stats.kruskal(*groups)
                stat_tests["frame_vs_position"] = {
                    "H": round(float(H), 2),
                    "p": round(float(p), 6),
                    "significant": bool(p < 0.05),
                }
        except Exception:
            pass

    # -- Cost --
    total_input  = int(df["input_tokens"].astype(int).sum())
    total_output = int(df["output_tokens"].astype(int).sum())
    total_cost   = round(
        total_input  / 1_000_000 * PRICE_INPUT_PER_MTOK +
        total_output / 1_000_000 * PRICE_OUTPUT_PER_MTOK,
        4,
    )

    # -- Trend --
    trend_direction = "unknown"
    trend_slope     = 0.0
    if "year_month" in merged.columns and not merged["year_month"].isna().all():
        monthly = merged.groupby("year_month").apply(
            lambda g: ((g["frame_adoption"].astype(str) == "adopt") |
                       (g["frame_adoption"].astype(str) == "extend")).sum() / max(len(g), 1) * 100
        ).sort_index()
        if len(monthly) >= 3:
            try:
                x = np.arange(len(monthly))
                slope, _, _, _, _ = stats.linregress(x, monthly.values)
                trend_slope = round(float(slope), 3)
                if slope > 0.1:
                    trend_direction = "increasing"
                elif slope < -0.1:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            except Exception:
                pass

    # -- Signature fragment --
    dominant = llm_counts.index[0] if len(llm_counts) > 0 else "unknown"
    dominant_pct = round(float(llm_counts.iloc[0] / max(llm_total, 1) * 100), 1) if len(llm_counts) > 0 else 0.0

    # Most influenced/independent function
    most_infl_func = ""
    most_indp_func = ""
    if func_class is not None and "by_function" in cross_tabs:
        try:
            func_rates = {}
            for func_name, frame_counts in cross_tabs["by_function"].items():
                n = sum(frame_counts.values())
                if n < 10:
                    continue
                adopt_ext = frame_counts.get("adopt", 0) + frame_counts.get("extend", 0)
                func_rates[func_name] = adopt_ext / n * 100
            if func_rates:
                most_infl_func = max(func_rates, key=func_rates.get)
                most_indp_func = min(func_rates, key=func_rates.get)
        except Exception:
            pass

    summary_text = (
        f"Influence rate: {infl_pct}% (adopt+extend), "
        f"independence rate: {indp_pct}% (redirect+reject), "
        f"driving rate: {driv_pct}% (steer+ignore). "
        f"Influence ratio: {infl_ratio:.2f} "
        f"({'user is more AI-shaped' if infl_ratio > 1 else 'user is more independent'}). "
        f"Trend: {trend_direction} ({trend_slope:+.2f} pp/month)."
    )

    report = {
        "module":         "frame_adoption",
        "module_version": "1.0",
        "generated_at":   pd.Timestamp.now().isoformat(),
        "model":          MODEL,
        "batch_id":       batch_id,
        "input_data": {
            "total_user_messages":   int(total_msgs),
            "rule_classified":       int(rule_msgs),
            "llm_classified":        int(llm_msgs),
            "conversation_openers":  int(openers),
            "classification_errors": int(errors),
            "token_threshold":       token_threshold,
        },
        "distribution": {
            "all_messages":       dist_all,
            "llm_classified_only": dist_llm,
        },
        "influence_profile": {
            "influenced_pct":  infl_pct,
            "independent_pct": indp_pct,
            "driving_pct":     driv_pct,
            "influence_ratio": infl_ratio,
        },
        "confidence_stats":              conf_stats,
        "cross_tabulations":             cross_tabs,
        "conversation_position_analysis": pos_analysis,
        "statistical_tests":             stat_tests,
        "cost": {
            "input_tokens":   total_input,
            "output_tokens":  total_output,
            "total_cost_usd": total_cost,
            "pricing_note":   (
                "Haiku 4.5 Batch API: $0.40/MTok input, "
                "$2.00/MTok output (50% batch discount)"
            ),
        },
        "influence_signature_fragment": {
            "dominant_frame":           dominant,
            "dominant_pct":             dominant_pct,
            "influence_rate":           infl_pct,
            "independence_rate":        indp_pct,
            "most_influenced_function": most_infl_func,
            "most_independent_function": most_indp_func,
            "trend_direction":          trend_direction,
            "trend_slope_pp_month":     trend_slope,
            "summary":                  summary_text,
        },
        "figures_generated": [
            "frame_distribution.png",
            "frame_distribution_llm_only.png",
            "frame_by_function.png",
            "frame_by_emotion.png",
            "frame_over_time.html",
            "influence_profile.png",
            "frame_by_position.png",
            "frame_by_length.png",
            "adoption_by_depth.png",
            "influence_dashboard.png",
        ],
        "data_outputs": ["data/processed/frame_adoption.parquet"],
        "warnings":     warnings_list,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.4f}")
    return report


# -- Step 8: Validation -------------------------------------------------------
def run_validation(df, conv, report):
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    checks = []

    def chk(label, result):
        status = "PASS" if result else "FAIL"
        checks.append((label, status))
        print(f"  [{status}] {label}")

    # 1. Parquet exists
    chk("frame_adoption.parquet exists", os.path.exists(OUT_PARQUET))

    # 2. Required columns
    required_cols = {
        "conversation_id", "message_index", "frame_adoption", "frame_confidence",
        "classification_method", "is_conversation_opener", "user_tokens",
        "assistant_tokens_preceding", "input_tokens", "output_tokens",
    }
    chk("Required columns present", required_cols.issubset(set(df.columns)))

    # 3. Row count matches total user messages
    analysable_ids = set(conv.loc[conv["is_analysable"], "conversation_id"])
    msgs = pd.read_parquet(MSGS_PATH)
    expected = len(msgs[(msgs["conversation_id"].isin(analysable_ids)) & (msgs["role"] == "user")])
    actual   = len(df)
    pct_diff = abs(actual - expected) / max(expected, 1) * 100
    chk(f"Row count matches user messages ({actual:,} / {expected:,}, diff {pct_diff:.1f}%)",
        pct_diff <= 1)

    # 4. All conversation_ids exist in conversations_clean
    valid_ids = set(conv["conversation_id"])
    orphans   = set(df["conversation_id"]) - valid_ids
    chk("No orphan conversation_ids", len(orphans) == 0)

    # 5. No duplicate (conversation_id, message_index)
    dupes = df.duplicated(subset=["conversation_id", "message_index"]).sum()
    chk("No duplicate (conversation_id, message_index) pairs", dupes == 0)

    # 6. All frame_adoption values valid
    all_vals     = set(df["frame_adoption"].astype(str).unique())
    valid_labels = set(ALL_LABELS) | {"error"}
    invalid      = all_vals - valid_labels
    chk(f"All frame_adoption values valid ({invalid if invalid else 'none invalid'})",
        len(invalid) == 0)

    # 7. Rule-classified messages are opener-steer only
    rule_df = df[df["classification_method"].astype(str) == "rule"]
    rule_labels = set(rule_df["frame_adoption"].astype(str).unique())
    chk("Rule-classified are steer only (openers)",
        rule_labels.issubset({"steer"}))

    # 8. LLM-classified have method=llm
    llm_df = df[df["classification_method"].astype(str) == "llm"]
    chk("LLM-classified messages exist", len(llm_df) > 0)

    # 9. Conversation openers flagged
    openers = df[df["is_conversation_opener"]]
    chk(f"Conversation openers flagged ({len(openers):,})", len(openers) > 0)

    # 10. Confidence in [0, 1] for LLM messages
    llm_conf = llm_df["frame_confidence"].astype(float).dropna()
    chk("Confidence scores in [0, 1]",
        bool(len(llm_conf) == 0 or ((llm_conf >= 0.0).all() and (llm_conf <= 1.0).all())))

    # 11. Error rate < 1%
    err_pct = (df["frame_adoption"].astype(str) == "error").mean() * 100
    chk(f"Classification errors < 1% ({err_pct:.2f}%)", err_pct < 1.0)

    # 12. Report JSON exists
    chk("Report JSON exists", os.path.exists(OUT_REPORT))

    # 13. All figures exist
    expected_figs = report.get("figures_generated", [])
    missing = [f for f in expected_figs if not os.path.exists(figpath(f))]
    chk(f"All {len(expected_figs)} figures exist ({len(missing)} missing)", len(missing) == 0)

    # 14. PNGs >= 10KB
    png_figs   = [f for f in expected_figs if f.endswith(".png")]
    small_pngs = [
        f for f in png_figs
        if os.path.exists(figpath(f)) and os.path.getsize(figpath(f)) < 10_000
    ]
    chk(f"All PNGs >= 10KB ({len(small_pngs)} too small)", len(small_pngs) == 0)

    # 15. HTML self-contained
    for html_name in [f for f in expected_figs if f.endswith(".html")]:
        html_path = figpath(html_name)
        html_ok   = False
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as fh:
                html_ok = "plotly" in fh.read(5000).lower()
        chk(f"{html_name} contains plotlyjs", html_ok)

    # 16. No NaN/Infinity in report
    report_str = json.dumps(clean_dict(report))
    chk("No NaN/Infinity in report JSON",
        "NaN" not in report_str and "Infinity" not in report_str)

    # 17. Signature summary non-empty
    sig = report.get("influence_signature_fragment", {})
    chk("Influence signature summary non-empty",
        bool(sig.get("summary", "").strip()))

    # 18. Cross-tabs non-empty
    xtabs = report.get("cross_tabulations", {})
    chk("Cross-tabulation by_function non-empty",
        len(xtabs.get("by_function", {})) > 0)
    chk("Cross-tabulation by_emotion non-empty",
        len(xtabs.get("by_emotion", {})) > 0)

    # 19. Total = rule + llm
    input_data = report.get("input_data", {})
    total_check = input_data.get("rule_classified", 0) + input_data.get("llm_classified", 0)
    chk(f"Total = rule + llm ({total_check:,} == {input_data.get('total_user_messages', 0):,})",
        total_check == input_data.get("total_user_messages", 0))

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
        description="Module 3.2e: Frame Adoption Classification via Claude Haiku Batch API"
    )
    parser.add_argument("--resume-batch-id", default=None,
                        help="Resume from a previously submitted batch ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build requests and print stats without submitting")
    parser.add_argument("--token-threshold", type=int, default=DEFAULT_TOKEN_THRESHOLD,
                        help=f"Token threshold for rule classification (default: {DEFAULT_TOKEN_THRESHOLD})")
    parser.add_argument("--standard-api", action="store_true",
                        help="Use standard API with async concurrency instead of Batch API")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Number of concurrent requests for --standard-api mode (default: 20)")
    args = parser.parse_args()

    warnings_list    = []
    token_threshold  = args.token_threshold

    print("=" * 80)
    print("Module 3.2e: Frame Adoption Classification")
    print("=" * 80)

    # -- Dry run: no API key needed --
    if args.dry_run:
        conv, msgs, user_msgs, config, func_class, emot_class = load_data(token_threshold)
        rule_rows, llm_inputs = build_message_pair_inputs(msgs, user_msgs, token_threshold)
        n_requests = build_batch_requests(llm_inputs)

        # Cost estimate
        enc = tiktoken.get_encoding("cl100k_base")
        est_input_per_call = MAX_ASSISTANT_TOKENS + 150 + 200  # ~850 tokens
        est_output_per_call = 20
        est_total_input  = n_requests * est_input_per_call
        est_total_output = n_requests * est_output_per_call
        est_cost = (est_total_input / 1_000_000 * PRICE_INPUT_PER_MTOK +
                    est_total_output / 1_000_000 * PRICE_OUTPUT_PER_MTOK)

        print(f"\n== DRY RUN SUMMARY ====================================================")
        print(f"  Rule-classified messages : {len(rule_rows):,}")
        print(f"  LLM requests built       : {n_requests:,}")
        print(f"  Estimated input tokens   : {est_total_input:,}")
        print(f"  Estimated output tokens  : {est_total_output:,}")
        print(f"  Estimated cost           : ${est_cost:.2f}")
        print(f"  Not submitting. --")
        return

    # -- API key check --
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print(f"API key found: {api_key[:12]}...{api_key[-4:]}")

    import anthropic
    client = anthropic.Anthropic()

    conv, msgs, user_msgs, config, func_class, emot_class = load_data(token_threshold)
    rule_rows, llm_inputs = build_message_pair_inputs(msgs, user_msgs, token_threshold)

    batch_id = "standard_api"

    if args.standard_api:
        # -- Standard API with async concurrency --
        results, std_errors = run_standard_api(llm_inputs, concurrency=args.concurrency)
        if std_errors:
            warnings_list.append(f"{len(std_errors)} standard API error(s)")
    else:
        # -- Batch API flow --
        n_requests = build_batch_requests(llm_inputs)

        if args.resume_batch_id:
            batch_id = args.resume_batch_id
            print(f"\n-- Resuming batch: {batch_id} --")
        else:
            batch_id = submit_batch(client)

        poll_batch(client, batch_id)

        results, errors = retrieve_results(client, batch_id)
        retried         = retry_errors(client, errors, llm_inputs)
        results.update(retried)

    llm_rows, parse_errors = build_output_rows(results, llm_inputs)

    # Build lookup for fill_placeholders
    rows_dict = {(r["conversation_id"], r["message_index"]): r for r in llm_rows}
    fill_placeholders(rows_dict, llm_inputs)

    if parse_errors > 0:
        warnings_list.append(f"{parse_errors} parse error(s) — set to 'error'")

    # Combine rule + LLM rows
    all_rows = rule_rows + list(rows_dict.values())

    df = save_parquet(all_rows)

    try:
        make_figures(df, conv, config, func_class, emot_class)
    except Exception as e:
        msg = f"Figure generation error: {e}"
        print(f"  WARNING: {msg}")
        warnings_list.append(msg)

    report = generate_report(
        df, conv, batch_id, parse_errors, warnings_list,
        token_threshold, func_class, emot_class,
    )

    run_validation(df, conv, report)
    print("\nDone!")


if __name__ == "__main__":
    main()
