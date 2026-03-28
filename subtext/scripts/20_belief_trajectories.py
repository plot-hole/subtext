"""
Module 3.3: Belief Trajectory Analysis
Script: 20_belief_trajectories.py

Tracks how beliefs form, evolve, persist, and dissolve across time in
AI-assisted conversation. Three passes:
  Pass 1 — clusters hypotheses into belief threads (LLM batch)
  Pass 2 — classifies provenance of each hypothesis (LLM batch)
  Pass 3 — computes trajectory statistics per thread (local, no API)

Produces thread assignments, provenance labels, trajectory metrics, and
10 figures visualising belief dynamics over time.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/20_belief_trajectories.py

    # Dry run (build Pass 1 requests, don't submit):
    python scripts/20_belief_trajectories.py --dry-run

    # Resume Pass 1 (thread clustering):
    python scripts/20_belief_trajectories.py --resume-pass1-id msgbatch_xxxxx

    # Resume Pass 2 (provenance classification):
    python scripts/20_belief_trajectories.py --resume-pass2-id msgbatch_xxxxx

    # Threads only (skip provenance):
    python scripts/20_belief_trajectories.py --threads-only

    # Adjust clustering batch size:
    python scripts/20_belief_trajectories.py --cluster-batch-size 30
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
from scipy import stats as sp_stats
from tqdm import tqdm
import tiktoken

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HYPO_PATH       = os.path.join(BASE, "data", "processed", "hypotheses.parquet")
FRAME_PATH      = os.path.join(BASE, "data", "processed", "frame_adoption.parquet")
MSGS_PATH       = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
VOCAB_PATH      = os.path.join(BASE, "data", "processed", "vocab_transfer.parquet")
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")
INTERIM_DIR     = os.path.join(BASE, "data", "interim")

PASS1_BATCH_FILE    = os.path.join(INTERIM_DIR, "belief_pass1_batch.jsonl")
PASS1_BATCH_ID_FILE = os.path.join(INTERIM_DIR, "belief_pass1_batch_id.txt")
PASS1_RESULTS_CACHE = os.path.join(INTERIM_DIR, "belief_pass1_results.json")
PASS2_BATCH_FILE    = os.path.join(INTERIM_DIR, "belief_pass2_batch.jsonl")
PASS2_BATCH_ID_FILE = os.path.join(INTERIM_DIR, "belief_pass2_batch_id.txt")

OUT_THREADS     = os.path.join(BASE, "data", "processed", "belief_threads.parquet")
OUT_SUMMARIES   = os.path.join(BASE, "data", "processed", "belief_thread_summaries.parquet")
OUT_PROVENANCE  = os.path.join(BASE, "data", "processed", "belief_provenance.parquet")
OUT_REPORT      = os.path.join(BASE, "outputs", "reports", "belief_trajectory_report.json")
OUT_CATALOG     = os.path.join(BASE, "outputs", "reports", "belief_thread_catalog.csv")
FIG_DIR         = os.path.join(BASE, "outputs", "figures", "belief_trajectories")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Model & Pricing ---------------------------------------------------------
MODEL                   = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS_PASS1 = 4096   # thread assignments for batch of hypotheses
MAX_OUTPUT_TOKENS_PASS2 = 80     # single provenance label + reasoning
PRICE_INPUT_PER_MTOK    = 0.40   # batch discount
PRICE_OUTPUT_PER_MTOK   = 2.00   # batch discount

# -- Constants ---------------------------------------------------------------
CLUSTER_BATCH_SIZE = 50

VALID_TYPES      = ["prediction", "assessment", "intuition", "interpretation"]
VALID_CONFIDENCE = ["high", "moderate", "low"]
VALID_PROVENANCE = ["user_sourced", "ai_sourced", "interaction_emergent", "counter_position"]
VALID_FRAMES     = ["adopt", "extend", "redirect", "reject", "ignore", "steer"]

TYPE_ORDINAL       = {"intuition": 1, "interpretation": 2, "prediction": 3, "assessment": 4}
CONFIDENCE_ORDINAL = {"low": 1, "moderate": 2, "high": 3}

DORMANCY_THRESHOLD_HOURS = 48

# -- Style constants ---------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
DPI = 150

HYPO_COLORS = {
    "prediction":     "#4E79A7",
    "assessment":     "#59A14F",
    "intuition":      "#F28E2B",
    "interpretation": "#B07AA1",
}

PROVENANCE_COLORS = {
    "user_sourced":         "#59A14F",
    "ai_sourced":           "#E15759",
    "interaction_emergent": "#4E79A7",
    "counter_position":     "#F28E2B",
}

CONFIDENCE_COLORS = {
    "high":     "#E15759",
    "moderate": "#F28E2B",
    "low":      "#76B7B2",
}


# -- System Prompts ----------------------------------------------------------
SYSTEM_PROMPT_PASS1 = "\n".join([
    "You are a research assistant organizing hypotheses into belief threads.",
    "",
    "A belief thread is a cluster of hypotheses about the same entity, relationship, situation, or idea. Hypotheses belong to the same thread if a person revisiting them would say 'these are all about the same thing.'",
    "",
    "Thread naming rules:",
    "- Use descriptive lowercase snake_case names: person_a_intentions, project_validity, person_b_role, self_worth_pattern, ai_limitations, workplace_visibility",
    "- A thread can contain predictions, assessments, intuitions, and interpretations — what matters is the shared subject",
    "- A hypothesis can belong to only one thread",
    "- If a hypothesis doesn't fit any existing thread, create a new one",
    "- Prefer merging into existing threads over creating new ones unless the subject is genuinely different",
    "- Threads should be specific enough to be meaningful but broad enough to capture related beliefs",
    "",
    "Respond with ONLY a JSON object:",
    '{"assignments": [{"id": "hypothesis_id", "thread": "thread_name"}, ...]}',
])

SYSTEM_PROMPT_PASS2 = "\n".join([
    "You are a research assistant classifying the origin of beliefs expressed in human-AI conversation.",
    "",
    "Given a user's hypothesis and the surrounding conversation context, determine who introduced the core idea:",
    "",
    "- user_sourced: The user stated this belief before the AI suggested it. The idea originated with the user.",
    "- ai_sourced: The AI suggested this idea (or a close variant) in a preceding turn, and the user adopted or built on it.",
    "- interaction_emergent: The idea synthesizes elements from both speakers — neither stated it independently, but the dialogue produced it.",
    "- counter_position: The user formed this belief in direct opposition to what the AI suggested. The AI offered a frame and the user contradicted it.",
    "",
    "Look at the conversation flow carefully. The key question is: did this idea exist in the user's message BEFORE the AI introduced it, or did it appear AFTER?",
    "",
    "Respond with ONLY a JSON object:",
    '{"provenance": "user_sourced|ai_sourced|interaction_emergent|counter_position", "reasoning": "one sentence explaining the classification"}',
])


# -- Helpers -----------------------------------------------------------------
def clean(v):
    if isinstance(v, (float, np.floating)):
        val = float(v)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def clean_dict(d):
    if isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict(v) for v in d]
    return clean(d)


def figpath(name):
    return os.path.join(FIG_DIR, name)


def _strip_json_fences(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    return text.strip()


def _placeholder_figure(name, message):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=10)
    ax.set_title(name.replace(".png", "").replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(figpath(name), dpi=DPI)
    plt.close()


# ============================================================================
# STEP 0: LOAD DATA
# ============================================================================
def load_data():
    print("\n== Step 0: Load data ===================================================")

    for path, label in [
        (HYPO_PATH,  "hypotheses.parquet"),
        (MSGS_PATH,  "messages_clean.parquet"),
        (CONV_PATH,  "conversations_clean.parquet"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    hypotheses     = pd.read_parquet(HYPO_PATH)
    messages       = pd.read_parquet(MSGS_PATH)
    conversations  = pd.read_parquet(CONV_PATH)

    print(f"  Hypotheses               : {len(hypotheses):,}")
    print(f"  Messages                 : {len(messages):,}")
    print(f"  Conversations            : {len(conversations):,}")

    # Load frame adoption
    frame_adoption = None
    if os.path.exists(FRAME_PATH):
        frame_adoption = pd.read_parquet(FRAME_PATH)
        print(f"  Frame adoption labels    : {len(frame_adoption):,}")
    else:
        print(f"  Frame adoption           : NOT FOUND (correlation analysis will be skipped)")

    # Load functional classifications
    func_class = None
    if os.path.exists(FUNC_CLASS_PATH):
        func_class = pd.read_parquet(FUNC_CLASS_PATH)
        print(f"  Functional classifications: {len(func_class):,}")

    # Load emotional states
    emot_class = None
    if os.path.exists(EMOT_PATH):
        emot_class = pd.read_parquet(EMOT_PATH)
        print(f"  Emotional states         : {len(emot_class):,}")

    return hypotheses, messages, conversations, frame_adoption, func_class, emot_class


# ============================================================================
# PASS 1: THREAD CLUSTERING
# ============================================================================
def build_pass1_inputs(hypotheses, batch_size):
    """Build batched clustering inputs from hypotheses."""
    print("\n== Building Pass 1 inputs (thread clustering) ==========================")

    # Sort by timestamp for chronological processing
    hyp = hypotheses.sort_values("message_timestamp").reset_index(drop=True)

    batches = []
    for start in range(0, len(hyp), batch_size):
        batch = hyp.iloc[start:start + batch_size]
        items = []
        for _, row in batch.iterrows():
            items.append({
                "hypothesis_id": str(row["hypothesis_id"]),
                "resolved_text": str(row.get("resolved_text", "")),
                "topic":         str(row.get("topic", "")),
            })
        batches.append(items)

    print(f"  Total hypotheses         : {len(hyp):,}")
    print(f"  Batch size               : {batch_size}")
    print(f"  Total batches            : {len(batches):,}")
    return batches


def _format_pass1_message(batch_items, thread_list):
    """Format a Pass 1 batch request with running thread context."""
    thread_context = "(none yet — create new threads as needed)"
    if thread_list:
        lines = []
        for tname, example in sorted(thread_list.items()):
            lines.append(f"  - {tname}: e.g. \"{example[:100]}\"")
        thread_context = "\n".join(lines[:60])  # cap at 60 threads in context

    hypo_lines = []
    for item in batch_items:
        hypo_lines.append(
            f"  ID: {item['hypothesis_id']}\n"
            f"  Text: {item['resolved_text']}\n"
            f"  Topic: {item['topic']}"
        )

    return (
        f"Existing threads so far:\n{thread_context}\n\n"
        f"Assign each hypothesis below to a thread name:\n\n"
        + "\n\n".join(hypo_lines)
    )


def build_pass1_batch_file(batches, thread_list):
    """Write Pass 1 JSONL batch file."""
    print(f"\n  Building batch: {os.path.basename(PASS1_BATCH_FILE)}")
    with open(PASS1_BATCH_FILE, "w", encoding="utf-8") as f:
        for i, batch_items in enumerate(batches):
            request = {
                "custom_id": f"cluster_batch_{i:04d}",
                "params": {
                    "model":      MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS_PASS1,
                    "system":     SYSTEM_PROMPT_PASS1,
                    "messages":   [{"role": "user", "content": _format_pass1_message(batch_items, thread_list)}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    count = sum(1 for _ in open(PASS1_BATCH_FILE, encoding="utf-8"))
    print(f"  Batch file               : {PASS1_BATCH_FILE}")
    print(f"  Total requests           : {count:,}")
    return count


def parse_pass1(results, batches):
    """Parse Pass 1 thread clustering results."""
    print("\n  Parsing Pass 1 thread clustering responses...")
    assignments  = {}  # hypothesis_id -> thread_name
    thread_list  = {}  # thread_name -> example text
    parse_errors = 0

    # Build lookup from batches
    batch_lookup = {}
    for i, batch_items in enumerate(batches):
        batch_lookup[f"cluster_batch_{i:04d}"] = {
            item["hypothesis_id"]: item["resolved_text"] for item in batch_items
        }

    for custom_id, res in tqdm(results.items(), desc="Parsing Pass 1"):
        try:
            text = _strip_json_fences(res.get("raw", ""))
            data = json.loads(text)
            items = data.get("assignments", [])
            if not isinstance(items, list):
                parse_errors += 1
                continue

            lookup = batch_lookup.get(custom_id, {})
            for item in items:
                hid    = str(item.get("id", ""))
                thread = str(item.get("thread", "")).strip().lower().replace(" ", "_")
                if hid and thread:
                    assignments[hid] = thread
                    if thread not in thread_list and hid in lookup:
                        thread_list[thread] = lookup[hid][:100]

        except Exception:
            parse_errors += 1

    n_threads = len(set(assignments.values()))
    print(f"  Assignments parsed       : {len(assignments):,}")
    print(f"  Unique threads           : {n_threads:,}")
    print(f"  Parse errors             : {parse_errors:,}")
    return assignments, thread_list, parse_errors


# ============================================================================
# PASS 2: PROVENANCE CLASSIFICATION
# ============================================================================
def build_pass2_inputs(hypotheses, messages, frame_adoption):
    """Build provenance classification inputs with conversation context."""
    print("\n== Building Pass 2 inputs (provenance classification) ==================")
    enc = tiktoken.get_encoding("cl100k_base")

    # Pre-sort messages by conversation and index
    msgs_sorted = messages.sort_values(["conversation_id", "msg_index"])
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

    # Build frame adoption lookup: (conversation_id, message_index) -> label
    frame_map = {}
    if frame_adoption is not None:
        for _, row in frame_adoption.iterrows():
            key = (str(row["conversation_id"]), int(row["message_index"]))
            frame_map[key] = str(row.get("frame_adoption", ""))

    inputs = {}
    for _, row in tqdm(hypotheses.iterrows(), total=len(hypotheses), desc="Building Pass 2"):
        hid     = str(row["hypothesis_id"])
        cid     = str(row["conversation_id"])
        msg_idx = int(row["message_index"])
        resolved = str(row.get("resolved_text", ""))

        # Get frame adoption label for this message
        frame_label = frame_map.get((cid, msg_idx), "")

        # Build context window: preceding 3-4 exchanges
        context_lines = []
        context_tokens = 0
        max_ctx = 1500
        if cid in conv_msgs:
            preceding = [m for m in conv_msgs[cid] if m["msg_index"] < msg_idx]
            preceding = preceding[-8:]  # last 4 exchanges
            for m in reversed(preceding):
                m_text = m["text"]
                try:
                    m_toks = len(enc.encode(m_text))
                except Exception:
                    m_toks = len(m_text) // 4
                if context_tokens + m_toks > max_ctx:
                    remaining = max_ctx - context_tokens
                    if remaining > 20:
                        try:
                            m_text = enc.decode(enc.encode(m_text)[:remaining])
                        except Exception:
                            m_text = m_text[:remaining * 4]
                        context_lines.insert(0, f"{m['role'].title()}: {m_text}")
                    break
                context_lines.insert(0, f"{m['role'].title()}: {m_text}")
                context_tokens += m_toks

        context_str = "\n\n".join(context_lines) if context_lines else "(no preceding context)"

        inputs[hid] = {
            "hypothesis_id": hid,
            "resolved_text": resolved,
            "context":       context_str,
            "frame_label":   frame_label,
        }

    print(f"  Pass 2 inputs built      : {len(inputs):,}")
    return inputs


def _format_pass2_message(inp):
    frame_info = f"\nFrame adoption label for this exchange: {inp['frame_label']}" if inp['frame_label'] else ""
    return (
        f"Context:\n{inp['context']}\n\n"
        f"User's hypothesis:\n{inp['resolved_text']}"
        f"{frame_info}\n"
    )


def build_pass2_batch_file(inputs):
    """Write Pass 2 JSONL batch file."""
    print(f"\n  Building batch: {os.path.basename(PASS2_BATCH_FILE)}")
    with open(PASS2_BATCH_FILE, "w", encoding="utf-8") as f:
        for hid, inp in inputs.items():
            request = {
                "custom_id": hid,
                "params": {
                    "model":      MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS_PASS2,
                    "system":     SYSTEM_PROMPT_PASS2,
                    "messages":   [{"role": "user", "content": _format_pass2_message(inp)}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    count = sum(1 for _ in open(PASS2_BATCH_FILE, encoding="utf-8"))
    print(f"  Batch file               : {PASS2_BATCH_FILE}")
    print(f"  Total requests           : {count:,}")
    return count


def parse_pass2(results):
    """Parse Pass 2 provenance classification results."""
    print("\n  Parsing Pass 2 provenance responses...")
    parsed       = {}
    parse_errors = 0

    for hid, res in tqdm(results.items(), desc="Parsing Pass 2"):
        try:
            text = _strip_json_fences(res.get("raw", ""))
            data = json.loads(text)
            prov = str(data.get("provenance", "")).strip().lower()
            reasoning = str(data.get("reasoning", "")).strip()

            if prov not in VALID_PROVENANCE:
                prov = "user_sourced"  # safe default

            parsed[hid] = {
                "provenance":     prov,
                "reasoning":      reasoning,
                "input_tokens":   res.get("input_tokens", 0),
                "output_tokens":  res.get("output_tokens", 0),
            }
        except Exception:
            parse_errors += 1
            parsed[hid] = {
                "provenance":     "user_sourced",
                "reasoning":      "parse_error",
                "input_tokens":   res.get("input_tokens", 0),
                "output_tokens":  res.get("output_tokens", 0),
            }

    prov_counts = pd.Series([v["provenance"] for v in parsed.values()]).value_counts()
    print(f"  Provenance parsed        : {len(parsed):,}")
    print(f"  Parse errors             : {parse_errors:,}")
    for p in VALID_PROVENANCE:
        c = int(prov_counts.get(p, 0))
        print(f"    {p:25s}: {c:,}")
    return parsed, parse_errors


# ============================================================================
# BATCH SUBMISSION / POLLING / RETRIEVAL (shared)
# ============================================================================
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

    with open(batch_id_file, "w") as f:
        f.write(batch.id)
    return batch.id


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


# ============================================================================
# PASS 3: TRAJECTORY COMPUTATION (local, no API)
# ============================================================================
def compute_trajectories(threads_df):
    """Compute per-thread trajectory statistics."""
    print("\n== Pass 3: Computing trajectories ======================================")

    if len(threads_df) == 0:
        print("  WARNING: No thread data to compute trajectories from.")
        return pd.DataFrame()

    summaries = []

    for thread_name, group in tqdm(threads_df.groupby("thread"), desc="Computing trajectories"):
        g = group.sort_values("message_timestamp")
        n = len(g)

        first_seen = g["message_timestamp"].min()
        last_seen  = g["message_timestamp"].max()

        # Span in days
        if pd.notna(first_seen) and pd.notna(last_seen):
            span_days = (last_seen - first_seen).total_seconds() / 86400
        else:
            span_days = 0.0

        active_conversations = g["conversation_id"].nunique()

        # Type escalation slope
        type_vals = g["type"].astype(str).map(TYPE_ORDINAL).dropna()
        type_slope = 0.0
        if len(type_vals) >= 3:
            try:
                x = np.arange(len(type_vals))
                slope, _, _, _, _ = sp_stats.linregress(x, type_vals.values)
                type_slope = float(slope)
            except Exception:
                pass

        # Confidence trend slope
        conf_vals = g["confidence_level"].astype(str).map(CONFIDENCE_ORDINAL).dropna()
        conf_slope = 0.0
        if len(conf_vals) >= 3:
            try:
                x = np.arange(len(conf_vals))
                slope, _, _, _, _ = sp_stats.linregress(x, conf_vals.values)
                conf_slope = float(slope)
            except Exception:
                pass

        # Dormancy and reactivation
        timestamps = g["message_timestamp"].dropna().sort_values()
        dormancy_periods  = 0
        reactivation_count = 0
        if len(timestamps) >= 2:
            gaps = timestamps.diff().dt.total_seconds() / 3600  # hours
            dormancy_periods = int((gaps > DORMANCY_THRESHOLD_HOURS).sum())
            reactivation_count = dormancy_periods  # each dormancy ends in reactivation

        # Provenance mix
        prov_counts = g["provenance"].astype(str).value_counts()
        prov_total  = max(prov_counts.sum(), 1)
        pct_user     = float(prov_counts.get("user_sourced", 0)) / prov_total
        pct_ai       = float(prov_counts.get("ai_sourced", 0)) / prov_total
        pct_emergent = float(prov_counts.get("interaction_emergent", 0)) / prov_total
        pct_counter  = float(prov_counts.get("counter_position", 0)) / prov_total
        dominant_prov = str(prov_counts.index[0]) if len(prov_counts) > 0 else "unknown"

        # Frame adoption correlation
        pct_after_adopt  = 0.0
        pct_after_reject = 0.0
        if "nearest_frame_label" in g.columns:
            frame_vals = g["nearest_frame_label"].astype(str)
            frame_total = max(frame_vals.isin(VALID_FRAMES).sum(), 1)
            pct_after_adopt  = float(frame_vals.isin(["adopt", "extend"]).sum()) / frame_total
            pct_after_reject = float(frame_vals.isin(["reject", "redirect"]).sum()) / frame_total

        summaries.append({
            "thread":                  thread_name,
            "hypothesis_count":        n,
            "first_seen":              first_seen,
            "last_seen":               last_seen,
            "span_days":               round(span_days, 1),
            "active_conversations":    active_conversations,
            "type_escalation_slope":   round(type_slope, 4),
            "confidence_trend_slope":  round(conf_slope, 4),
            "dormancy_periods":        dormancy_periods,
            "reactivation_count":      reactivation_count,
            "pct_user_sourced":        round(pct_user, 3),
            "pct_ai_sourced":          round(pct_ai, 3),
            "pct_interaction_emergent": round(pct_emergent, 3),
            "pct_counter_position":    round(pct_counter, 3),
            "dominant_provenance":     dominant_prov,
            "pct_after_adopt_frame":   round(pct_after_adopt, 3),
            "pct_after_reject_frame":  round(pct_after_reject, 3),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("hypothesis_count", ascending=False).reset_index(drop=True)

    print(f"  Threads computed         : {len(summary_df):,}")
    print(f"  Largest thread           : {summary_df.iloc[0]['thread']} ({summary_df.iloc[0]['hypothesis_count']:,})" if len(summary_df) > 0 else "")

    # Trajectory stats
    if len(summary_df) > 0:
        hardening  = (summary_df["confidence_trend_slope"] > 0.01).sum()
        softening  = (summary_df["confidence_trend_slope"] < -0.01).sum()
        stable     = len(summary_df) - hardening - softening
        escalating = (summary_df["type_escalation_slope"] > 0.01).sum()
        regressing = (summary_df["type_escalation_slope"] < -0.01).sum()
        print(f"  Hardening threads        : {hardening:,}")
        print(f"  Softening threads        : {softening:,}")
        print(f"  Stable threads           : {stable:,}")
        print(f"  Type escalating          : {escalating:,}")
        print(f"  Type regressing          : {regressing:,}")

    return summary_df


# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_results(hypotheses, assignments, provenance_parsed, frame_adoption, summary_df):
    print("\n== Step 5: Save results ================================================")

    # Build threads dataframe
    hyp = hypotheses.copy()
    hyp["thread"] = hyp["hypothesis_id"].map(assignments).fillna("unassigned")

    # Add provenance
    if provenance_parsed:
        hyp["provenance"]           = hyp["hypothesis_id"].map(
            lambda h: provenance_parsed.get(h, {}).get("provenance", "unknown"))
        hyp["provenance_reasoning"] = hyp["hypothesis_id"].map(
            lambda h: provenance_parsed.get(h, {}).get("reasoning", ""))
    else:
        hyp["provenance"]           = "unknown"
        hyp["provenance_reasoning"] = ""

    # Add nearest frame label
    if frame_adoption is not None:
        frame_map = {}
        for _, row in frame_adoption.iterrows():
            key = (str(row["conversation_id"]), int(row["message_index"]))
            frame_map[key] = str(row.get("frame_adoption", ""))
        hyp["nearest_frame_label"] = hyp.apply(
            lambda r: frame_map.get((str(r["conversation_id"]), int(r["message_index"])), ""),
            axis=1,
        )
    else:
        hyp["nearest_frame_label"] = ""

    # Save belief_threads.parquet
    thread_cols = [
        "hypothesis_id", "thread", "provenance", "provenance_reasoning",
        "message_timestamp", "type", "confidence_level", "resolved_text",
        "conversation_id", "message_index", "nearest_frame_label",
    ]
    threads_df = hyp[[c for c in thread_cols if c in hyp.columns]].copy()
    threads_df.to_parquet(OUT_THREADS, index=False)
    print(f"  Saved: {OUT_THREADS} ({len(threads_df):,} rows)")

    # Save belief_thread_summaries.parquet
    if len(summary_df) > 0:
        summary_df.to_parquet(OUT_SUMMARIES, index=False)
        print(f"  Saved: {OUT_SUMMARIES} ({len(summary_df):,} rows)")

    # Save belief_provenance.parquet
    if provenance_parsed:
        prov_rows = []
        for hid, p in provenance_parsed.items():
            frame_label = ""
            match = hyp[hyp["hypothesis_id"] == hid]
            if len(match) > 0:
                frame_label = str(match.iloc[0].get("nearest_frame_label", ""))
            prov_rows.append({
                "hypothesis_id":      hid,
                "provenance":         p["provenance"],
                "provenance_reasoning": p["reasoning"],
                "nearest_frame_label": frame_label,
                "pass2_input_tokens":  p.get("input_tokens", 0),
                "pass2_output_tokens": p.get("output_tokens", 0),
            })
        prov_df = pd.DataFrame(prov_rows)
        prov_df.to_parquet(OUT_PROVENANCE, index=False)
        print(f"  Saved: {OUT_PROVENANCE} ({len(prov_df):,} rows)")

    # Save thread catalog CSV
    if len(summary_df) > 0:
        # Get an example hypothesis for each thread
        examples = {}
        for thread_name, group in threads_df.groupby("thread"):
            examples[thread_name] = str(group.iloc[0].get("resolved_text", ""))[:120]

        catalog = summary_df[["thread", "hypothesis_count", "span_days",
                               "dominant_provenance", "confidence_trend_slope",
                               "type_escalation_slope"]].copy()
        catalog["example_hypothesis"] = catalog["thread"].map(examples)
        catalog.to_csv(OUT_CATALOG, index=False, encoding="utf-8")
        print(f"  Saved: {OUT_CATALOG}")

    return threads_df


# ============================================================================
# FIGURES
# ============================================================================
def make_figures(threads_df, summary_df):
    print("\n== Step 6: Generating figures ==========================================")

    if len(threads_df) == 0 or len(summary_df) == 0:
        print("  No data — creating placeholder figures only")
        for name in [
            "thread_size_distribution.png", "thread_timeline.png",
            "provenance_distribution.png", "provenance_by_frame.png",
            "confidence_trajectory.png", "type_escalation.png",
            "provenance_sankey.png", "dormancy_reactivation.png",
            "ai_influence_quadrant.png", "belief_trajectory_dashboard.png",
        ]:
            _placeholder_figure(name, "No data available")
        return

    _fig1_thread_sizes(summary_df)
    _fig2_thread_timeline(threads_df, summary_df)
    _fig3_provenance_distribution(threads_df, summary_df)
    _fig4_provenance_by_frame(threads_df)
    _fig5_confidence_trajectory(threads_df, summary_df)
    _fig6_type_escalation(threads_df, summary_df)
    _fig7_provenance_flow(threads_df)
    _fig8_dormancy(summary_df)
    _fig9_ai_influence_quadrant(summary_df)
    _fig10_dashboard(threads_df, summary_df)
    print("  All figures saved.")


def _fig1_thread_sizes(summary_df):
    """Horizontal bar chart of top 25 threads by hypothesis count."""
    top = summary_df.head(25)
    colors = [PROVENANCE_COLORS.get(p, COLOR_PRIMARY) for p in top["dominant_provenance"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top)), top["hypothesis_count"].values, color=colors)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["thread"].str.replace("_", " "), fontsize=8)
    ax.invert_yaxis()

    for bar, count in zip(bars, top["hypothesis_count"].values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=8)

    ax.set_xlabel("Number of Hypotheses")
    ax.set_title("Top 25 Belief Threads by Size (colored by dominant provenance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p.replace("_", " "))
                       for p, c in PROVENANCE_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(figpath("thread_size_distribution.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: thread_size_distribution.png")


def _fig2_thread_timeline(threads_df, summary_df):
    """Swimlane chart of top 15 threads over time."""
    top_threads = summary_df.head(15)["thread"].tolist()
    subset = threads_df[threads_df["thread"].isin(top_threads)].copy()

    if len(subset) == 0 or "message_timestamp" not in subset.columns:
        _placeholder_figure("thread_timeline.png", "No timeline data")
        return

    subset["ts"] = pd.to_datetime(subset["message_timestamp"], errors="coerce")
    subset = subset.dropna(subset=["ts"])

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, thread in enumerate(top_threads):
        t_data = subset[subset["thread"] == thread]
        if len(t_data) == 0:
            continue

        # Map confidence to dot size
        sizes = t_data["confidence_level"].astype(str).map(
            {"high": 40, "moderate": 20, "low": 8}).fillna(15)
        colors = t_data["type"].astype(str).map(HYPO_COLORS).fillna(COLOR_PRIMARY)

        ax.scatter(t_data["ts"], [i] * len(t_data), s=sizes, c=colors, alpha=0.7, zorder=2)

    ax.set_yticks(range(len(top_threads)))
    ax.set_yticklabels([t.replace("_", " ") for t in top_threads], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Date")
    ax.set_title("Belief Thread Timeline (top 15 threads)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for types
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                              markersize=8, label=t) for t, c in HYPO_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(figpath("thread_timeline.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: thread_timeline.png")


def _fig3_provenance_distribution(threads_df, summary_df):
    """Stacked bar chart: top 15 threads x provenance."""
    top_threads = summary_df.head(15)["thread"].tolist()
    subset = threads_df[threads_df["thread"].isin(top_threads)]

    ct = pd.crosstab(subset["thread"].astype(str), subset["provenance"].astype(str))
    ct = ct.reindex(index=top_threads)
    ct = ct.reindex(columns=[p for p in VALID_PROVENANCE if p in ct.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_threads))
    bottom = np.zeros(len(top_threads))

    for prov in ct.columns:
        vals = ct[prov].values
        ax.barh(x, vals, left=bottom,
                color=PROVENANCE_COLORS.get(prov, COLOR_PRIMARY),
                label=prov.replace("_", " "), height=0.7)
        bottom += vals

    ax.set_yticks(x)
    ax.set_yticklabels([t.replace("_", " ") for t in top_threads], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Hypotheses")
    ax.set_title("Provenance Distribution by Thread (top 15)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("provenance_distribution.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: provenance_distribution.png")


def _fig4_provenance_by_frame(threads_df):
    """Grouped bar chart: frame adoption label x provenance."""
    if "nearest_frame_label" not in threads_df.columns:
        _placeholder_figure("provenance_by_frame.png", "No frame adoption data")
        return

    valid = threads_df[threads_df["nearest_frame_label"].isin(VALID_FRAMES)]
    if len(valid) == 0:
        _placeholder_figure("provenance_by_frame.png", "No matching frame data")
        return

    ct = pd.crosstab(valid["nearest_frame_label"].astype(str),
                     valid["provenance"].astype(str))
    frames = [f for f in VALID_FRAMES if f in ct.index]
    provs  = [p for p in VALID_PROVENANCE if p in ct.columns]
    ct = ct.reindex(index=frames, columns=provs, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(frames))
    width = 0.8 / max(len(provs), 1)

    for i, prov in enumerate(provs):
        offset = (i - len(provs) / 2 + 0.5) * width
        ax.bar(x + offset, ct[prov].values, width,
               color=PROVENANCE_COLORS.get(prov, COLOR_PRIMARY),
               label=prov.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(frames, fontsize=10)
    ax.set_xlabel("Frame Adoption Label")
    ax.set_ylabel("Hypothesis Count")
    ax.set_title("Belief Provenance by Frame Adoption Response")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("provenance_by_frame.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: provenance_by_frame.png")


def _fig5_confidence_trajectory(threads_df, summary_df):
    """Small multiples: confidence over time for top 12 threads."""
    top_threads = summary_df.head(12)["thread"].tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for i, thread in enumerate(top_threads):
        ax = axes_flat[i]
        t_data = threads_df[threads_df["thread"] == thread].sort_values("message_timestamp")
        if len(t_data) == 0:
            ax.set_visible(False)
            continue

        conf_vals = t_data["confidence_level"].astype(str).map(CONFIDENCE_ORDINAL).values
        x = np.arange(len(conf_vals))

        ax.plot(x, conf_vals, "o-", color=COLOR_PRIMARY, markersize=3, linewidth=1)
        if len(conf_vals) >= 3:
            try:
                slope, intercept, _, _, _ = sp_stats.linregress(x, conf_vals)
                ax.plot(x, intercept + slope * x, "--", color=COLOR_ACCENT, linewidth=1, alpha=0.7)
            except Exception:
                pass

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["low", "mod", "high"], fontsize=7)
        ax.set_title(thread.replace("_", " ")[:25], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for i in range(len(top_threads), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Confidence Trajectory per Thread (top 12)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath("confidence_trajectory.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confidence_trajectory.png")


def _fig6_type_escalation(threads_df, summary_df):
    """Small multiples: type ordinal over time for top 12 threads."""
    top_threads = summary_df.head(12)["thread"].tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for i, thread in enumerate(top_threads):
        ax = axes_flat[i]
        t_data = threads_df[threads_df["thread"] == thread].sort_values("message_timestamp")
        if len(t_data) == 0:
            ax.set_visible(False)
            continue

        type_vals = t_data["type"].astype(str).map(TYPE_ORDINAL).values
        x = np.arange(len(type_vals))

        colors = [HYPO_COLORS.get(t, COLOR_PRIMARY) for t in t_data["type"].astype(str)]
        ax.scatter(x, type_vals, c=colors, s=15, zorder=2)
        if len(type_vals) >= 3:
            try:
                slope, intercept, _, _, _ = sp_stats.linregress(x, type_vals)
                ax.plot(x, intercept + slope * x, "--", color=COLOR_SECONDARY, linewidth=1, alpha=0.7)
            except Exception:
                pass

        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["intuit", "interp", "pred", "assess"], fontsize=6)
        ax.set_title(thread.replace("_", " ")[:25], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(len(top_threads), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Type Escalation per Thread (top 12)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath("type_escalation.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: type_escalation.png")


def _fig7_provenance_flow(threads_df):
    """Provenance flow: frame label -> provenance -> hypothesis type."""
    # Simplified as a grouped heatmap since matplotlib Sankey is limited
    if "nearest_frame_label" not in threads_df.columns:
        _placeholder_figure("provenance_sankey.png", "No frame data")
        return

    valid = threads_df[threads_df["nearest_frame_label"].isin(VALID_FRAMES)].copy()
    if len(valid) == 0:
        _placeholder_figure("provenance_sankey.png", "No matching data")
        return

    # Frame -> Provenance heatmap
    ct1 = pd.crosstab(valid["nearest_frame_label"].astype(str),
                      valid["provenance"].astype(str))
    # Provenance -> Type heatmap
    ct2 = pd.crosstab(valid["provenance"].astype(str),
                      valid["type"].astype(str))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: frame -> provenance
    frames = [f for f in VALID_FRAMES if f in ct1.index]
    provs  = [p for p in VALID_PROVENANCE if p in ct1.columns]
    ct1 = ct1.reindex(index=frames, columns=provs, fill_value=0)
    im1 = ax1.imshow(ct1.values, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(provs)))
    ax1.set_xticklabels([p.replace("_", "\n") for p in provs], fontsize=7)
    ax1.set_yticks(range(len(frames)))
    ax1.set_yticklabels(frames, fontsize=9)
    for i in range(len(frames)):
        for j in range(len(provs)):
            v = int(ct1.values[i, j])
            if v > 0:
                ax1.text(j, i, str(v), ha="center", va="center", fontsize=8,
                         color="white" if v > ct1.values.max() * 0.6 else "black")
    ax1.set_title("Frame Response -> Provenance")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Right: provenance -> type
    provs2 = [p for p in VALID_PROVENANCE if p in ct2.index]
    types  = [t for t in VALID_TYPES if t in ct2.columns]
    ct2 = ct2.reindex(index=provs2, columns=types, fill_value=0)
    im2 = ax2.imshow(ct2.values, cmap="YlGnBu", aspect="auto")
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, fontsize=9)
    ax2.set_yticks(range(len(provs2)))
    ax2.set_yticklabels([p.replace("_", "\n") for p in provs2], fontsize=7)
    for i in range(len(provs2)):
        for j in range(len(types)):
            v = int(ct2.values[i, j])
            if v > 0:
                ax2.text(j, i, str(v), ha="center", va="center", fontsize=8,
                         color="white" if v > ct2.values.max() * 0.6 else "black")
    ax2.set_title("Provenance -> Hypothesis Type")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle("Belief Formation Flow: Frame Response -> Provenance -> Type",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figpath("provenance_sankey.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: provenance_sankey.png")


def _fig8_dormancy(summary_df):
    """Scatter: thread span vs reactivation count."""
    if len(summary_df) == 0:
        _placeholder_figure("dormancy_reactivation.png", "No data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PROVENANCE_COLORS.get(p, COLOR_PRIMARY) for p in summary_df["dominant_provenance"]]
    sizes  = np.clip(summary_df["hypothesis_count"].values * 3, 15, 300)

    ax.scatter(summary_df["span_days"], summary_df["reactivation_count"],
               s=sizes, c=colors, alpha=0.6, edgecolors="white", linewidth=0.5)

    # Label top threads
    for _, row in summary_df.head(8).iterrows():
        ax.annotate(row["thread"].replace("_", " ")[:20],
                    (row["span_days"], row["reactivation_count"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Thread Span (days)")
    ax.set_ylabel("Reactivation Count")
    ax.set_title("Thread Persistence: Span vs Reactivation (size = hypothesis count)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p.replace("_", " "))
                       for p, c in PROVENANCE_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(figpath("dormancy_reactivation.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: dormancy_reactivation.png")


def _fig9_ai_influence_quadrant(summary_df):
    """Four-quadrant chart: % AI-sourced vs confidence trend."""
    if len(summary_df) == 0:
        _placeholder_figure("ai_influence_quadrant.png", "No data")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    x = summary_df["pct_ai_sourced"].values
    y = summary_df["confidence_trend_slope"].values
    sizes = np.clip(summary_df["hypothesis_count"].values * 3, 15, 300)
    colors = [PROVENANCE_COLORS.get(p, COLOR_PRIMARY) for p in summary_df["dominant_provenance"]]

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors="white", linewidth=0.5)

    # Quadrant lines
    ax.axhline(y=0, color=COLOR_SECONDARY, linestyle="--", alpha=0.3)
    ax.axvline(x=0.5, color=COLOR_SECONDARY, linestyle="--", alpha=0.3)

    # Quadrant labels
    ax.text(0.75, max(y) * 0.8, "AI-shaped\n& hardening", ha="center", fontsize=8,
            color=COLOR_SECONDARY, alpha=0.7, style="italic")
    ax.text(0.25, max(y) * 0.8, "User-owned\n& hardening", ha="center", fontsize=8,
            color=COLOR_SECONDARY, alpha=0.7, style="italic")
    ax.text(0.75, min(y) * 0.8, "AI-shaped\n& fading", ha="center", fontsize=8,
            color=COLOR_SECONDARY, alpha=0.7, style="italic")
    ax.text(0.25, min(y) * 0.8, "User-owned\n& softening", ha="center", fontsize=8,
            color=COLOR_SECONDARY, alpha=0.7, style="italic")

    # Label top threads
    for _, row in summary_df.head(10).iterrows():
        ax.annotate(row["thread"].replace("_", " ")[:20],
                    (row["pct_ai_sourced"], row["confidence_trend_slope"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("% AI-Sourced Hypotheses")
    ax.set_ylabel("Confidence Trend Slope")
    ax.set_title("AI Influence Quadrant (size = hypothesis count)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figpath("ai_influence_quadrant.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: ai_influence_quadrant.png")


def _fig10_dashboard(threads_df, summary_df):
    """Summary dashboard panel."""
    total_threads = len(summary_df)
    total_hypos   = len(threads_df)

    # Provenance totals
    prov_counts = threads_df["provenance"].astype(str).value_counts()
    pct_user = round(prov_counts.get("user_sourced", 0) / max(total_hypos, 1) * 100, 1)
    pct_ai   = round(prov_counts.get("ai_sourced", 0) / max(total_hypos, 1) * 100, 1)

    # Strongest hardening / softening
    strongest_hard = ""
    strongest_soft = ""
    most_reactivated = ""
    if len(summary_df) > 0:
        idx_hard = summary_df["confidence_trend_slope"].idxmax()
        idx_soft = summary_df["confidence_trend_slope"].idxmin()
        idx_react = summary_df["reactivation_count"].idxmax()
        strongest_hard   = str(summary_df.loc[idx_hard, "thread"])
        strongest_soft   = str(summary_df.loc[idx_soft, "thread"])
        most_reactivated = str(summary_df.loc[idx_react, "thread"])

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # Panel A: total threads
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.text(0.5, 0.6, f"{total_threads}", ha="center", va="center",
              fontsize=28, fontweight="bold", color=COLOR_PRIMARY, transform=ax_a.transAxes)
    ax_a.text(0.5, 0.25, "belief threads", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_a.transAxes)
    ax_a.set_title("Threads Identified", fontsize=11)

    # Panel B: provenance split
    ax_b = fig.add_subplot(gs[0, 1])
    if len(prov_counts) > 0:
        provs = [p for p in VALID_PROVENANCE if p in prov_counts.index]
        sizes = [int(prov_counts.get(p, 0)) for p in provs]
        cols  = [PROVENANCE_COLORS.get(p, COLOR_PRIMARY) for p in provs]
        ax_b.pie(sizes, labels=[p.replace("_", "\n") for p in provs],
                 colors=cols, autopct="%1.0f%%", startangle=90,
                 textprops={"fontsize": 7})
    ax_b.set_title("Provenance Split", fontsize=11)

    # Panel C: user vs AI
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.text(0.5, 0.65, f"{pct_user}%", ha="center", va="center",
              fontsize=20, fontweight="bold", color=PROVENANCE_COLORS["user_sourced"],
              transform=ax_c.transAxes)
    ax_c.text(0.5, 0.45, "user-sourced", ha="center", va="center",
              fontsize=9, color=COLOR_SECONDARY, transform=ax_c.transAxes)
    ax_c.text(0.5, 0.25, f"{pct_ai}% AI-sourced", ha="center", va="center",
              fontsize=10, color=PROVENANCE_COLORS["ai_sourced"], transform=ax_c.transAxes)
    ax_c.set_title("Source Balance", fontsize=11)

    # Panel D: strongest hardening
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.axis("off")
    ax_d.text(0.5, 0.55, strongest_hard.replace("_", "\n")[:30],
              ha="center", va="center", fontsize=9, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9"))
    ax_d.set_title("Strongest Hardening", fontsize=10)

    # Panel E: strongest softening
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.axis("off")
    ax_e.text(0.5, 0.55, strongest_soft.replace("_", "\n")[:30],
              ha="center", va="center", fontsize=9, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0"))
    ax_e.set_title("Strongest Softening", fontsize=10)

    # Panel F: most reactivated
    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.axis("off")
    ax_f.text(0.5, 0.55, most_reactivated.replace("_", "\n")[:30],
              ha="center", va="center", fontsize=9, transform=ax_f.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EDE7F6"))
    ax_f.set_title("Most Reactivated", fontsize=10)

    # Panel G: hardening vs softening count
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.axis("off")
    hardening = int((summary_df["confidence_trend_slope"] > 0.01).sum())
    softening = int((summary_df["confidence_trend_slope"] < -0.01).sum())
    ax_g.text(0.5, 0.6, f"{hardening} hardening\n{softening} softening",
              ha="center", va="center", fontsize=11, transform=ax_g.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_g.set_title("Trajectory Direction", fontsize=10)

    # Panel H: total hypotheses
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.axis("off")
    ax_h.text(0.5, 0.6, f"{total_hypos:,}", ha="center", va="center",
              fontsize=24, fontweight="bold", color=COLOR_PRIMARY, transform=ax_h.transAxes)
    ax_h.text(0.5, 0.25, "hypotheses analyzed", ha="center", va="center",
              fontsize=10, color=COLOR_SECONDARY, transform=ax_h.transAxes)
    ax_h.set_title("Total Hypotheses", fontsize=11)

    plt.suptitle("Belief Trajectory Dashboard", fontsize=14, fontweight="bold")
    plt.savefig(figpath("belief_trajectory_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: belief_trajectory_dashboard.png")


# ============================================================================
# REPORT
# ============================================================================
def generate_report(threads_df, summary_df, provenance_parsed,
                    pass1_batch_id, pass2_batch_id, warnings_list):
    print("\n== Step 7: Generate report =============================================")

    total_hypos   = len(threads_df)
    total_threads = len(summary_df)
    unassigned    = int((threads_df["thread"] == "unassigned").sum()) if total_hypos > 0 else 0

    # Thread stats
    thread_stats = {}
    if total_threads > 0:
        thread_stats = {
            "total_threads":            total_threads,
            "mean_thread_size":         round(summary_df["hypothesis_count"].mean(), 1),
            "median_thread_size":       int(summary_df["hypothesis_count"].median()),
            "max_thread_size":          int(summary_df["hypothesis_count"].max()),
            "largest_thread":           str(summary_df.iloc[0]["thread"]),
            "threads_with_reactivation": int((summary_df["reactivation_count"] > 0).sum()),
            "mean_span_days":           round(summary_df["span_days"].mean(), 1),
        }

    # Provenance distribution
    prov_dist = {}
    if total_hypos > 0:
        prov_counts = threads_df["provenance"].astype(str).value_counts()
        for p in VALID_PROVENANCE:
            c = int(prov_counts.get(p, 0))
            prov_dist[p] = {"count": c, "pct": round(c / total_hypos * 100, 1)}

    # Trajectory stats
    traj_stats = {}
    if total_threads > 0:
        traj_stats = {
            "threads_hardening":       int((summary_df["confidence_trend_slope"] > 0.01).sum()),
            "threads_softening":       int((summary_df["confidence_trend_slope"] < -0.01).sum()),
            "threads_stable":          int(((summary_df["confidence_trend_slope"] >= -0.01) &
                                            (summary_df["confidence_trend_slope"] <= 0.01)).sum()),
            "threads_escalating_type": int((summary_df["type_escalation_slope"] > 0.01).sum()),
            "threads_regressing_type": int((summary_df["type_escalation_slope"] < -0.01).sum()),
            "mean_confidence_slope":   round(summary_df["confidence_trend_slope"].mean(), 4),
            "mean_type_slope":         round(summary_df["type_escalation_slope"].mean(), 4),
        }

    # Frame adoption correlation
    frame_corr = {}
    if total_hypos > 0 and "nearest_frame_label" in threads_df.columns:
        valid_frames = threads_df[threads_df["nearest_frame_label"].isin(VALID_FRAMES)]
        if len(valid_frames) > 0:
            frame_corr["hypotheses_after_adopt"]  = int(valid_frames["nearest_frame_label"].isin(["adopt", "extend"]).sum())
            frame_corr["hypotheses_after_reject"] = int(valid_frames["nearest_frame_label"].isin(["reject", "redirect"]).sum())

            adopted = valid_frames[valid_frames["nearest_frame_label"].isin(["adopt", "extend"])]
            rejected = valid_frames[valid_frames["nearest_frame_label"].isin(["reject", "redirect"])]

            if len(adopted) > 0:
                frame_corr["pct_ai_sourced_after_adopt"] = round(
                    (adopted["provenance"] == "ai_sourced").sum() / len(adopted) * 100, 1)
            if len(rejected) > 0:
                frame_corr["pct_counter_position_after_reject"] = round(
                    (rejected["provenance"] == "counter_position").sum() / len(rejected) * 100, 1)

    # Key findings
    key_findings = {}
    if total_threads > 0:
        most_ai = summary_df.loc[summary_df["pct_ai_sourced"].idxmax(), "thread"]
        most_user = summary_df.loc[summary_df["pct_user_sourced"].idxmax(), "thread"]
        hard_idx = summary_df["confidence_trend_slope"].idxmax()
        soft_idx = summary_df["confidence_trend_slope"].idxmin()
        react_idx = summary_df["reactivation_count"].idxmax()

        key_findings = {
            "most_ai_influenced_thread":  str(most_ai),
            "most_user_owned_thread":     str(most_user),
            "strongest_hardening_thread": str(summary_df.loc[hard_idx, "thread"]),
            "strongest_softening_thread": str(summary_df.loc[soft_idx, "thread"]),
            "most_reactivated_thread":    str(summary_df.loc[react_idx, "thread"]),
        }

        pct_user = prov_dist.get("user_sourced", {}).get("pct", 0)
        pct_ai   = prov_dist.get("ai_sourced", {}).get("pct", 0)
        key_findings["summary"] = (
            f"{total_hypos:,} hypotheses across {total_threads} belief threads. "
            f"{pct_user}% user-sourced, {pct_ai}% AI-sourced. "
            f"Strongest hardening: {key_findings['strongest_hardening_thread']}. "
            f"Most reactivated: {key_findings['most_reactivated_thread']}."
        )

    # Cost
    p1_input = p1_output = p2_input = p2_output = 0
    if provenance_parsed:
        for v in provenance_parsed.values():
            p2_input  += v.get("input_tokens", 0)
            p2_output += v.get("output_tokens", 0)

    total_cost = round(
        (p1_input + p2_input) / 1_000_000 * PRICE_INPUT_PER_MTOK +
        (p1_output + p2_output) / 1_000_000 * PRICE_OUTPUT_PER_MTOK,
        4,
    )

    report = {
        "module":         "belief_trajectories",
        "module_version": "1.0",
        "generated_at":   pd.Timestamp.now().isoformat(),
        "model":          MODEL,
        "pass1_batch_id": pass1_batch_id or "",
        "pass2_batch_id": pass2_batch_id or "",
        "input_data": {
            "total_hypotheses":    total_hypos,
            "threads_identified":  total_threads,
            "hypotheses_assigned": total_hypos - unassigned,
            "unassigned":          unassigned,
        },
        "thread_statistics":         thread_stats,
        "provenance_distribution":   prov_dist,
        "trajectory_statistics":     traj_stats,
        "frame_adoption_correlation": frame_corr,
        "key_findings":              key_findings,
        "cost": {
            "pass1_input_tokens":  p1_input,
            "pass1_output_tokens": p1_output,
            "pass2_input_tokens":  p2_input,
            "pass2_output_tokens": p2_output,
            "total_cost_usd":      total_cost,
        },
        "figures_generated": [
            "thread_size_distribution.png", "thread_timeline.png",
            "provenance_distribution.png", "provenance_by_frame.png",
            "confidence_trajectory.png", "type_escalation.png",
            "provenance_sankey.png", "dormancy_reactivation.png",
            "ai_influence_quadrant.png", "belief_trajectory_dashboard.png",
        ],
        "data_outputs": [
            "data/processed/belief_threads.parquet",
            "data/processed/belief_thread_summaries.parquet",
            "data/processed/belief_provenance.parquet",
            "outputs/reports/belief_thread_catalog.csv",
        ],
        "warnings": warnings_list,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.4f}")
    return report


# ============================================================================
# VALIDATION
# ============================================================================
def run_validation(threads_df, summary_df, report):
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    checks = []

    def chk(label, result):
        status = "PASS" if result else "FAIL"
        checks.append((label, status))
        print(f"  [{status}] {label}")

    chk("belief_threads.parquet exists", os.path.exists(OUT_THREADS))
    chk("belief_thread_summaries.parquet exists", os.path.exists(OUT_SUMMARIES))
    chk("belief_provenance.parquet exists", os.path.exists(OUT_PROVENANCE))

    if len(threads_df) > 0:
        null_threads = threads_df["thread"].isna().sum()
        chk(f"All hypotheses have thread assignment ({null_threads} nulls)", null_threads == 0)

        null_prov = (threads_df["provenance"].astype(str).isin(["", "unknown", "nan"])).sum()
        chk(f"All hypotheses have provenance label ({null_prov} missing)", null_prov == 0)

        prov_vals = set(threads_df["provenance"].astype(str).unique()) - {"unknown", "nan", ""}
        invalid_prov = prov_vals - set(VALID_PROVENANCE)
        chk(f"Provenance values valid ({invalid_prov if invalid_prov else 'none invalid'})",
            len(invalid_prov) == 0)
    else:
        chk("All hypotheses have thread assignment (no data)", True)
        chk("All hypotheses have provenance label (no data)", True)
        chk("Provenance values valid (no data)", True)

    n_threads = len(summary_df)
    chk(f"Thread count 15-150 ({n_threads})", 15 <= n_threads <= 150 or n_threads == 0)

    if len(summary_df) > 0:
        singletons = (summary_df["hypothesis_count"] < 2).sum()
        chk(f"No singleton threads ({singletons} singletons)", singletons == 0)

        generic = summary_df["thread"].str.match(r"^thread_\d+$").sum()
        chk(f"Thread names are descriptive ({generic} generic)", generic == 0)

    if len(threads_df) > 0 and "provenance_reasoning" in threads_df.columns:
        empty_reasoning = (threads_df["provenance_reasoning"].astype(str).str.strip() == "").sum()
        chk(f"Provenance reasoning non-empty ({empty_reasoning} empty)", empty_reasoning == 0)

    chk("Report JSON exists", os.path.exists(OUT_REPORT))

    expected_figs = report.get("figures_generated", [])
    missing = [f for f in expected_figs if not os.path.exists(figpath(f))]
    chk(f"All {len(expected_figs)} figures exist ({len(missing)} missing)", len(missing) == 0)

    png_figs = [f for f in expected_figs if f.endswith(".png")]
    small_pngs = [f for f in png_figs
                  if os.path.exists(figpath(f)) and os.path.getsize(figpath(f)) < 10_000]
    chk(f"All PNGs >= 10KB ({len(small_pngs)} too small)", len(small_pngs) == 0)

    report_str = json.dumps(clean_dict(report))
    chk("No NaN/Infinity in report JSON",
        "NaN" not in report_str and "Infinity" not in report_str)

    key_findings = report.get("key_findings", {})
    chk("Key findings summary non-empty", bool(key_findings.get("summary", "").strip()))

    chk("belief_thread_catalog.csv exists", os.path.exists(OUT_CATALOG))

    passed = sum(1 for _, s in checks if s == "PASS")
    total  = len(checks)
    print("=" * 80)
    if passed == total:
        print(f"  ALL {total} CHECKS PASSED")
    else:
        print(f"  {passed}/{total} CHECKS PASSED -- review FAIL items above")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Module 3.3: Belief Trajectory Analysis via Claude Haiku Batch API"
    )
    parser.add_argument("--resume-pass1-id", default=None,
                        help="Resume Pass 1 from a previously submitted batch ID")
    parser.add_argument("--resume-pass2-id", default=None,
                        help="Resume Pass 2 from a previously submitted batch ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build Pass 1 requests and print stats without submitting")
    parser.add_argument("--threads-only", action="store_true",
                        help="Run Pass 1 only (thread clustering), skip provenance")
    parser.add_argument("--cluster-batch-size", type=int, default=CLUSTER_BATCH_SIZE,
                        help=f"Hypotheses per clustering batch (default: {CLUSTER_BATCH_SIZE})")
    args = parser.parse_args()

    warnings_list   = []
    pass1_batch_id  = None
    pass2_batch_id  = None
    provenance_parsed = {}

    # ---- Load data ----
    hypotheses, messages, conversations, frame_adoption, func_class, emot_class = load_data()

    # ---- Pass 1: Thread Clustering ----
    batches = build_pass1_inputs(hypotheses, args.cluster_batch_size)

    # For initial batch, no existing threads
    thread_list = {}
    n_pass1 = build_pass1_batch_file(batches, thread_list)

    # Estimate cost
    est_p1_cost = (n_pass1 * 3000) / 1_000_000 * PRICE_INPUT_PER_MTOK + \
                  (n_pass1 * 400) / 1_000_000 * PRICE_OUTPUT_PER_MTOK
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

    # ---- Run Pass 1 ----
    if args.resume_pass2_id:
        print("\n== Skipping Pass 1 (resuming Pass 2) ==================================")
        if os.path.exists(PASS1_RESULTS_CACHE):
            with open(PASS1_RESULTS_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            assignments = cached.get("assignments", {})
            thread_list = cached.get("thread_list", {})
            print(f"  Loaded {len(assignments):,} assignments from cache")
        else:
            print("ERROR: Cannot resume Pass 2 without Pass 1 results.")
            print(f"  Expected: {PASS1_RESULTS_CACHE}")
            sys.exit(1)
    else:
        if args.resume_pass1_id:
            pass1_batch_id = args.resume_pass1_id
            print(f"\n== Resuming Pass 1 batch: {pass1_batch_id} ==")
        else:
            pass1_batch_id = submit_batch(client, PASS1_BATCH_FILE, PASS1_BATCH_ID_FILE, "Pass 1")

        poll_batch(client, pass1_batch_id, "Pass 1")
        results_p1, errors_p1 = retrieve_results(client, pass1_batch_id, "Pass 1")
        if errors_p1:
            warnings_list.append(f"Pass 1 had {len(errors_p1)} errors")

        assignments, thread_list, p1_errors = parse_pass1(results_p1, batches)

        # Cache Pass 1 results
        with open(PASS1_RESULTS_CACHE, "w", encoding="utf-8") as f:
            json.dump({"assignments": assignments, "thread_list": thread_list}, f)
        print(f"  Cached Pass 1 results to {PASS1_RESULTS_CACHE}")

    # ---- Pass 2: Provenance Classification ----
    if not args.threads_only:
        pass2_inputs = build_pass2_inputs(hypotheses, messages, frame_adoption)
        n_pass2 = build_pass2_batch_file(pass2_inputs)

        est_p2_cost = (n_pass2 * 1850) / 1_000_000 * PRICE_INPUT_PER_MTOK + \
                      (n_pass2 * 40) / 1_000_000 * PRICE_OUTPUT_PER_MTOK
        print(f"\n  Estimated Pass 2 cost    : ${est_p2_cost:.2f}")

        if args.resume_pass2_id:
            pass2_batch_id = args.resume_pass2_id
            print(f"\n== Resuming Pass 2 batch: {pass2_batch_id} ==")
        else:
            pass2_batch_id = submit_batch(client, PASS2_BATCH_FILE, PASS2_BATCH_ID_FILE, "Pass 2")

        poll_batch(client, pass2_batch_id, "Pass 2")
        results_p2, errors_p2 = retrieve_results(client, pass2_batch_id, "Pass 2")
        if errors_p2:
            warnings_list.append(f"Pass 2 had {len(errors_p2)} errors")

        provenance_parsed, p2_errors = parse_pass2(results_p2)
    else:
        print("\n== Skipping Pass 2 (--threads-only) ===================================")

    # ---- Save & compute trajectories ----
    threads_df = save_results(hypotheses, assignments, provenance_parsed, frame_adoption, pd.DataFrame())

    # Pass 3: Trajectory computation
    summary_df = compute_trajectories(threads_df)

    # Re-save with trajectory data
    if len(summary_df) > 0:
        summary_df.to_parquet(OUT_SUMMARIES, index=False)
        print(f"  Updated: {OUT_SUMMARIES}")

        # Update catalog
        examples = {}
        for thread_name, group in threads_df.groupby("thread"):
            examples[thread_name] = str(group.iloc[0].get("resolved_text", ""))[:120]
        catalog = summary_df[["thread", "hypothesis_count", "span_days",
                               "dominant_provenance", "confidence_trend_slope",
                               "type_escalation_slope"]].copy()
        catalog["example_hypothesis"] = catalog["thread"].map(examples)
        catalog.to_csv(OUT_CATALOG, index=False, encoding="utf-8")

    # ---- Figures ----
    make_figures(threads_df, summary_df)

    # ---- Report ----
    report = generate_report(threads_df, summary_df, provenance_parsed,
                             pass1_batch_id, pass2_batch_id, warnings_list)

    # ---- Validation ----
    run_validation(threads_df, summary_df, report)

    # ---- Done ----
    print("\n" + "=" * 70)
    print("HIGHLIGHTS")
    print("=" * 70)
    kf = report.get("key_findings", {})
    if kf:
        print(f"  {kf.get('summary', 'No summary')}")
    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
