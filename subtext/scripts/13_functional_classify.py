"""
Module 3.2a: Functional Classification
Script: 13_functional_classify.py

Classifies every analysable conversation into one of 12 functional categories
using the Claude Haiku Batch API. Uses conversation summaries as input, falling
back to raw user messages for conversations where summarization failed.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/13_functional_classify.py

    # Resume a previously submitted batch:
    python scripts/13_functional_classify.py --resume-batch-id msgbatch_xxxxx

    # Skip calibration check:
    python scripts/13_functional_classify.py --skip-calibration

    # Dry run (build requests, don't submit):
    python scripts/13_functional_classify.py --dry-run
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

CONV_PATH        = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH        = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
SUMMARIES_PATH   = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
CONFIG_PATH      = os.path.join(BASE, "config", "quality_config.json")
CALIBRATION_PATH = os.path.join(BASE, "prompts", "examples", "functional_calibration_set.json")
INTERIM_DIR      = os.path.join(BASE, "data", "interim")
BATCH_FILE       = os.path.join(INTERIM_DIR, "functional_classify_batch.jsonl")
BATCH_ID_FILE    = os.path.join(INTERIM_DIR, "functional_classify_batch_id.txt")

OUT_PARQUET  = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
OUT_REPORT   = os.path.join(BASE, "outputs", "reports", "functional_classification_report.json")
FIG_DIR      = os.path.join(BASE, "outputs", "figures", "functional_classification")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Model & Pricing ---------------------------------------------------------
MODEL             = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 100
PRICE_INPUT_PER_MTOK  = 0.40   # $0.80/MTok * 0.5 batch discount
PRICE_OUTPUT_PER_MTOK = 2.00   # $4.00/MTok * 0.5 batch discount

# -- Valid function labels ----------------------------------------------------
VALID_FUNCTIONS = [
    "interpersonal_analysis",
    "emotional_processing",
    "creative_expression",
    "career_strategy",
    "self_modeling",
    "practical",
    "learning",
    "problem_solving",
    "coding",
    "social_rehearsal",
    "work_professional",
    "planning",
]

# -- Style constants (match existing scripts) ---------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"

FUNC_COLORS = {
    "interpersonal_analysis": "#4E79A7",
    "emotional_processing":   "#F28E2B",
    "creative_expression":    "#E15759",
    "career_strategy":        "#76B7B2",
    "self_modeling":          "#59A14F",
    "practical":              "#EDC948",
    "learning":               "#B07AA1",
    "problem_solving":        "#FF9DA7",
    "coding":                 "#9C755F",
    "social_rehearsal":       "#BAB0AC",
    "work_professional":      "#86BCB6",
    "planning":               "#F1CE63",
}

DPI = 150

# -- System Prompt ------------------------------------------------------------
SYSTEM_PROMPT = "\n".join([
    "You are a research assistant classifying conversations by their primary function.",
    "",
    "Given a summary of a conversation between a user and an AI assistant, classify the",
    "conversation into exactly ONE of these categories:",
    "",
    "- interpersonal_analysis: The user is analyzing relationships, decoding social dynamics,",
    "  mapping power structures, or processing interactions with specific people. The mode is",
    "  analytical rather than emotional.",
    "- emotional_processing: The user is processing feelings, grief, anxiety, burnout, or",
    "  trauma. The mode is therapeutic rather than analytical.",
    "- creative_expression: The user is writing poetry, analyzing art, building personal",
    "  mythology, or doing expressive journaling.",
    "- career_strategy: The user is preparing for jobs, analyzing roles, building pitches,",
    "  planning for conferences, or positioning themselves professionally.",
    "- self_modeling: The user is asking the AI to analyze, profile, or mirror their own",
    "  patterns, identity, or psychology.",
    "- practical: Quick utilitarian questions -- health, logistics, products, how-to, factual lookups.",
    "- learning: The user is seeking to understand a concept, domain, or skill.",
    "- problem_solving: The user is working through a specific technical or logical problem.",
    "- coding: Writing, debugging, reviewing, or generating code is the primary activity.",
    "- social_rehearsal: The user is drafting messages, emails, or preparing for specific conversations.",
    "- work_professional: Work-specific tasks -- security policy, governance documents, vendor",
    "  review, tool evaluation -- that are not about career strategy.",
    "- planning: Organizing actions, trips, or projects that are not career-specific.",
    "",
    "Choose the BEST single category. If a conversation spans multiple functions, choose the",
    "one that best describes the PRIMARY purpose.",
    "",
    "Key distinctions:",
    "- interpersonal_analysis vs emotional_processing: analytical (decoding, mapping, tracking)",
    "  vs feeling (grief, venting, therapeutic)",
    "- career_strategy vs work_professional: user's trajectory vs doing the current job",
    "- self_modeling vs interpersonal_analysis: AI analyzing the user vs AI helping analyze",
    "  other people",
    "",
    'Respond with ONLY a JSON object in this exact format:',
    '{"function": "<category>", "confidence": <0.0-1.0>, "secondary": ["<category>", ...]}',
    "",
    '- "function": The primary function label (one of the 12 categories above)',
    '- "confidence": Your confidence in the primary classification (0.0 to 1.0)',
    '- "secondary": A list of 0-2 additional function labels if the conversation clearly spans',
    "  multiple functions. Empty list [] if single-function.",
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
def load_data():
    print("\n-- Step 0: Load data -------------------------------------------------")

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
    msgs = messages[messages["conversation_id"].isin(conv["conversation_id"])].copy()

    print(f"  Analysable conversations : {len(conv):,}")
    print(f"  Messages in scope        : {len(msgs):,}")
    print(f"  Summaries available      : {len(summaries_raw):,}")

    return conv, msgs, summaries_raw, config


def build_classification_inputs(conv, msgs, summaries_raw):
    """For each analysable conversation, determine input for the classifier."""
    print("\n-- Building classification inputs ------------------------------------")
    enc = tiktoken.get_encoding("cl100k_base")

    sum_map = {}
    for _, row in summaries_raw.iterrows():
        cid = row.get("conversation_id")
        s   = row.get("summary", "")
        if pd.notna(cid) and pd.notna(s):
            sum_map[str(cid)] = str(s)

    inputs     = {}
    n_summary  = 0
    n_fallback = 0

    for _, c in tqdm(conv.iterrows(), total=len(conv), desc="Building inputs"):
        cid     = c["conversation_id"]
        title   = c.get("title", "") or ""
        summary = sum_map.get(str(cid), "")

        if summary and summary.strip() and "[SUMMARIZATION FAILED]" not in summary:
            inputs[cid] = {"text": summary.strip(), "title": title, "source": "summary"}
            n_summary += 1
        else:
            user_msgs = msgs[
                (msgs["conversation_id"] == cid) & (msgs["role"] == "user")
            ].sort_values("msg_index")
            texts    = [str(t) for t in user_msgs["text"].dropna() if str(t).strip()]
            combined = " ".join(texts)
            try:
                tokens = enc.encode(combined)
                if len(tokens) > 2000:
                    combined = enc.decode(tokens[:2000])
            except Exception:
                combined = combined[:8000]
            inputs[cid] = {"text": combined, "title": title, "source": "fallback_messages"}
            n_fallback += 1

    print(f"  Using summaries          : {n_summary:,}")
    print(f"  Using fallback messages  : {n_fallback:,}")
    return inputs


# -- Step 1: Calibration ------------------------------------------------------
def _format_user_message(inp):
    title  = (inp.get("title", "") or "").strip()
    text   = (inp.get("text", "")  or "").strip()
    source = inp.get("source", "summary")
    if source == "summary":
        return f"Classify this conversation:\n\nTitle: {title}\nSummary: {text}"
    else:
        return (
            f"Classify this conversation based on the user's messages:\n\n"
            f"Title: {title}\nUser messages:\n{text}"
        )


def run_calibration(client, inputs, skip_calibration):
    print("\n-- Step 1: Calibration -----------------------------------------------")
    if skip_calibration:
        print("  Skipping calibration (--skip-calibration)")
        return {"note": "skipped via --skip-calibration"}

    if not os.path.exists(CALIBRATION_PATH):
        print("  WARNING: Calibration set not found -- skipped")
        print(f"    Expected: {CALIBRATION_PATH}")
        return {"note": "calibration set not found -- skipped"}

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        calibration_set = json.load(f)
    print(f"  Calibration set loaded   : {len(calibration_set)} items")

    manual_labels = []
    claude_labels = []
    disagreements = []

    for item in tqdm(calibration_set, desc="Calibrating"):
        cid    = item["conversation_id"]
        manual = item["function"]
        inp    = inputs.get(cid)
        if not inp:
            continue
        try:
            response     = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _format_user_message(inp)}],
            )
            parsed       = json.loads(response.content[0].text.strip())
            claude_label = parsed.get("function", "unknown")
        except Exception:
            claude_label = "unknown"
        manual_labels.append(manual)
        claude_labels.append(claude_label)
        if manual != claude_label:
            disagreements.append({
                "conversation_id": cid,
                "manual":          manual,
                "claude":          claude_label,
                "notes":           item.get("notes", ""),
            })
        time.sleep(0.5)

    if not manual_labels:
        return {"note": "no matching conversations in calibration set"}

    kappa = None
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(manual_labels, claude_labels)
    except Exception as e:
        print(f"  WARNING: Could not compute Cohen's kappa: {e}")

    kappa_ok = kappa is not None and kappa >= 0.75
    if kappa is not None:
        print(f"  Cohen's kappa            : {kappa:.3f}")
    else:
        print("  Cohen's kappa: N/A")
    print(f"  Disagreements            : {len(disagreements)} / {len(manual_labels)}")
    if not kappa_ok:
        print("  WARNING: kappa < 0.75 -- review disagreements before relying on results")

    result = {
        "cohens_kappa":        round(kappa, 3) if kappa is not None else None,
        "n_calibration_items": len(manual_labels),
        "disagreements":       len(disagreements),
        "kappa_target_met":    kappa_ok,
        "disagreement_details": disagreements[:10],
    }

    cal_report_path = os.path.join(BASE, "outputs", "reports", "functional_calibration.json")
    with open(cal_report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Calibration report saved : {cal_report_path}")
    return result


# -- Step 2: Build batch requests ---------------------------------------------
def build_batch_requests(inputs):
    print("\n-- Step 2: Build batch requests --------------------------------------")
    with open(BATCH_FILE, "w", encoding="utf-8") as f:
        for cid, inp in inputs.items():
            request = {
                "custom_id": cid,
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


# -- Step 3: Submit batch -----------------------------------------------------
def submit_batch(client):
    print("\n-- Step 3: Submit batch ----------------------------------------------")
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


# -- Step 4: Poll -------------------------------------------------------------
def poll_batch(client, batch_id):
    print(f"\n-- Step 4: Poll batch {batch_id} --")
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


# -- Step 5: Retrieve and parse results ---------------------------------------
def retrieve_results(client, batch_id):
    print(f"\n-- Step 5: Retrieve results ------------------------------------------")
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
                "conversation_id": cid,
                "error_type":      result.result.type,
                "error":           str(result.result.error)
                                   if hasattr(result.result, "error") else "unknown",
            })

    print(f"  Batch succeeded          : {len(results):,}")
    print(f"  Batch errored            : {len(errors):,}")
    return results, errors


def retry_errors(client, errors, inputs):
    if not errors:
        return {}
    print(f"\n  Retrying {len(errors)} failure(s) via standard API...")
    retried = {}
    for err in tqdm(errors, desc="Retrying"):
        cid = err["conversation_id"]
        inp = inputs.get(cid)
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
            retried[cid] = {
                "raw":           raw_text.strip(),
                "input_tokens":  response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Retry failed for {cid}: {e}")
        time.sleep(1)
    print(f"  Retries succeeded        : {len(retried):,}")
    return retried


def parse_classification(raw_text):
    """Parse JSON response. Returns (function, confidence, secondary_json, error_msg)."""
    try:
        # Strip markdown code fences if model wrapped the JSON
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        func   = parsed.get("function", "").strip()
        conf   = float(parsed.get("confidence", 0.0))
        sec    = parsed.get("secondary", [])

        if func not in VALID_FUNCTIONS:
            return None, 0.0, "[]", f"invalid function label: {func!r}"

        if not isinstance(sec, list):
            sec = []
        sec = [s for s in sec if isinstance(s, str) and s in VALID_FUNCTIONS and s != func]
        sec = sec[:2]

        conf = max(0.0, min(1.0, conf))
        return func, float(conf), json.dumps(sec), None
    except Exception as e:
        return None, 0.0, "[]", f"parse error: {e}"


def build_output_rows(results, inputs):
    print("\n  Parsing classification responses...")
    rows         = []
    parse_errors = 0

    for cid, res in tqdm(results.items(), desc="Parsing"):
        inp                       = inputs.get(cid, {})
        func, conf, sec_json, err = parse_classification(res.get("raw", ""))
        if err:
            parse_errors += 1
            func, conf, sec_json = "unknown", 0.0, "[]"
        rows.append({
            "conversation_id":      cid,
            "function_primary":     func,
            "function_confidence":  conf,
            "function_secondary":   sec_json,
            "classification_input": inp.get("source", "summary"),
            "input_tokens":         res.get("input_tokens", 0),
            "output_tokens":        res.get("output_tokens", 0),
        })

    print(f"  Parse errors             : {parse_errors:,}")
    return rows, parse_errors


def fill_placeholders(rows_dict, inputs):
    filled = 0
    for cid in inputs:
        if cid not in rows_dict:
            rows_dict[cid] = {
                "conversation_id":      cid,
                "function_primary":     "unknown",
                "function_confidence":  0.0,
                "function_secondary":   "[]",
                "classification_input": inputs[cid].get("source", "summary"),
                "input_tokens":         0,
                "output_tokens":        0,
            }
            filled += 1
    if filled:
        print(f"  Filled {filled} placeholder(s) for permanently failed conversations")
    return filled


# -- Step 6: Save parquet -----------------------------------------------------
def save_parquet(rows):
    print("\n-- Step 6: Save results ----------------------------------------------")
    df = pd.DataFrame(rows)

    df["function_primary"]     = df["function_primary"].astype("category")
    df["classification_input"] = df["classification_input"].astype("category")
    df["function_confidence"]  = df["function_confidence"].astype("float32")
    df["input_tokens"]         = df["input_tokens"].astype("int32")
    df["output_tokens"]        = df["output_tokens"].astype("int32")
    # function_secondary is a JSON-serialized string, e.g. '["emotional_processing"]'
    # Downstream consumers should parse with json.loads()

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Rows:  {len(df):,}")
    return df


# -- Step 7: Figures ----------------------------------------------------------
def make_figures(df, conv, config):
    print("\n-- Step 7: Generating figures ----------------------------------------")

    merged = df.merge(
        conv[[
            "conversation_id", "conversation_type", "model_era",
            "year_month", "time_of_day", "turns", "duration_minutes",
            "user_token_ratio", "msg_count",
        ]],
        on="conversation_id", how="left",
    )
    merged = merged[merged["function_primary"].astype(str) != "unknown"].copy()

    _fig1_distribution(df)
    _fig2_by_conversation_type(merged)
    _fig3_by_model_era(merged)
    _fig4_over_time(merged, config)
    _fig5_confidence(df)
    _fig6_by_time_of_day(merged)
    _fig7_depth(merged)
    _fig8_dashboard(merged)
    print("  All figures saved.")


def _stacked_pct_bar(ax, df, group_col, title, xlabel, row_order=None):
    ct = pd.crosstab(df[group_col], df["function_primary"].astype(str))
    ct = ct.reindex(columns=[f for f in VALID_FUNCTIONS if f in ct.columns])
    if row_order:
        ct = ct.reindex([r for r in row_order if r in ct.index])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(ct_pct))
    x = np.arange(len(ct_pct))
    for func in ct_pct.columns:
        vals = ct_pct[func].values
        ax.bar(x, vals, bottom=bottom,
               color=FUNC_COLORS.get(func, "#888888"), label=func, width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ct_pct.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)


def _fig1_distribution(df):
    valid  = df[df["function_primary"].astype(str) != "unknown"]
    counts = valid["function_primary"].astype(str).value_counts()
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = [FUNC_COLORS.get(f, COLOR_PRIMARY) for f in counts.index]
    bars    = ax.barh(counts.index, counts.values, color=colors)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Number of Conversations")
    ax.set_title("Functional Classification Distribution")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.25)
    plt.tight_layout()
    plt.savefig(figpath("function_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: function_distribution.png")


def _fig2_by_conversation_type(merged):
    fig, ax = plt.subplots(figsize=(12, 6))
    _stacked_pct_bar(ax, merged, "conversation_type",
                     "Function Distribution by Conversation Type", "Conversation Type")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FUNC_COLORS.get(f, "#888"))
               for f in VALID_FUNCTIONS]
    ax.legend(handles, VALID_FUNCTIONS, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figpath("function_by_conversation_type.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: function_by_conversation_type.png")


def _fig3_by_model_era(merged):
    fig, ax = plt.subplots(figsize=(12, 6))
    _stacked_pct_bar(ax, merged, "model_era",
                     "Function Distribution by Model Era", "Model Era")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FUNC_COLORS.get(f, "#888"))
               for f in VALID_FUNCTIONS]
    ax.legend(handles, VALID_FUNCTIONS, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figpath("function_by_model_era.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: function_by_model_era.png")


def _fig4_over_time(merged, config):
    if "year_month" not in merged.columns or merged["year_month"].isna().all():
        print("  Skipping function_over_time.html (no year_month data)")
        return

    ct     = pd.crosstab(merged["year_month"].astype(str),
                         merged["function_primary"].astype(str))
    ct     = ct.reindex(columns=[f for f in VALID_FUNCTIONS if f in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.sort_index()

    fig = go.Figure()
    for func in ct_pct.columns:
        fig.add_trace(go.Scatter(
            x=ct_pct.index.tolist(),
            y=ct_pct[func].round(1).tolist(),
            mode="lines",
            stackgroup="one",
            name=func,
            line_color=FUNC_COLORS.get(func, "#888888"),
            hovertemplate="%{x}<br>" + func + ": %{y:.1f}%<extra></extra>",
        ))

    era_boundaries = config.get("model_era_boundaries", {})
    for era_name, era_date in era_boundaries.items():
        era_ym = era_date[:7]
        if ct_pct.index.min() <= era_ym <= ct_pct.index.max():
            fig.add_vline(
                x=era_ym,
                line_dash="dash",
                line_color="rgba(100,100,100,0.5)",
                annotation_text=era_name.replace("_", " "),
                annotation_position="top left",
                annotation_font_size=9,
            )

    fig.update_layout(
        title="Function Distribution Over Time",
        xaxis_title="Month",
        yaxis_title="Proportion (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        legend_title="Function",
        height=500,
    )
    fig.write_html(figpath("function_over_time.html"), include_plotlyjs=True)
    print(f"  Saved: function_over_time.html")


def _fig5_confidence(df):
    conf = df["function_confidence"].astype(float).dropna()
    bins = np.linspace(0, 1, 26)

    fig, ax = plt.subplots(figsize=(9, 5))
    n, _, patches = ax.hist(conf, bins=bins, edgecolor="white")

    for i, patch in enumerate(patches):
        mid = (bins[i] + bins[i + 1]) / 2
        if mid >= 0.8:
            patch.set_facecolor(COLOR_PRIMARY)
        elif mid >= 0.5:
            patch.set_facecolor("#EDC948")
        else:
            patch.set_facecolor(COLOR_ACCENT)

    ax.axvline(conf.mean(),   color="black", linestyle="--", linewidth=1.2,
               label=f"Mean: {conf.mean():.2f}")
    ax.axvline(conf.median(), color="black", linestyle=":",  linewidth=1.2,
               label=f"Median: {conf.median():.2f}")

    pct_high   = (conf >= 0.8).mean() * 100
    pct_medium = ((conf >= 0.5) & (conf < 0.8)).mean() * 100
    pct_low    = (conf < 0.5).mean() * 100
    ax.text(
        0.02, 0.95,
        f"High (>=0.8): {pct_high:.1f}%\nMedium (0.5-0.8): {pct_medium:.1f}%\nLow (<0.5): {pct_low:.1f}%",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Classification Confidence Distribution")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath("confidence_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: confidence_distribution.png")


def _fig6_by_time_of_day(merged):
    order   = ["morning", "afternoon", "evening", "night"]
    present = [o for o in order if o in merged["time_of_day"].values]
    subset  = merged[merged["time_of_day"].isin(present)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    _stacked_pct_bar(ax, subset, "time_of_day",
                     "Function Distribution by Time of Day", "Time of Day",
                     row_order=present)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FUNC_COLORS.get(f, "#888"))
               for f in VALID_FUNCTIONS]
    ax.legend(handles, VALID_FUNCTIONS, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figpath("function_by_time_of_day.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: function_by_time_of_day.png")


def _fig7_depth(merged):
    metrics = [
        ("turns",            "Mean Turns"),
        ("duration_minutes", "Mean Duration (min)"),
        ("user_token_ratio", "Mean Token Ratio"),
        ("msg_count",        "Mean Message Count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (col, label) in enumerate(metrics):
        ax = axes[i]
        if col not in merged.columns:
            ax.set_visible(False)
            continue

        groups       = merged.groupby("function_primary")[col]
        group_labels = [f for f in VALID_FUNCTIONS if f in groups.groups]
        means  = [groups.get_group(f).mean() for f in group_labels]
        sems   = [groups.get_group(f).sem()  for f in group_labels]
        colors = [FUNC_COLORS.get(f, COLOR_PRIMARY) for f in group_labels]

        ax.bar(range(len(group_labels)), means, yerr=sems, color=colors,
               capsize=3, error_kw={"linewidth": 0.8})

        group_data   = [
            groups.get_group(f).dropna().values
            for f in group_labels if f in groups.groups
        ]
        group_data   = [g for g in group_data if len(g) > 0]
        title_suffix = ""
        if len(group_data) >= 2:
            try:
                _, p = stats.kruskal(*group_data)
                if p < 0.05:
                    title_suffix = " *"
            except Exception:
                pass

        ax.set_title(f"{label} by Function{title_suffix}", fontsize=10)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=9)

    plt.suptitle(
        "Function Predicts Conversation Depth\n(* = Kruskal-Wallis p < 0.05)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(figpath("function_vs_depth.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: function_vs_depth.png")


def _fig8_dashboard(merged):
    counts = merged["function_primary"].astype(str).value_counts()
    total  = counts.sum()
    if total == 0:
        print("  Skipping cognitive_signature_dashboard.png (no data)")
        return

    dominant_func = counts.index[0]
    dominant_pct  = counts.iloc[0] / total * 100
    top3          = counts.head(3)
    probs         = counts / total
    entropy       = float(-(probs * np.log2(probs + 1e-12)).sum())
    top2_pct      = round(
        (top3.iloc[0] + (top3.iloc[1] if len(top3) > 1 else 0)) / total * 100, 1
    )
    insight = (
        f"{top3.index[0] if len(top3) > 0 else 'unknown'} and "
        f"{top3.index[1] if len(top3) > 1 else 'others'} account for "
        f"{top2_pct}% of conversations"
    )

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.text(0.5, 0.6, dominant_func.replace("_", "\n"),
              ha="center", va="center", fontsize=18, fontweight="bold",
              color=FUNC_COLORS.get(dominant_func, COLOR_PRIMARY),
              transform=ax_a.transAxes)
    ax_a.text(0.5, 0.2, f"{dominant_pct:.1f}% of conversations",
              ha="center", va="center", fontsize=12, transform=ax_a.transAxes)
    ax_a.set_title("Dominant Function", fontsize=11)

    ax_b = fig.add_subplot(gs[0, 1])
    pie_labels = top3.index.tolist()
    pie_colors = [FUNC_COLORS.get(f, "#888") for f in pie_labels]
    ax_b.pie(top3.values, labels=pie_labels, colors=pie_colors,
             autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax_b.set_title("Top 3 Functions", fontsize=11)

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.axis("off")
    ax_c.text(0.5, 0.6, f"{entropy:.2f}",
              ha="center", va="center", fontsize=24, fontweight="bold",
              color=COLOR_PRIMARY, transform=ax_c.transAxes)
    ax_c.text(0.5, 0.25,
              f"Shannon Entropy\n(bits, max = {np.log2(len(VALID_FUNCTIONS)):.2f})",
              ha="center", va="center", fontsize=10, color=COLOR_SECONDARY,
              transform=ax_c.transAxes)
    ax_c.set_title("Function Diversity", fontsize=11)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    ax_d.text(0.5, 0.5, insight,
              ha="center", va="center", fontsize=11, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_d.set_title("Key Insight", fontsize=11)

    plt.suptitle("Cognitive Signature: Functional Profile", fontsize=14, fontweight="bold")
    plt.savefig(figpath("cognitive_signature_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cognitive_signature_dashboard.png")


# -- Step 8: Report -----------------------------------------------------------
def generate_report(df, conv, batch_id, calibration_result, parse_errors,
                    n_fallback, warnings_list):
    print("\n-- Step 8: Generate report -------------------------------------------")

    merged = df.merge(
        conv[[
            "conversation_id", "conversation_type", "model_era",
            "year_month", "time_of_day", "turns", "duration_minutes",
            "user_token_ratio", "msg_count",
        ]],
        on="conversation_id", how="left",
    )

    valid  = merged[merged["function_primary"].astype(str) != "unknown"]
    counts = valid["function_primary"].astype(str).value_counts()
    total  = len(valid)

    distribution = {
        func: {
            "count": int(counts.get(func, 0)),
            "pct":   round(counts.get(func, 0) / max(total, 1) * 100, 1),
        }
        for func in VALID_FUNCTIONS
    }

    conf = df["function_confidence"].astype(float)
    conf_stats = {
        "mean":       round(float(conf.mean()), 3),
        "median":     round(float(conf.median()), 3),
        "std":        round(float(conf.std()), 3),
        "pct_high":   round(float((conf >= 0.8).mean() * 100), 1),
        "pct_medium": round(float(((conf >= 0.5) & (conf < 0.8)).mean() * 100), 1),
        "pct_low":    round(float((conf < 0.5).mean() * 100), 1),
    }

    def cross_tab(group_col):
        if group_col not in merged.columns:
            return {}
        ct = pd.crosstab(merged[group_col].astype(str),
                         merged["function_primary"].astype(str))
        return {str(g): {str(f): int(v) for f, v in row.items()}
                for g, row in ct.iterrows()}

    stat_tests = {}
    for metric in ["turns", "duration_minutes", "user_token_ratio", "msg_count"]:
        if metric not in merged.columns:
            continue
        group_data = [
            merged[merged["function_primary"].astype(str) == func][metric].dropna().values
            for func in VALID_FUNCTIONS
        ]
        group_data = [g for g in group_data if len(g) > 0]
        if len(group_data) >= 2:
            try:
                H, p = stats.kruskal(*group_data)
                stat_tests[f"function_vs_{metric}"] = {
                    "H":           round(float(H), 2),
                    "p":           round(float(p), 6),
                    "significant": bool(p < 0.05),
                }
            except Exception:
                pass

    total_input  = int(df["input_tokens"].astype(int).sum())
    total_output = int(df["output_tokens"].astype(int).sum())
    total_cost   = round(
        total_input  / 1_000_000 * PRICE_INPUT_PER_MTOK +
        total_output / 1_000_000 * PRICE_OUTPUT_PER_MTOK,
        4,
    )

    top3     = counts.head(3).index.tolist() if len(counts) >= 3 else counts.index.tolist()
    probs    = counts / max(total, 1)
    entropy  = float(-(probs * np.log2(probs + 1e-12)).sum())
    top2_pct = round(
        (counts.iloc[0] + (counts.iloc[1] if len(counts) > 1 else 0))
        / max(total, 1) * 100, 1
    )
    cog_summary = (
        f"{top3[0] if top3 else 'unknown'} and "
        f"{top3[1] if len(top3) > 1 else 'others'} account for "
        f"{top2_pct}% of conversations; "
        f"function diversity (Shannon entropy) = {entropy:.2f} bits."
    )

    report = {
        "module":         "functional_classification",
        "module_version": "1.0",
        "generated_at":   pd.Timestamp.now().isoformat(),
        "model":          MODEL,
        "batch_id":       batch_id,
        "input_data": {
            "conversations_classified": len(df),
            "input_source_summary":     int(
                (df["classification_input"].astype(str) == "summary").sum()
            ),
            "input_source_fallback":    int(
                (df["classification_input"].astype(str) == "fallback_messages").sum()
            ),
            "classification_errors":    parse_errors,
        },
        "distribution":     distribution,
        "confidence_stats": conf_stats,
        "cross_tabulations": {
            "by_conversation_type": cross_tab("conversation_type"),
            "by_model_era":         cross_tab("model_era"),
            "by_time_of_day":       cross_tab("time_of_day"),
        },
        "statistical_tests": stat_tests,
        "calibration":       calibration_result,
        "cost": {
            "input_tokens":   total_input,
            "output_tokens":  total_output,
            "total_cost_usd": total_cost,
            "pricing_note":   (
                "Haiku 4.5 Batch API: $0.40/MTok input, "
                "$2.00/MTok output (50% batch discount)"
            ),
        },
        "cognitive_signature_fragment": {
            "dominant_function":          top3[0] if top3 else "unknown",
            "dominant_pct":               round(
                float(counts.iloc[0] / max(total, 1) * 100), 1
            ) if len(counts) else 0.0,
            "function_diversity_entropy": round(entropy, 3),
            "top_3":                      top3,
            "summary":                    cog_summary,
        },
        "figures_generated": [
            "function_distribution.png",
            "function_by_conversation_type.png",
            "function_by_model_era.png",
            "function_over_time.html",
            "confidence_distribution.png",
            "function_by_time_of_day.png",
            "function_vs_depth.png",
            "cognitive_signature_dashboard.png",
        ],
        "data_outputs": ["data/processed/functional_classifications.parquet"],
        "warnings":       warnings_list,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.4f}")
    return report


# -- Step 9: Validation -------------------------------------------------------
def run_validation(df, conv, report):
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    checks = []

    def chk(label, result):
        status = "PASS" if result else "FAIL"
        checks.append((label, status))
        print(f"  [{status}] {label}")

    chk("functional_classifications.parquet exists", os.path.exists(OUT_PARQUET))

    required_cols = {
        "conversation_id", "function_primary", "function_confidence",
        "function_secondary", "classification_input", "input_tokens", "output_tokens",
    }
    chk("Required columns present", required_cols.issubset(set(df.columns)))

    expected = len(conv)
    pct_diff = abs(len(df) - expected) / max(expected, 1) * 100
    chk(f"Row count within 1% of analysable ({len(df):,} / {expected:,})", pct_diff <= 1)

    valid_ids = set(conv["conversation_id"])
    orphans   = set(df["conversation_id"]) - valid_ids
    chk("No orphan conversation_ids", len(orphans) == 0)

    chk("No duplicate conversation_ids", df["conversation_id"].duplicated().sum() == 0)

    all_vals      = set(df["function_primary"].astype(str).unique())
    invalid_funcs = all_vals - set(VALID_FUNCTIONS) - {"unknown"}
    chk("All function_primary values valid", len(invalid_funcs) == 0)

    conf = df["function_confidence"].astype(float)
    chk("Confidence scores in [0, 1]",
        bool((conf >= 0.0).all() and (conf <= 1.0).all()))

    err_pct = (df["function_primary"].astype(str) == "unknown").mean() * 100
    chk(f"Classification errors < 1% ({err_pct:.2f}%)", err_pct < 1.0)

    req_keys = {
        "module", "distribution", "confidence_stats",
        "cost", "cognitive_signature_fragment", "figures_generated",
    }
    chk("Report JSON has required keys",
        os.path.exists(OUT_REPORT) and req_keys.issubset(set(report.keys())))

    expected_figs = [
        "function_distribution.png",
        "function_by_conversation_type.png",
        "function_by_model_era.png",
        "function_over_time.html",
        "confidence_distribution.png",
        "function_by_time_of_day.png",
        "function_vs_depth.png",
        "cognitive_signature_dashboard.png",
    ]
    missing_figs = [f for f in expected_figs if not os.path.exists(figpath(f))]
    chk(f"All 8 figures exist ({len(missing_figs)} missing)", len(missing_figs) == 0)

    png_figs   = [f for f in expected_figs if f.endswith(".png")]
    small_pngs = [
        f for f in png_figs
        if os.path.exists(figpath(f)) and os.path.getsize(figpath(f)) < 10_000
    ]
    chk(f"All PNGs >= 10KB ({len(small_pngs)} too small)", len(small_pngs) == 0)

    html_path = figpath("function_over_time.html")
    html_ok   = False
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as fh:
            html_ok = "plotly" in fh.read(5000).lower()
    chk("HTML figure contains plotlyjs", html_ok)

    cog    = report.get("cognitive_signature_fragment", {})
    cog_ok = bool(cog.get("summary", "").strip())
    chk("Cognitive signature summary non-empty", cog_ok)

    passed = sum(1 for _, s in checks if s == "PASS")
    total  = len(checks)
    print("=" * 80)
    if passed == total:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {passed}/{total} CHECKS PASSED -- review FAIL items above")
    print("=" * 80)


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Module 3.2a: Functional Classification via Claude Haiku Batch API"
    )
    parser.add_argument("--resume-batch-id", default=None,
                        help="Resume from a previously submitted batch ID")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip calibration protocol")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build requests and print stats without submitting")
    args = parser.parse_args()

    warnings_list = []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print(f"API key found: {api_key[:12]}...{api_key[-4:]}")

    import anthropic
    client = anthropic.Anthropic()

    conv, msgs, summaries_raw, config = load_data()
    inputs = build_classification_inputs(conv, msgs, summaries_raw)

    n_fallback = sum(1 for v in inputs.values() if v["source"] == "fallback_messages")
    if n_fallback > 0:
        warnings_list.append(
            f"{n_fallback} conversation(s) used fallback messages (no summary available)"
        )

    calibration_result = run_calibration(client, inputs, args.skip_calibration)
    n_requests = build_batch_requests(inputs)

    if args.dry_run:
        print(f"\n-- DRY RUN: {n_requests:,} requests built. Not submitting. --")
        return

    if args.resume_batch_id:
        batch_id = args.resume_batch_id
        print(f"\n-- Resuming batch: {batch_id} --")
    else:
        batch_id = submit_batch(client)

    poll_batch(client, batch_id)

    results, errors = retrieve_results(client, batch_id)
    retried          = retry_errors(client, errors, inputs)
    results.update(retried)

    rows, parse_errors = build_output_rows(results, inputs)
    rows_dict = {r["conversation_id"]: r for r in rows}
    fill_placeholders(rows_dict, inputs)

    if parse_errors > 0:
        warnings_list.append(f"{parse_errors} parse error(s) -- set to 'unknown'")

    df = save_parquet(list(rows_dict.values()))

    try:
        make_figures(df, conv, config)
    except Exception as e:
        msg = f"Figure generation error: {e}"
        print(f"  WARNING: {msg}")
        warnings_list.append(msg)

    report = generate_report(
        df, conv, batch_id, calibration_result,
        parse_errors, n_fallback, warnings_list,
    )

    run_validation(df, conv, report)
    print("\nDone!")


if __name__ == "__main__":
    main()
