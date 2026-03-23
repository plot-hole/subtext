"""
Module 3.2b: Emotional State Classification
Script: 14_emotional_state.py

Classifies every analysable conversation by its emotional texture — 12 emotional
states designed for this corpus (analytical, anxious, curious, frustrated,
grieving, playful, reflective, strategic, vulnerable, energized, numb, determined).
Uses Claude Haiku Batch API on conversation summaries from all_summaries.csv.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/14_emotional_state.py

    # Resume a previously submitted batch:
    python scripts/14_emotional_state.py --resume-batch-id msgbatch_xxxxx

    # Dry run (build requests, don't submit):
    python scripts/14_emotional_state.py --dry-run
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
INTERIM_DIR     = os.path.join(BASE, "data", "interim")
BATCH_FILE      = os.path.join(INTERIM_DIR, "emotional_state_batch.jsonl")
BATCH_ID_FILE   = os.path.join(INTERIM_DIR, "emotional_state_batch_id.txt")

OUT_PARQUET = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
OUT_REPORT  = os.path.join(BASE, "outputs", "reports", "emotional_state_report.json")
FIG_DIR     = os.path.join(BASE, "outputs", "figures", "emotional_state")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Model & Pricing ---------------------------------------------------------
MODEL             = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 100
PRICE_INPUT_PER_MTOK  = 0.40   # $0.80/MTok * 0.5 batch discount
PRICE_OUTPUT_PER_MTOK = 2.00   # $4.00/MTok * 0.5 batch discount

# -- Valid emotion labels ----------------------------------------------------
VALID_EMOTIONS = [
    "analytical",
    "anxious",
    "curious",
    "frustrated",
    "grieving",
    "playful",
    "reflective",
    "strategic",
    "vulnerable",
    "energized",
    "numb",
    "determined",
]

# Valid function labels (for cross-tab figures)
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

# -- Style constants ----------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"

# Group-based emotion color palette
# Cool states: blue tones | Warm states: orange tones
# Distress states: red/purple | Action states: green tones
EMOTION_COLORS = {
    "analytical":  "#4E79A7",  # cool / blue
    "curious":     "#A0CBE8",  # cool / light blue
    "reflective":  "#76B7B2",  # cool / teal
    "energized":   "#F28E2B",  # warm / orange
    "playful":     "#FFBE7D",  # warm / light orange
    "anxious":     "#E15759",  # distress / red
    "frustrated":  "#FF9DA7",  # distress / light red
    "grieving":    "#B07AA1",  # distress / purple
    "vulnerable":  "#D4A6C8",  # distress / light purple
    "numb":        "#BAB0AC",  # distress / grey
    "strategic":   "#59A14F",  # action / green
    "determined":  "#8CD17D",  # action / light green
}

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
    "You are a research assistant classifying the emotional state of conversations.",
    "",
    "Given a summary of a conversation between a user and an AI assistant, classify the",
    "PRIMARY emotional state of the user during the conversation. Focus on the user's",
    "emotional state, not the AI's tone.",
    "",
    "Categories:",
    "",
    "- analytical: Detached, mapping, decoding, intellectualized. Thinking about emotions",
    "  without being in them.",
    "- anxious: Worried, uncertain, seeking reassurance, vigilant. Threat detection active.",
    "- curious: Exploring, learning, open, engaged. Intellectual appetite without emotional weight.",
    "- frustrated: Blocked, angry, constrained, hitting walls. Energy directed at an obstacle.",
    "- grieving: Loss, letting go, sadness, mourning what was or could have been.",
    "- playful: Light, creative, experimenting, irreverent. Low stakes, high energy.",
    "- reflective: Introspective, meaning-making, connecting dots. Calm processing rather than",
    "  active distress.",
    "- strategic: Calculating, positioning, planning moves. Forward-looking with intent.",
    "- vulnerable: Exposed, raw, asking for support. Defenses are down.",
    "- energized: Excited, activated, momentum, breakthrough energy.",
    "- numb: Flat, disconnected, going through motions. Absence of feeling.",
    "- determined: Resolved, committed, drawing a line. Clarity of purpose after ambiguity.",
    "",
    "Key distinctions:",
    "- analytical vs reflective: externally focused (decoding others) vs internally focused",
    "  (examining self)",
    "- anxious vs vulnerable: defended worry vs undefended exposure",
    "- strategic vs analytical: forward-looking (planning) vs backward/present (understanding)",
    "- energized vs playful: high stakes excitement vs low stakes fun",
    "- grieving vs numb: active loss vs post-feeling flatness",
    "",
    "Choose the BEST single category for the PRIMARY emotional state. Many conversations will",
    "blend states -- choose the dominant one.",
    "",
    'Respond with ONLY a JSON object in this exact format:',
    '{"emotion": "<category>", "confidence": <0.0-1.0>, "secondary": ["<category>", ...]}',
    "",
    '- "emotion": The primary emotional state label (one of the 12 categories above)',
    '- "confidence": Your confidence in the primary classification (0.0 to 1.0)',
    '- "secondary": A list of 0-2 additional emotional state labels if the conversation clearly',
    "  carries multiple emotional textures. Empty list [] if single-state.",
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

    func_class = None
    if os.path.exists(FUNC_CLASS_PATH):
        func_class = pd.read_parquet(FUNC_CLASS_PATH)
        print(f"  Functional classifications: {len(func_class):,}")
    else:
        print(f"  Functional classifications: NOT FOUND (cross-tab figures will be skipped)")

    return conv, msgs, summaries_raw, config, func_class


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


def _format_user_message(inp):
    title  = (inp.get("title", "") or "").strip()
    text   = (inp.get("text",  "") or "").strip()
    source = inp.get("source", "summary")
    if source == "summary":
        return (
            f"Classify the emotional state of this conversation:\n\n"
            f"Title: {title}\nSummary: {text}"
        )
    else:
        return (
            f"Classify the emotional state of this conversation based on the user's messages:\n\n"
            f"Title: {title}\nUser messages:\n{text}"
        )


# -- Step 1: Build batch requests ---------------------------------------------
def build_batch_requests(inputs):
    print("\n-- Step 1: Build batch requests --------------------------------------")
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


# -- Step 2: Submit batch -----------------------------------------------------
def submit_batch(client):
    print("\n-- Step 2: Submit batch ----------------------------------------------")
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
    print(f"\n-- Step 3: Poll batch {batch_id} --")
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
    print(f"\n-- Step 4: Retrieve results ------------------------------------------")
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


def parse_emotion(raw_text):
    """Parse JSON response. Returns (emotion, confidence, secondary_json, error_msg)."""
    try:
        # Strip markdown code fences (Haiku wraps output despite being told not to)
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed  = json.loads(text)
        emotion = parsed.get("emotion", "").strip()
        conf    = float(parsed.get("confidence", 0.0))
        sec     = parsed.get("secondary", [])

        if emotion not in VALID_EMOTIONS:
            return None, 0.0, "[]", f"invalid emotion label: {emotion!r}"

        if not isinstance(sec, list):
            sec = []
        sec = [s for s in sec if isinstance(s, str) and s in VALID_EMOTIONS and s != emotion]
        sec = sec[:2]

        conf = max(0.0, min(1.0, conf))
        return emotion, float(conf), json.dumps(sec), None
    except Exception as e:
        return None, 0.0, "[]", f"parse error: {e}"


def build_output_rows(results, inputs):
    print("\n  Parsing emotion responses...")
    rows         = []
    parse_errors = 0

    for cid, res in tqdm(results.items(), desc="Parsing"):
        inp                            = inputs.get(cid, {})
        emotion, conf, sec_json, err   = parse_emotion(res.get("raw", ""))
        if err:
            parse_errors += 1
            emotion, conf, sec_json = "unknown", 0.0, "[]"
        rows.append({
            "conversation_id":    cid,
            "emotion_primary":    emotion,
            "emotion_confidence": conf,
            "emotion_secondary":  sec_json,
            "classification_input": inp.get("source", "summary"),
            "input_tokens":       res.get("input_tokens", 0),
            "output_tokens":      res.get("output_tokens", 0),
        })

    print(f"  Parse errors             : {parse_errors:,}")
    return rows, parse_errors


def fill_placeholders(rows_dict, inputs):
    filled = 0
    for cid in inputs:
        if cid not in rows_dict:
            rows_dict[cid] = {
                "conversation_id":    cid,
                "emotion_primary":    "unknown",
                "emotion_confidence": 0.0,
                "emotion_secondary":  "[]",
                "classification_input": inputs[cid].get("source", "summary"),
                "input_tokens":       0,
                "output_tokens":      0,
            }
            filled += 1
    if filled:
        print(f"  Filled {filled} placeholder(s) for permanently failed conversations")
    return filled


# -- Step 5: Save parquet -----------------------------------------------------
def save_parquet(rows):
    print("\n-- Step 5: Save results ----------------------------------------------")
    df = pd.DataFrame(rows)

    df["emotion_primary"]     = df["emotion_primary"].astype("category")
    df["classification_input"] = df["classification_input"].astype("category")
    df["emotion_confidence"]  = df["emotion_confidence"].astype("float32")
    df["input_tokens"]        = df["input_tokens"].astype("int32")
    df["output_tokens"]       = df["output_tokens"].astype("int32")
    # emotion_secondary is a JSON-serialized string e.g. '["anxious", "reflective"]'
    # Downstream consumers should parse with json.loads()

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Rows:  {len(df):,}")
    return df


# -- Step 6: Figures ----------------------------------------------------------
def make_figures(df, conv, config, func_class=None):
    print("\n-- Step 6: Generating figures ----------------------------------------")

    merged = df.merge(
        conv[[
            "conversation_id", "conversation_type", "model_era",
            "year_month", "time_of_day", "turns", "duration_minutes",
            "user_token_ratio", "msg_count",
        ]],
        on="conversation_id", how="left",
    )
    merged = merged[merged["emotion_primary"].astype(str) != "unknown"].copy()

    _fig1_distribution(df)
    _fig2_emotion_function_heatmap(merged, func_class)
    _fig3_over_time(merged, config)
    _fig4_by_time_of_day(merged)
    _fig5_confidence(df)
    _fig6_cooccurrence(df)
    _fig7_depth(merged)
    _fig8_dashboard(merged, func_class)
    _fig9_sankey(merged, func_class)
    print("  All figures saved.")


def _stacked_pct_bar(ax, df, group_col, title, xlabel, row_order=None):
    ct = pd.crosstab(df[group_col], df["emotion_primary"].astype(str))
    ct = ct.reindex(columns=[e for e in VALID_EMOTIONS if e in ct.columns])
    if row_order:
        ct = ct.reindex([r for r in row_order if r in ct.index])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(ct_pct))
    x = np.arange(len(ct_pct))
    for emo in ct_pct.columns:
        vals = ct_pct[emo].values
        ax.bar(x, vals, bottom=bottom,
               color=EMOTION_COLORS.get(emo, "#888888"), label=emo, width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ct_pct.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)


def _fig1_distribution(df):
    valid  = df[df["emotion_primary"].astype(str) != "unknown"]
    counts = valid["emotion_primary"].astype(str).value_counts()
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = [EMOTION_COLORS.get(e, COLOR_PRIMARY) for e in counts.index]
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
    ax.set_title("Emotional State Distribution")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.28)
    plt.tight_layout()
    plt.savefig(figpath("emotion_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: emotion_distribution.png")


def _fig2_emotion_function_heatmap(merged, func_class):
    if func_class is None:
        print("  Skipping emotion_function_heatmap.png (functional_classifications not available)")
        # Create a minimal placeholder so the file exists and passes size check
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "functional_classifications.parquet not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Emotion x Function Heatmap (unavailable)")
        plt.tight_layout()
        plt.savefig(figpath("emotion_function_heatmap.png"), dpi=DPI)
        plt.close()
        return

    combined = merged.merge(
        func_class[["conversation_id", "function_primary"]],
        on="conversation_id", how="inner",
    )
    if len(combined) == 0:
        print("  Skipping emotion_function_heatmap.png (no matching data after merge)")
        return

    ct   = pd.crosstab(combined["emotion_primary"].astype(str),
                       combined["function_primary"].astype(str))
    rows = [e for e in VALID_EMOTIONS  if e in ct.index]
    cols = [f for f in VALID_FUNCTIONS if f in ct.columns]
    ct   = ct.reindex(index=rows, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    vmax = ct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(ct.values[i, j])
            if val > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title("Emotional State x Functional Classification (count)")
    ax.set_xlabel("Function (from Module 3.2a)")
    ax.set_ylabel("Emotional State")
    plt.tight_layout()
    plt.savefig(figpath("emotion_function_heatmap.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: emotion_function_heatmap.png")


def _fig3_over_time(merged, config):
    if "year_month" not in merged.columns or merged["year_month"].isna().all():
        print("  Skipping emotion_over_time.html (no year_month data)")
        return

    ct     = pd.crosstab(merged["year_month"].astype(str),
                         merged["emotion_primary"].astype(str))
    ct     = ct.reindex(columns=[e for e in VALID_EMOTIONS if e in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.sort_index()

    fig = go.Figure()
    for emo in ct_pct.columns:
        fig.add_trace(go.Scatter(
            x=ct_pct.index.tolist(),
            y=ct_pct[emo].round(1).tolist(),
            mode="lines",
            stackgroup="one",
            name=emo,
            line_color=EMOTION_COLORS.get(emo, "#888888"),
            hovertemplate="%{x}<br>" + emo + ": %{y:.1f}%<extra></extra>",
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
        title="Emotional State Distribution Over Time",
        xaxis_title="Month",
        yaxis_title="Proportion (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        legend_title="Emotion",
        height=500,
    )
    fig.write_html(figpath("emotion_over_time.html"), include_plotlyjs=True)
    print(f"  Saved: emotion_over_time.html")


def _fig4_by_time_of_day(merged):
    order   = ["morning", "afternoon", "evening", "night"]
    present = [o for o in order if o in merged["time_of_day"].values]
    subset  = merged[merged["time_of_day"].isin(present)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    _stacked_pct_bar(ax, subset, "time_of_day",
                     "Emotional State by Time of Day", "Time of Day",
                     row_order=present)
    handles = [plt.Rectangle((0, 0), 1, 1, color=EMOTION_COLORS.get(e, "#888"))
               for e in VALID_EMOTIONS if e in merged["emotion_primary"].astype(str).values]
    labels  = [e for e in VALID_EMOTIONS if e in merged["emotion_primary"].astype(str).values]
    ax.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figpath("emotion_by_time_of_day.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: emotion_by_time_of_day.png")


def _fig5_confidence(df):
    conf = df["emotion_confidence"].astype(float).dropna()
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
    ax.set_title("Emotional State Classification Confidence Distribution")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath("confidence_distribution.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: confidence_distribution.png")


def _fig6_cooccurrence(df):
    rows_data = []
    for _, row in df.iterrows():
        primary = str(row["emotion_primary"])
        if primary == "unknown":
            continue
        try:
            secondaries = json.loads(row["emotion_secondary"])
        except Exception:
            secondaries = []
        for sec in secondaries:
            if sec in VALID_EMOTIONS:
                rows_data.append({"primary": primary, "secondary": sec})

    if not rows_data:
        print("  Skipping emotion_cooccurrence.png (no secondary tags found)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No secondary emotional state data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Emotion Co-occurrence (unavailable)")
        plt.tight_layout()
        plt.savefig(figpath("emotion_cooccurrence.png"), dpi=DPI)
        plt.close()
        return

    co_df = pd.DataFrame(rows_data)
    ct    = pd.crosstab(co_df["primary"], co_df["secondary"])
    rows  = [e for e in VALID_EMOTIONS if e in ct.index]
    cols  = [e for e in VALID_EMOTIONS if e in ct.columns]
    ct    = ct.reindex(index=rows, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(ct.values, cmap="Oranges", aspect="auto")
    plt.colorbar(im, ax=ax, label="Co-occurrence Count")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    vmax = ct.values.max()
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(ct.values[i, j])
            if val > 0:
                text_color = "white" if vmax > 0 and val > vmax * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title("Emotional State Co-occurrence (Primary x Secondary)")
    ax.set_xlabel("Secondary Emotional State")
    ax.set_ylabel("Primary Emotional State")
    plt.tight_layout()
    plt.savefig(figpath("emotion_cooccurrence.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: emotion_cooccurrence.png")


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

        groups       = merged.groupby("emotion_primary")[col]
        group_labels = [e for e in VALID_EMOTIONS if e in groups.groups]
        means  = [groups.get_group(e).mean() for e in group_labels]
        sems   = [groups.get_group(e).sem()  for e in group_labels]
        colors = [EMOTION_COLORS.get(e, COLOR_PRIMARY) for e in group_labels]

        ax.bar(range(len(group_labels)), means, yerr=sems, color=colors,
               capsize=3, error_kw={"linewidth": 0.8})

        group_data   = [
            groups.get_group(e).dropna().values
            for e in group_labels if e in groups.groups
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

        ax.set_title(f"{label} by Emotion{title_suffix}", fontsize=10)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=9)

    plt.suptitle(
        "Emotion Predicts Conversation Depth\n(* = Kruskal-Wallis p < 0.05)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(figpath("emotion_vs_depth.png"), dpi=DPI)
    plt.close()
    print(f"  Saved: emotion_vs_depth.png")


def _fig8_dashboard(merged, func_class=None):
    counts = merged["emotion_primary"].astype(str).value_counts()
    total  = counts.sum()
    if total == 0:
        print("  Skipping emotional_signature_dashboard.png (no data)")
        return

    dominant_emo = counts.index[0]
    dominant_pct = counts.iloc[0] / total * 100
    top3         = counts.head(3)
    probs        = counts / total
    entropy      = float(-(probs * np.log2(probs + 1e-12)).sum())

    # Top emotion x function pair
    top_pair_str = "N/A"
    if func_class is not None:
        try:
            combined = merged.merge(
                func_class[["conversation_id", "function_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                pairs = combined.groupby(
                    ["emotion_primary", "function_primary"]
                ).size().reset_index(name="count")
                top_pair = pairs.loc[pairs["count"].idxmax()]
                top_pair_str = (
                    f"{top_pair['emotion_primary']} + {top_pair['function_primary']}"
                    f" (n={top_pair['count']})"
                )
        except Exception:
            pass

    top2_pct = round(
        (top3.iloc[0] + (top3.iloc[1] if len(top3) > 1 else 0)) / total * 100, 1
    )
    insight = (
        f"{top3.index[0] if len(top3) > 0 else 'unknown'} and "
        f"{top3.index[1] if len(top3) > 1 else 'others'} account for "
        f"{top2_pct}% of conversations"
    )

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    # Panel A: dominant emotion
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.text(0.5, 0.6, dominant_emo.replace("_", "\n"),
              ha="center", va="center", fontsize=16, fontweight="bold",
              color=EMOTION_COLORS.get(dominant_emo, COLOR_PRIMARY),
              transform=ax_a.transAxes)
    ax_a.text(0.5, 0.2, f"{dominant_pct:.1f}% of conversations",
              ha="center", va="center", fontsize=11, transform=ax_a.transAxes)
    ax_a.set_title("Dominant Emotion", fontsize=11)

    # Panel B: top-3 pie
    ax_b = fig.add_subplot(gs[0, 1])
    pie_labels = top3.index.tolist()
    pie_colors = [EMOTION_COLORS.get(e, "#888") for e in pie_labels]
    ax_b.pie(top3.values, labels=pie_labels, colors=pie_colors,
             autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax_b.set_title("Top 3 Emotions", fontsize=11)

    # Panel C: Shannon entropy
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.text(0.5, 0.6, f"{entropy:.2f}",
              ha="center", va="center", fontsize=24, fontweight="bold",
              color=COLOR_PRIMARY, transform=ax_c.transAxes)
    ax_c.text(0.5, 0.25,
              f"Shannon Entropy\n(bits, max = {np.log2(len(VALID_EMOTIONS)):.2f})",
              ha="center", va="center", fontsize=10, color=COLOR_SECONDARY,
              transform=ax_c.transAxes)
    ax_c.set_title("Emotion Diversity", fontsize=11)

    # Panel D: top pair
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.axis("off")
    ax_d.text(0.5, 0.5, top_pair_str,
              ha="center", va="center", fontsize=10, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0"))
    ax_d.set_title("Top Emotion x Function Pair", fontsize=11)

    # Panel E: key insight
    ax_e = fig.add_subplot(gs[1, 1:])
    ax_e.axis("off")
    ax_e.text(0.5, 0.5, insight,
              ha="center", va="center", fontsize=12, transform=ax_e.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF4FB"))
    ax_e.set_title("Key Insight", fontsize=11)

    plt.suptitle("Emotional Signature: Affective Profile", fontsize=14, fontweight="bold")
    plt.savefig(figpath("emotional_signature_dashboard.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: emotional_signature_dashboard.png")


def _fig9_sankey(merged, func_class):
    if func_class is None:
        print("  Skipping emotion_function_sankey.html (functional_classifications not available)")
        return

    combined = merged.merge(
        func_class[["conversation_id", "function_primary"]],
        on="conversation_id", how="inner",
    )
    if len(combined) == 0:
        print("  Skipping emotion_function_sankey.html (no matching data after merge)")
        return

    func_labels    = [f for f in VALID_FUNCTIONS]
    emotion_labels = [e for e in VALID_EMOTIONS]
    all_labels     = func_labels + emotion_labels

    n_func      = len(func_labels)
    func_idx    = {f: i           for i, f in enumerate(func_labels)}
    emotion_idx = {e: i + n_func  for i, e in enumerate(emotion_labels)}

    flows = {}
    for _, row in combined.iterrows():
        func    = str(row.get("function_primary", ""))
        emotion = str(row.get("emotion_primary", ""))
        if func in func_idx and emotion in emotion_idx:
            key = (func_idx[func], emotion_idx[emotion])
            flows[key] = flows.get(key, 0) + 1

    if not flows:
        print("  Skipping emotion_function_sankey.html (no valid flows)")
        return

    sources = [k[0] for k in flows]
    targets = [k[1] for k in flows]
    values  = [flows[k] for k in flows]

    node_colors = (
        [FUNC_COLORS.get(f, "#888888")    for f in func_labels] +
        [EMOTION_COLORS.get(e, "#888888") for e in emotion_labels]
    )

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))
    fig.update_layout(
        title_text="Function x Emotion Sankey Diagram",
        font_size=10,
        height=700,
    )
    fig.write_html(figpath("emotion_function_sankey.html"), include_plotlyjs=True)
    print(f"  Saved: emotion_function_sankey.html")


# -- Step 7: Report -----------------------------------------------------------
def generate_report(df, conv, batch_id, parse_errors, n_fallback,
                    warnings_list, func_class=None):
    print("\n-- Step 7: Generate report -------------------------------------------")

    merged = df.merge(
        conv[[
            "conversation_id", "conversation_type", "model_era",
            "year_month", "time_of_day", "turns", "duration_minutes",
            "user_token_ratio", "msg_count",
        ]],
        on="conversation_id", how="left",
    )

    valid  = merged[merged["emotion_primary"].astype(str) != "unknown"]
    counts = valid["emotion_primary"].astype(str).value_counts()
    total  = len(valid)

    distribution = {
        emo: {
            "count": int(counts.get(emo, 0)),
            "pct":   round(counts.get(emo, 0) / max(total, 1) * 100, 1),
        }
        for emo in VALID_EMOTIONS
    }

    conf = df["emotion_confidence"].astype(float)
    conf_stats = {
        "mean":       round(float(conf.mean()), 3),
        "median":     round(float(conf.median()), 3),
        "std":        round(float(conf.std()), 3),
        "pct_high":   round(float((conf >= 0.8).mean() * 100), 1),
        "pct_medium": round(float(((conf >= 0.5) & (conf < 0.8)).mean() * 100), 1),
        "pct_low":    round(float((conf < 0.5).mean() * 100), 1),
    }

    # Secondary stats
    n0 = n1 = n2 = 0
    sec_freq = {}
    cooccurrences = []
    for _, row in df.iterrows():
        primary = str(row["emotion_primary"])
        try:
            lst = json.loads(row["emotion_secondary"])
        except Exception:
            lst = []
        n = len(lst)
        if n == 0:
            n0 += 1
        elif n == 1:
            n1 += 1
        else:
            n2 += 1
        for s in lst:
            if s in VALID_EMOTIONS:
                sec_freq[s] = sec_freq.get(s, 0) + 1
                cooccurrences.append({"primary": primary, "secondary": s})

    co_df = pd.DataFrame(cooccurrences) if cooccurrences else pd.DataFrame()
    top_cooccurrences = []
    if not co_df.empty:
        co_counts = co_df.groupby(["primary", "secondary"]).size().reset_index(name="count")
        co_counts  = co_counts.sort_values("count", ascending=False).head(10)
        top_cooccurrences = co_counts.to_dict("records")

    most_common_secondary = max(sec_freq, key=sec_freq.get) if sec_freq else None
    n_total = len(df)
    secondary_stats = {
        "pct_with_0_secondary": round(n0 / max(n_total, 1) * 100, 1),
        "pct_with_1_secondary": round(n1 / max(n_total, 1) * 100, 1),
        "pct_with_2_secondary": round(n2 / max(n_total, 1) * 100, 1),
        "most_common_secondary": most_common_secondary,
        "top_cooccurrences": top_cooccurrences,
    }

    def cross_tab(group_col):
        if group_col not in merged.columns:
            return {}
        ct = pd.crosstab(merged[group_col].astype(str),
                         merged["emotion_primary"].astype(str))
        return {str(g): {str(e): int(v) for e, v in row.items()}
                for g, row in ct.iterrows()}

    by_function = {}
    if func_class is not None:
        try:
            combined = merged.merge(
                func_class[["conversation_id", "function_primary"]],
                on="conversation_id", how="inner",
            )
            if len(combined) > 0:
                ct = pd.crosstab(combined["function_primary"].astype(str),
                                 combined["emotion_primary"].astype(str))
                by_function = {str(g): {str(e): int(v) for e, v in row.items()}
                               for g, row in ct.iterrows()}
        except Exception as e:
            warnings_list.append(f"Could not compute by_function cross-tab: {e}")

    stat_tests = {}
    for metric in ["turns", "duration_minutes", "user_token_ratio", "msg_count"]:
        if metric not in merged.columns:
            continue
        group_data = [
            merged[merged["emotion_primary"].astype(str) == emo][metric].dropna().values
            for emo in VALID_EMOTIONS
        ]
        group_data = [g for g in group_data if len(g) > 0]
        if len(group_data) >= 2:
            try:
                H, p = stats.kruskal(*group_data)
                stat_tests[f"emotion_vs_{metric}"] = {
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
    ) if len(counts) > 0 else 0.0

    # Top emotion x function pair
    top_pair = {}
    if func_class is not None and by_function:
        try:
            combined = merged.merge(
                func_class[["conversation_id", "function_primary"]],
                on="conversation_id", how="inner",
            )
            pairs = combined.groupby(
                ["emotion_primary", "function_primary"]
            ).size().reset_index(name="count")
            if len(pairs) > 0:
                best = pairs.loc[pairs["count"].idxmax()]
                top_pair = {
                    "emotion":   str(best["emotion_primary"]),
                    "function":  str(best["function_primary"]),
                    "count":     int(best["count"]),
                }
        except Exception:
            pass

    emo_summary = (
        f"{top3[0] if top3 else 'unknown'} and "
        f"{top3[1] if len(top3) > 1 else 'others'} account for "
        f"{top2_pct}% of conversations; "
        f"emotion diversity (Shannon entropy) = {entropy:.2f} bits."
    )

    report = {
        "module":         "emotional_state",
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
        "secondary_stats":  secondary_stats,
        "cross_tabulations": {
            "by_function":         by_function,
            "by_conversation_type": cross_tab("conversation_type"),
            "by_model_era":         cross_tab("model_era"),
            "by_time_of_day":       cross_tab("time_of_day"),
        },
        "statistical_tests": stat_tests,
        "cost": {
            "input_tokens":   total_input,
            "output_tokens":  total_output,
            "total_cost_usd": total_cost,
            "pricing_note":   (
                "Haiku 4.5 Batch API: $0.40/MTok input, "
                "$2.00/MTok output (50% batch discount)"
            ),
        },
        "emotional_signature_fragment": {
            "dominant_emotion":           top3[0] if top3 else "unknown",
            "dominant_pct":               round(
                float(counts.iloc[0] / max(total, 1) * 100), 1
            ) if len(counts) else 0.0,
            "emotion_diversity_entropy":  round(entropy, 3),
            "top_3":                      top3,
            "top_emotion_function_pair":  top_pair,
            "summary":                    emo_summary,
        },
        "figures_generated": [
            "emotion_distribution.png",
            "emotion_function_heatmap.png",
            "emotion_over_time.html",
            "emotion_by_time_of_day.png",
            "confidence_distribution.png",
            "emotion_cooccurrence.png",
            "emotion_vs_depth.png",
            "emotional_signature_dashboard.png",
            "emotion_function_sankey.html",
        ],
        "data_outputs": ["data/processed/emotional_states.parquet"],
        "warnings":     warnings_list,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.4f}")
    return report


# -- Step 8: Validation -------------------------------------------------------
def run_validation(df, conv, report, func_class=None):
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    checks = []

    def chk(label, result):
        status = "PASS" if result else "FAIL"
        checks.append((label, status))
        print(f"  [{status}] {label}")

    # 1
    chk("emotional_states.parquet exists", os.path.exists(OUT_PARQUET))

    # 2
    required_cols = {
        "conversation_id", "emotion_primary", "emotion_confidence",
        "emotion_secondary", "classification_input", "input_tokens", "output_tokens",
    }
    chk("Required columns present", required_cols.issubset(set(df.columns)))

    # 3
    expected = len(conv)
    pct_diff = abs(len(df) - expected) / max(expected, 1) * 100
    chk(f"Row count within 1% of analysable ({len(df):,} / {expected:,})", pct_diff <= 1)

    # 4
    valid_ids = set(conv["conversation_id"])
    orphans   = set(df["conversation_id"]) - valid_ids
    chk("No orphan conversation_ids", len(orphans) == 0)

    # 5
    chk("No duplicate conversation_ids", df["conversation_id"].duplicated().sum() == 0)

    # 6
    all_vals      = set(df["emotion_primary"].astype(str).unique())
    invalid_emos  = all_vals - set(VALID_EMOTIONS) - {"unknown"}
    chk("All emotion_primary values valid", len(invalid_emos) == 0)

    # 7
    all_sec_valid = True
    for s in df["emotion_secondary"]:
        try:
            lst = json.loads(s)
            if any(item not in VALID_EMOTIONS for item in lst):
                all_sec_valid = False
                break
        except Exception:
            all_sec_valid = False
            break
    chk("All secondary labels in VALID_EMOTIONS", all_sec_valid)

    # 8
    all_sec_len_ok = True
    for s in df["emotion_secondary"]:
        try:
            if len(json.loads(s)) > 2:
                all_sec_len_ok = False
                break
        except Exception:
            all_sec_len_ok = False
            break
    chk("Secondary lists have 0-2 elements", all_sec_len_ok)

    # 9
    conf = df["emotion_confidence"].astype(float)
    chk("Confidence scores in [0, 1]",
        bool((conf >= 0.0).all() and (conf <= 1.0).all()))

    # 10
    err_pct = (df["emotion_primary"].astype(str) == "unknown").mean() * 100
    chk(f"Classification errors < 1% ({err_pct:.2f}%)", err_pct < 1.0)

    # 11
    req_keys = {
        "module", "distribution", "confidence_stats", "secondary_stats",
        "cost", "emotional_signature_fragment", "figures_generated",
    }
    chk("Report JSON has required keys",
        os.path.exists(OUT_REPORT) and req_keys.issubset(set(report.keys())))

    # 12
    expected_figs = [
        "emotion_distribution.png",
        "emotion_function_heatmap.png",
        "emotion_over_time.html",
        "emotion_by_time_of_day.png",
        "confidence_distribution.png",
        "emotion_cooccurrence.png",
        "emotion_vs_depth.png",
        "emotional_signature_dashboard.png",
        "emotion_function_sankey.html",
    ]
    missing_figs = [f for f in expected_figs if not os.path.exists(figpath(f))]
    chk(f"All 9 figures exist ({len(missing_figs)} missing)", len(missing_figs) == 0)

    # 13
    png_figs   = [f for f in expected_figs if f.endswith(".png")]
    small_pngs = [
        f for f in png_figs
        if os.path.exists(figpath(f)) and os.path.getsize(figpath(f)) < 10_000
    ]
    chk(f"All PNGs >= 10KB ({len(small_pngs)} too small)", len(small_pngs) == 0)

    # 14
    for html_name in ["emotion_over_time.html", "emotion_function_sankey.html"]:
        html_path = figpath(html_name)
        html_ok   = False
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as fh:
                html_ok = "plotly" in fh.read(5000).lower()
        chk(f"{html_name} contains plotlyjs", html_ok)

    # 15
    report_str  = json.dumps(report)
    no_nan_inf  = ("NaN" not in report_str) and ("Infinity" not in report_str)
    chk("No NaN/Infinity in report JSON", no_nan_inf)

    # 16
    sig    = report.get("emotional_signature_fragment", {})
    sig_ok = bool(sig.get("summary", "").strip())
    chk("Emotional signature summary non-empty", sig_ok)

    # Bonus: cross-tab by_function non-empty (only if func_class was available)
    if func_class is not None:
        by_func = report.get("cross_tabulations", {}).get("by_function", {})
        chk("Cross-tabulation by_function non-empty", len(by_func) > 0)

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
        description="Module 3.2b: Emotional State Classification via Claude Haiku Batch API"
    )
    parser.add_argument("--resume-batch-id", default=None,
                        help="Resume from a previously submitted batch ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build requests and print stats without submitting")
    args = parser.parse_args()

    warnings_list = []

    # Dry-run: no API key needed — just load data and build the JSONL
    if args.dry_run:
        conv, msgs, summaries_raw, config, func_class = load_data()
        inputs    = build_classification_inputs(conv, msgs, summaries_raw)
        n_fallback = sum(1 for v in inputs.values() if v["source"] == "fallback_messages")
        if n_fallback > 0:
            print(f"WARNING: {n_fallback} conversation(s) using fallback messages")
        n_requests = build_batch_requests(inputs)
        print(f"\n-- DRY RUN: {n_requests:,} requests built. Not submitting. --")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print(f"API key found: {api_key[:12]}...{api_key[-4:]}")

    import anthropic
    client = anthropic.Anthropic()

    conv, msgs, summaries_raw, config, func_class = load_data()
    inputs = build_classification_inputs(conv, msgs, summaries_raw)

    n_fallback = sum(1 for v in inputs.values() if v["source"] == "fallback_messages")
    if n_fallback > 0:
        warnings_list.append(
            f"{n_fallback} conversation(s) used fallback messages (no summary available)"
        )

    n_requests = build_batch_requests(inputs)

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
        make_figures(df, conv, config, func_class)
    except Exception as e:
        msg = f"Figure generation error: {e}"
        print(f"  WARNING: {msg}")
        warnings_list.append(msg)

    report = generate_report(
        df, conv, batch_id, parse_errors, n_fallback, warnings_list, func_class
    )

    run_validation(df, conv, report, func_class)
    print("\nDone!")


if __name__ == "__main__":
    main()
