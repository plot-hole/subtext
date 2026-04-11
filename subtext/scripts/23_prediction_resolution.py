"""
Module 23: Prediction Resolution
Takes the 898 predictions from Module 19 and determines which came true,
which were wrong, and which are unresolvable — using evidence from later
conversations in the corpus.

Uses Anthropic Batch API (Haiku) for classification. Evidence gathering
is entirely local — entity overlap + keyword matching against conversation
summaries within a 90-day window.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/23_prediction_resolution.py

    # Build evidence map only (no API call):
    python scripts/23_prediction_resolution.py --evidence-only

    # Build JSONL but don't submit:
    python scripts/23_prediction_resolution.py --dry-run

    # Resume a previously submitted batch:
    python scripts/23_prediction_resolution.py --resume-batch-id msgbatch_xxxxx
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
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
INTERIM = PROJECT_ROOT / "data" / "interim"
REPORTS = PROJECT_ROOT / "outputs" / "reports"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "prediction_resolution"

HYPOTHESES_PATH = PROCESSED / "hypotheses.parquet"
CONVOS_PATH = PROCESSED / "conversations_clean.parquet"
SUMMARIES_PATH = PROCESSED / "conversation_summaries.parquet"
ENTITY_ATT_PATH = PROCESSED / "entity_attention.parquet"

OUT_PARQUET = PROCESSED / "prediction_outcomes.parquet"
BATCH_FILE = INTERIM / "prediction_resolution_batch.jsonl"
BATCH_ID_FILE = INTERIM / "prediction_resolution_batch_id.txt"
OUT_REPORT = REPORTS / "prediction_resolution_report.json"

for d in [INTERIM, REPORTS, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model & Pricing
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 200
PRICE_INPUT_PER_MTOK = 0.40    # $0.80/MTok * 0.5 batch discount
PRICE_OUTPUT_PER_MTOK = 2.00   # $4.00/MTok * 0.5 batch discount

# ---------------------------------------------------------------------------
# Evidence Config
# ---------------------------------------------------------------------------
EVIDENCE_WINDOW_DAYS = 90
MAX_EVIDENCE_PER_PREDICTION = 5

# Stop words for keyword extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "that", "this", "these", "those", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "my", "your", "our", "we",
    "me", "him", "us", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just", "about",
    "up", "down", "here", "there", "when", "where", "why", "how", "all",
    "any", "what", "which", "who", "whom", "if", "because", "while",
    "also", "even", "still", "already", "well", "really", "going",
    "think", "know", "like", "want", "get", "make", "feel", "see",
    "say", "said", "tell", "told", "user", "conversation",
}

# Load excluded entities from config and add to stop words
_entity_config = PROJECT_ROOT / "config" / "known_entities.json"
if _entity_config.exists():
    with open(_entity_config, encoding="utf-8") as _f:
        _cfg = json.load(_f)
    STOP_WORDS |= {e.lower() for e in _cfg.get("exclude_entities", [])}
    del _cfg, _f

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = ["#59A14F", "#E15759", "#F28E2B", "#BAB0AC", "#4E79A7", "#B07AA1"]
OUTCOME_COLORS = {
    "confirmed": "#59A14F",
    "disconfirmed": "#E15759",
    "partially_confirmed": "#F28E2B",
    "unresolvable": "#BAB0AC",
}
DPI = 150
sns.set_theme(style="whitegrid", font_scale=0.9)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are classifying whether a prediction made by a user came true, based on evidence from their later conversations with an AI assistant.

You will see:
1. The prediction text (with resolved pronouns in [brackets])
2. The prediction's topic, confidence level, and temporal scope
3. Up to 5 summaries of conversations that occurred AFTER the prediction

STEP 1: Determine if this is actually a testable prediction.
These are NOT testable predictions and should be classified "unresolvable":
- Questions ("Is she ready?", "Can I actually do this?", "Will this work?")
- Self-assessments or judgments about the present ("I should be further along", "I worked myself up over nothing")
- Statements of current intent or desire ("I want to extract data", "I will need to know how to proceed")
- Vague or poetic statements that cannot be clearly verified ("energy will weave in and out", "the pull will persist")
If the text is not a testable prediction about a future outcome, classify as "unresolvable" with confidence 0.3 and note "not a testable prediction" in your evidence summary.

STEP 2: If it IS a testable prediction, classify the OUTCOME (not the mechanism):
- "confirmed": The predicted outcome occurred, even if the exact mechanism or path differed from what was predicted. Focus on WHAT happened, not HOW.
- "disconfirmed": Clear evidence the predicted outcome did NOT occur — the opposite happened. IMPORTANT: Absence of evidence is NOT disconfirmation. If the evidence simply doesn't mention the predicted event, that is "unresolvable", not "disconfirmed."
- "partially_confirmed": The core prediction was directionally correct but the details, timing, or magnitude differed significantly.
- "unresolvable": The evidence doesn't address this prediction, or is too ambiguous to determine the outcome.

CRITICAL RULES:
- Default to "unresolvable" unless you have CLEAR evidence for another category.
- NEVER classify as "disconfirmed" just because the evidence doesn't mention the predicted event. That's "unresolvable."
- Focus on whether the OUTCOME occurred, not whether the exact mechanism matched. If someone predicted "X will ask Y to do Z" and X achieved Z through a different approach, that is confirmed or partially_confirmed, not disconfirmed.
- Verify that evidence is about the RIGHT people. If a prediction is about Person A doing something, evidence about Person B doing that same thing is irrelevant.
- Catastrophic self-predictions ("I'll never get a job", "I'll become a stalker") should have a HIGH bar for disconfirmation — only disconfirm if there is direct evidence of the opposite.
- Cap your confidence at 0.7 for vague or abstract predictions, even if evidence seems to match.

Respond with JSON only (no markdown, no explanation outside JSON):
{"outcome": "<category>", "confidence": 0.0-1.0, "evidence_summary": "1-2 sentence explanation of your reasoning"}"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def clean(v):
    """Convert numpy types to Python natives, round floats."""
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
    return str(FIG_DIR / name)


def extract_keywords(text, min_len=3):
    """Extract meaningful keywords from text."""
    if not text or not isinstance(text, str):
        return set()
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return {w for w in words if len(w) >= min_len and w not in STOP_WORDS}


def extract_entities_from_text(text, known_entities):
    """Find known entity names mentioned in text."""
    if not text or not isinstance(text, str):
        return set()
    text_lower = text.lower()
    found = set()
    for entity in known_entities:
        if entity.lower() in text_lower:
            found.add(entity)
    return found


# ===========================================================================
# STEP 1: Load Data
# ===========================================================================
def load_data():
    """Load all required datasets."""
    print("=== Step 1: Load Data ===")

    # Hypotheses — filter to predictions
    hyp = pd.read_parquet(HYPOTHESES_PATH)
    predictions = hyp[hyp["type"] == "prediction"].copy()
    print(f"  Predictions: {len(predictions):,} / {len(hyp):,} total hypotheses")

    # Conversations
    convos = pd.read_parquet(CONVOS_PATH)
    convos["created_at"] = pd.to_datetime(convos["created_at"], utc=True)
    print(f"  Conversations: {len(convos):,}")

    # Summaries
    summaries = pd.read_parquet(SUMMARIES_PATH)
    print(f"  Summaries: {len(summaries):,}")

    # Entity attention
    entities = None
    if ENTITY_ATT_PATH.exists():
        entities = pd.read_parquet(ENTITY_ATT_PATH)
        print(f"  Entity attention: {len(entities):,} rows")
    else:
        print("  Entity attention: NOT FOUND (will use text-only matching)")

    return predictions, convos, summaries, entities


# ===========================================================================
# STEP 2: Build Evidence Map
# ===========================================================================
def build_evidence_map(predictions, convos, summaries, entities):
    """For each prediction, find candidate evidence conversations."""
    print("\n=== Step 2: Build Evidence Map ===")

    # Build lookups
    summary_lookup = dict(zip(summaries["conversation_id"], summaries["summary"]))
    timestamp_lookup = dict(zip(convos["conversation_id"], convos["created_at"]))

    # Entity lookup: conversation_id -> set of entities
    entity_by_conv = defaultdict(set)
    known_entities = set()
    if entities is not None:
        for _, row in entities.iterrows():
            entity_by_conv[row["conversation_id"]].add(row["entity"])
            known_entities.add(row["entity"])
    print(f"  Known entities: {len(known_entities)}")

    # All conversations sorted by time (for windowing)
    conv_times = sorted(timestamp_lookup.items(), key=lambda x: x[1])

    # Build evidence for each prediction
    evidence_map = {}
    stats = {"total": 0, "with_evidence": 0, "no_evidence": 0, "short_window": 0,
             "candidate_counts": []}

    for _, pred in tqdm(predictions.iterrows(), total=len(predictions), desc="  Building evidence"):
        hyp_id = pred["hypothesis_id"]
        pred_text = pred.get("resolved_text") or pred.get("raw_text", "")
        pred_topic = pred.get("topic", "")
        pred_conv_id = pred["conversation_id"]

        # Get prediction timestamp
        pred_ts = pred.get("message_timestamp")
        if pd.isna(pred_ts):
            pred_ts = timestamp_lookup.get(pred_conv_id)
        if pred_ts is None:
            evidence_map[hyp_id] = []
            stats["no_evidence"] += 1
            stats["total"] += 1
            continue
        pred_ts = pd.to_datetime(pred_ts, utc=True)

        # Window end
        corpus_end = max(t for _, t in conv_times)
        window_end = pred_ts + timedelta(days=EVIDENCE_WINDOW_DAYS)
        actual_window = min(window_end, corpus_end)
        short_window = (corpus_end - pred_ts).days < EVIDENCE_WINDOW_DAYS

        # Extract keywords and entities from prediction
        pred_keywords = extract_keywords(pred_text) | extract_keywords(pred_topic)
        pred_entities = extract_entities_from_text(pred_text, known_entities)

        # Find candidate conversations in window
        candidates = []
        for conv_id, conv_ts in conv_times:
            if conv_ts <= pred_ts or conv_ts > actual_window:
                continue
            if conv_id == pred_conv_id:
                continue
            if conv_id not in summary_lookup:
                continue

            summary = summary_lookup[conv_id]
            if not summary or not isinstance(summary, str):
                continue

            # Score this candidate
            score = 0.0

            # Entity overlap
            conv_entities = entity_by_conv.get(conv_id, set())
            entity_overlap = pred_entities & conv_entities
            score += len(entity_overlap) * 3.0

            # Keyword overlap with summary
            summary_words = set(re.findall(r'[a-zA-Z]+', summary.lower()))
            keyword_overlap = pred_keywords & summary_words
            score += len(keyword_overlap) * 1.0

            # Temporal proximity bonus (decay over 90 days)
            days_later = (conv_ts - pred_ts).days
            proximity_bonus = max(0, 1.0 - days_later / EVIDENCE_WINDOW_DAYS) * 0.5
            score += proximity_bonus

            if score > 0:
                candidates.append({
                    "conversation_id": conv_id,
                    "summary": summary,
                    "score": score,
                    "days_later": days_later,
                    "created_at": conv_ts,
                    "entity_overlap": list(entity_overlap),
                    "keyword_overlap_count": len(keyword_overlap),
                })

        # Sort by score, take top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:MAX_EVIDENCE_PER_PREDICTION]

        evidence_map[hyp_id] = top_candidates
        stats["total"] += 1
        stats["candidate_counts"].append(len(top_candidates))
        if len(top_candidates) > 0:
            stats["with_evidence"] += 1
        else:
            stats["no_evidence"] += 1
        if short_window:
            stats["short_window"] += 1

    avg_candidates = np.mean(stats["candidate_counts"]) if stats["candidate_counts"] else 0
    print(f"\n  Evidence map built:")
    print(f"    Total predictions:       {stats['total']}")
    print(f"    With evidence:           {stats['with_evidence']}")
    print(f"    No evidence:             {stats['no_evidence']}")
    print(f"    Short window (< 90d):    {stats['short_window']}")
    print(f"    Avg candidates/pred:     {avg_candidates:.1f}")

    return evidence_map, stats


# ===========================================================================
# STEP 3: Build Batch Requests
# ===========================================================================
def build_batch_requests(predictions, evidence_map):
    """Build JSONL batch file for Anthropic API."""
    print("\n=== Step 3: Build Batch Requests ===")

    pred_lookup = predictions.set_index("hypothesis_id")
    count = 0

    with open(BATCH_FILE, "w", encoding="utf-8") as f:
        for hyp_id, evidence in evidence_map.items():
            pred = pred_lookup.loc[hyp_id]
            pred_text = pred.get("resolved_text") or pred.get("raw_text", "")
            pred_topic = pred.get("topic", "unknown")
            confidence = pred.get("confidence_level", "unknown")
            temporal = pred.get("temporal_scope", "unspecified")
            pred_ts = pred.get("message_timestamp")
            if pd.notna(pred_ts):
                date_str = str(pred_ts)[:10]
            else:
                date_str = "unknown date"

            # Build user message
            parts = [
                f'PREDICTION (made {date_str}, confidence: {confidence}, temporal scope: {temporal}):',
                f'"{pred_text}"',
                f'',
                f'TOPIC: {pred_topic}',
            ]

            if evidence:
                parts.append('')
                parts.append('EVIDENCE FROM LATER CONVERSATIONS:')
                for i, ev in enumerate(evidence, 1):
                    ev_date = str(ev["created_at"])[:10]
                    days = ev["days_later"]
                    summary = ev["summary"][:500]  # truncate long summaries
                    parts.append(f'[{i}] ({ev_date}, {days}d later): {summary}')
            else:
                parts.append('')
                parts.append('NO EVIDENCE CONVERSATIONS FOUND IN THE 90-DAY WINDOW.')
                parts.append('Classify as "unresolvable" unless you have strong reason otherwise.')

            parts.append('')
            parts.append('Based on this evidence, classify the prediction outcome.')

            user_message = '\n'.join(parts)

            request = {
                "custom_id": hyp_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_message}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Batch file: {BATCH_FILE}")
    print(f"  Total requests: {count:,}")
    return count


# ===========================================================================
# STEP 4: Submit Batch
# ===========================================================================
def submit_batch(client):
    """Submit batch to Anthropic API."""
    print("\n=== Step 4: Submit Batch ===")
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


# ===========================================================================
# STEP 5: Poll Batch
# ===========================================================================
def poll_batch(client, batch_id):
    """Poll batch until completion."""
    print(f"\n=== Step 5: Poll Batch {batch_id} ===")
    print(f"  (Checking every 60 seconds. This may take 15-90 minutes.)\n")
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        total = (
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


# ===========================================================================
# STEP 6: Retrieve & Parse Results
# ===========================================================================
VALID_OUTCOMES = {"confirmed", "disconfirmed", "partially_confirmed", "unresolvable"}


def parse_outcome(raw_text):
    """Parse JSON response. Returns (outcome, confidence, evidence_summary, error)."""
    try:
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)

        outcome = data.get("outcome", "").lower().strip()
        if outcome not in VALID_OUTCOMES:
            return None, None, None, f"Invalid outcome: {outcome}"

        confidence = float(data.get("confidence", 0.5))
        evidence_summary = str(data.get("evidence_summary", ""))

        return outcome, confidence, evidence_summary, None
    except Exception as e:
        return None, None, None, str(e)


def retrieve_results(client, batch_id):
    """Retrieve and parse batch results."""
    print(f"\n=== Step 6: Retrieve Results ===")
    results = {}
    errors = []

    for result in client.messages.batches.results(batch_id):
        hyp_id = result.custom_id
        if result.result.type == "succeeded":
            response = result.result.message
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            results[hyp_id] = {
                "raw": raw_text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            errors.append({
                "hypothesis_id": hyp_id,
                "error_type": result.result.type,
                "error": str(result.result.error) if hasattr(result.result, "error") else "unknown",
            })

    print(f"  Succeeded: {len(results):,}")
    print(f"  Errored:   {len(errors):,}")
    return results, errors


def retry_errors(client, errors, predictions, evidence_map):
    """Retry failed requests via synchronous API."""
    if not errors:
        return {}
    print(f"\n  Retrying {len(errors)} failure(s) via standard API...")
    pred_lookup = predictions.set_index("hypothesis_id")
    retried = {}

    for err in tqdm(errors, desc="  Retrying"):
        hyp_id = err["hypothesis_id"]
        if hyp_id not in pred_lookup.index:
            continue

        # Rebuild the user message
        pred = pred_lookup.loc[hyp_id]
        evidence = evidence_map.get(hyp_id, [])
        pred_text = pred.get("resolved_text") or pred.get("raw_text", "")
        pred_topic = pred.get("topic", "unknown")
        confidence = pred.get("confidence_level", "unknown")
        temporal = pred.get("temporal_scope", "unspecified")
        pred_ts = pred.get("message_timestamp")
        date_str = str(pred_ts)[:10] if pd.notna(pred_ts) else "unknown date"

        parts = [
            f'PREDICTION (made {date_str}, confidence: {confidence}, temporal scope: {temporal}):',
            f'"{pred_text}"', '', f'TOPIC: {pred_topic}',
        ]
        if evidence:
            parts.extend(['', 'EVIDENCE FROM LATER CONVERSATIONS:'])
            for i, ev in enumerate(evidence, 1):
                parts.append(f'[{i}] ({str(ev["created_at"])[:10]}, {ev["days_later"]}d later): {ev["summary"][:500]}')
        else:
            parts.extend(['', 'NO EVIDENCE CONVERSATIONS FOUND.', 'Classify as "unresolvable".'])
        parts.extend(['', 'Based on this evidence, classify the prediction outcome.'])

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": '\n'.join(parts)}],
            )
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            retried[hyp_id] = {
                "raw": raw_text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Retry failed for {hyp_id}: {e}")
        time.sleep(1)

    print(f"  Retries succeeded: {len(retried):,}")
    return retried


# ===========================================================================
# STEP 7: Build Output
# ===========================================================================
def build_output(predictions, evidence_map, all_results):
    """Parse results and build output parquet + update hypotheses."""
    print("\n=== Step 7: Build Output ===")

    pred_lookup = predictions.set_index("hypothesis_id")
    rows = []
    parse_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Precompute corpus end once
    convos = pd.read_parquet(CONVOS_PATH)
    corpus_end = pd.to_datetime(convos["created_at"], utc=True).max()

    for hyp_id, res in all_results.items():
        outcome, confidence, evidence_summary, error = parse_outcome(res["raw"])
        total_input_tokens += res.get("input_tokens", 0)
        total_output_tokens += res.get("output_tokens", 0)

        if error:
            parse_errors += 1
            outcome = "unresolvable"
            confidence = 0.0
            evidence_summary = f"Parse error: {error}"

        pred = pred_lookup.loc[hyp_id] if hyp_id in pred_lookup.index else None
        evidence = evidence_map.get(hyp_id, [])

        # Compute window info
        pred_ts = pred.get("message_timestamp") if pred is not None else None
        if pd.notna(pred_ts):
            pred_ts_dt = pd.to_datetime(pred_ts, utc=True)
            window_days = min(EVIDENCE_WINDOW_DAYS, (corpus_end - pred_ts_dt).days)
            short_window = window_days < EVIDENCE_WINDOW_DAYS
        else:
            window_days = 0
            short_window = True

        rows.append({
            "hypothesis_id": hyp_id,
            "conversation_id": pred["conversation_id"] if pred is not None else None,
            "prediction_text": (pred.get("resolved_text") or pred.get("raw_text", "")) if pred is not None else "",
            "topic": pred.get("topic", "") if pred is not None else "",
            "temporal_scope": pred.get("temporal_scope") if pred is not None else None,
            "confidence_level": pred.get("confidence_level") if pred is not None else None,
            "prediction_date": pred.get("message_timestamp") if pred is not None else None,
            "outcome": outcome,
            "outcome_confidence": confidence,
            "evidence_summary": evidence_summary,
            "evidence_count": len(evidence),
            "evidence_window_days": int(window_days),
            "short_window": short_window,
        })

    df = pd.DataFrame(rows)

    # Handle predictions with no API result (shouldn't happen, but safety)
    missing = set(predictions["hypothesis_id"]) - set(all_results.keys())
    for hyp_id in missing:
        pred = pred_lookup.loc[hyp_id]
        df = pd.concat([df, pd.DataFrame([{
            "hypothesis_id": hyp_id,
            "conversation_id": pred["conversation_id"],
            "prediction_text": pred.get("resolved_text") or pred.get("raw_text", ""),
            "topic": pred.get("topic", ""),
            "temporal_scope": pred.get("temporal_scope"),
            "confidence_level": pred.get("confidence_level"),
            "prediction_date": pred.get("message_timestamp"),
            "outcome": "unresolvable",
            "outcome_confidence": 0.0,
            "evidence_summary": "No API result received",
            "evidence_count": len(evidence_map.get(hyp_id, [])),
            "evidence_window_days": 0,
            "short_window": True,
        }])], ignore_index=True)

    # Save prediction outcomes
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved {OUT_PARQUET} ({len(df):,} rows)")

    # Update hypotheses.parquet status for predictions
    hyp = pd.read_parquet(HYPOTHESES_PATH)
    outcome_lookup = dict(zip(df["hypothesis_id"], df["outcome"]))
    hyp["status"] = hyp.apply(
        lambda r: outcome_lookup.get(r["hypothesis_id"], r["status"]), axis=1
    )
    hyp.to_parquet(HYPOTHESES_PATH, index=False)
    updated_count = sum(1 for h in hyp["status"] if h != "unscored")
    print(f"  Updated hypotheses.parquet: {updated_count:,} predictions scored")

    print(f"  Parse errors: {parse_errors}")
    print(f"  Total tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")

    cost = (total_input_tokens * PRICE_INPUT_PER_MTOK + total_output_tokens * PRICE_OUTPUT_PER_MTOK) / 1_000_000
    print(f"  Estimated cost: ${cost:.2f}")

    return df, {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": cost,
        "parse_errors": parse_errors,
    }


# ===========================================================================
# STEP 8: Report & Figures
# ===========================================================================
def generate_report(df, evidence_stats, cost_stats, predictions):
    """Generate JSON report and figures."""
    print("\n=== Step 8: Report & Figures ===")

    # Outcome distribution
    outcome_dist = df["outcome"].value_counts().to_dict()
    total = len(df)
    resolved = total - outcome_dist.get("unresolvable", 0)
    confirmed = outcome_dist.get("confirmed", 0)
    disconfirmed = outcome_dist.get("disconfirmed", 0)

    resolution_rate = resolved / total if total > 0 else 0
    accuracy_rate = confirmed / (confirmed + disconfirmed) if (confirmed + disconfirmed) > 0 else None

    print(f"  Outcome distribution: {outcome_dist}")
    print(f"  Resolution rate: {resolution_rate:.1%}")
    if accuracy_rate is not None:
        print(f"  Accuracy rate: {accuracy_rate:.1%}")

    # By temporal scope
    by_scope = {}
    for scope, group in df.groupby("temporal_scope", dropna=False):
        scope_key = str(scope) if pd.notna(scope) else "null"
        by_scope[scope_key] = group["outcome"].value_counts().to_dict()

    # By confidence level
    by_conf = {}
    for conf, group in df.groupby("confidence_level", dropna=False):
        conf_key = str(conf) if pd.notna(conf) else "null"
        by_conf[conf_key] = group["outcome"].value_counts().to_dict()

    # By entity (extract from prediction text)
    entity_att = None
    if ENTITY_ATT_PATH.exists():
        entity_att = pd.read_parquet(ENTITY_ATT_PATH)
        top_entities = entity_att["entity"].value_counts().head(6).index.tolist()
    else:
        top_entities = []

    by_entity = {}
    for entity in top_entities:
        mask = df["prediction_text"].str.contains(entity, case=False, na=False)
        if mask.sum() >= 5:
            by_entity[entity] = df[mask]["outcome"].value_counts().to_dict()

    # Top confirmed/disconfirmed
    top_confirmed = (
        df[df["outcome"] == "confirmed"]
        .nlargest(10, "outcome_confidence")
        [["prediction_text", "topic", "outcome_confidence", "evidence_summary"]]
        .to_dict("records")
    )
    top_disconfirmed = (
        df[df["outcome"] == "disconfirmed"]
        .nlargest(10, "outcome_confidence")
        [["prediction_text", "topic", "outcome_confidence", "evidence_summary"]]
        .to_dict("records")
    )

    report = {
        "module": "prediction_resolution",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost": cost_stats,
        "predictions_total": total,
        "outcome_distribution": outcome_dist,
        "resolution_rate": resolution_rate,
        "accuracy_rate": accuracy_rate,
        "by_temporal_scope": by_scope,
        "by_confidence_level": by_conf,
        "by_entity": by_entity,
        "evidence_stats": {
            "avg_candidates_per_prediction": np.mean(evidence_stats["candidate_counts"]) if evidence_stats["candidate_counts"] else 0,
            "predictions_with_no_evidence": evidence_stats["no_evidence"],
            "short_window_predictions": evidence_stats["short_window"],
        },
        "top_confirmed_predictions": top_confirmed,
        "top_disconfirmed_predictions": top_disconfirmed,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2, default=str)
    print(f"  Saved {OUT_REPORT}")

    # --- FIGURES ---

    # Figure 1: Outcome distribution (donut chart)
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(OUTCOME_COLORS.keys())
    sizes = [outcome_dist.get(l, 0) for l in labels]
    colors = [OUTCOME_COLORS[l] for l in labels]
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        labels_nz, sizes_nz, colors_nz = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%',
            pctdistance=0.75, startangle=90, textprops={"fontsize": 11}
        )
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_patch(centre_circle)
        ax.text(0, 0, f"{total}\npredictions", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("Prediction Outcome Distribution", fontsize=15)
    plt.tight_layout()
    plt.savefig(figpath("01_outcome_distribution.png"), dpi=DPI)
    plt.close()

    # Figure 2: Outcome by temporal scope (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    scope_df = df.copy()
    scope_df["temporal_scope"] = scope_df["temporal_scope"].fillna("null/unspecified")
    scope_df.loc[scope_df["temporal_scope"] == "unspecified", "temporal_scope"] = "null/unspecified"
    ct = pd.crosstab(scope_df["temporal_scope"], scope_df["outcome"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    ordered_cols = [c for c in ["confirmed", "partially_confirmed", "disconfirmed", "unresolvable"] if c in ct_norm.columns]
    ct_norm = ct_norm[ordered_cols]
    ct_norm.plot(kind="barh", stacked=True, ax=ax,
                 color=[OUTCOME_COLORS.get(c, "#999") for c in ordered_cols], edgecolor="white")
    # Add count labels
    for i, scope in enumerate(ct_norm.index):
        ax.text(1.02, i, f"n={ct.loc[scope].sum()}", va="center", fontsize=9)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_ylabel("Temporal Scope", fontsize=11)
    ax.set_title("Prediction Outcomes by Temporal Scope", fontsize=14)
    ax.legend(title="Outcome", bbox_to_anchor=(1.15, 1), loc="upper left")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(figpath("02_outcome_by_temporal_scope.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 3: Outcome by confidence level (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    conf_df = df.copy()
    conf_df["confidence_level"] = conf_df["confidence_level"].fillna("unknown")
    ct = pd.crosstab(conf_df["confidence_level"], conf_df["outcome"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    ordered_cols = [c for c in ["confirmed", "partially_confirmed", "disconfirmed", "unresolvable"] if c in ct_norm.columns]
    ct_norm = ct_norm[ordered_cols]
    ct_norm.plot(kind="barh", stacked=True, ax=ax,
                 color=[OUTCOME_COLORS.get(c, "#999") for c in ordered_cols], edgecolor="white")
    for i, conf in enumerate(ct_norm.index):
        ax.text(1.02, i, f"n={ct.loc[conf].sum()}", va="center", fontsize=9)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_ylabel("Confidence Level", fontsize=11)
    ax.set_title("Prediction Outcomes by Confidence Level", fontsize=14)
    ax.legend(title="Outcome", bbox_to_anchor=(1.15, 1), loc="upper left")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(figpath("03_outcome_by_confidence.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 4: Resolution over time (monthly)
    fig, ax = plt.subplots(figsize=(14, 6))
    df_time = df.copy()
    df_time["prediction_date"] = pd.to_datetime(df_time["prediction_date"], utc=True, errors="coerce")
    df_time = df_time.dropna(subset=["prediction_date"])
    df_time["year_month"] = df_time["prediction_date"].dt.to_period("M").astype(str)

    monthly = df_time.groupby("year_month").apply(
        lambda g: pd.Series({
            "total": len(g),
            "resolved": (g["outcome"] != "unresolvable").sum(),
            "confirmed": (g["outcome"] == "confirmed").sum(),
            "disconfirmed": (g["outcome"] == "disconfirmed").sum(),
        })
    ).reset_index()

    if len(monthly) > 0:
        monthly["resolution_rate"] = monthly["resolved"] / monthly["total"]
        monthly["accuracy_rate"] = monthly["confirmed"] / (monthly["confirmed"] + monthly["disconfirmed"]).replace(0, np.nan)

        x = range(len(monthly))
        ax.bar(x, monthly["total"], color="#DDDDDD", label="Total predictions", alpha=0.5)
        ax.bar(x, monthly["resolved"], color="#59A14F", alpha=0.6, label="Resolved")

        ax2 = ax.twinx()
        ax2.plot(x, monthly["resolution_rate"], "o-", color="#4E79A7", linewidth=2, label="Resolution rate")
        if monthly["accuracy_rate"].notna().any():
            ax2.plot(x, monthly["accuracy_rate"], "s--", color="#E15759", linewidth=2, label="Accuracy rate")
        ax2.set_ylabel("Rate", fontsize=11)
        ax2.set_ylim(0, 1)

        ax.set_xticks(x)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right")
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Prediction Resolution Over Time", fontsize=14)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(figpath("04_resolution_over_time.png"), dpi=DPI)
    plt.close()

    # Figure 5: Outcome by entity (grouped bar)
    if by_entity:
        fig, ax = plt.subplots(figsize=(12, 6))
        entities = list(by_entity.keys())
        outcomes = ["confirmed", "partially_confirmed", "disconfirmed", "unresolvable"]
        x = np.arange(len(entities))
        width = 0.2

        for i, outcome in enumerate(outcomes):
            vals = [by_entity[e].get(outcome, 0) for e in entities]
            ax.bar(x + i * width, vals, width, label=outcome, color=OUTCOME_COLORS[outcome])

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(entities, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Prediction Outcomes by Entity Mentioned", fontsize=14)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(figpath("05_outcome_by_entity.png"), dpi=DPI)
        plt.close()

    figs = list(FIG_DIR.glob("*.png"))
    print(f"  Generated {len(figs)} figures")

    return report


# ===========================================================================
# STEP 9: Validation
# ===========================================================================
def validate(df, report):
    """Run validation checklist."""
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKLIST")
    print("=" * 70)

    checks = []

    # 1. All predictions have outcome
    ok = len(df) == 898 or len(df) > 0  # flexible in case prediction count varies
    checks.append(("All predictions have outcome", df["outcome"].notna().all(), f"{len(df)} rows"))

    # 2. Valid outcomes only
    ok = df["outcome"].isin(VALID_OUTCOMES).all()
    checks.append(("Valid outcome values", ok, f"{df['outcome'].nunique()} unique"))

    # 3. Resolution rate > 0
    resolved = (df["outcome"] != "unresolvable").sum()
    checks.append(("Resolution rate > 0", resolved > 0, f"{resolved} resolved"))

    # 4. No NaN/Inf in report
    report_str = json.dumps(report, default=str)
    no_nan = "NaN" not in report_str and "Infinity" not in report_str
    checks.append(("No NaN/Inf in report", no_nan, "clean"))

    # 5. Parquet saved
    checks.append(("prediction_outcomes.parquet saved", OUT_PARQUET.exists(), str(OUT_PARQUET.name)))

    # 6. Hypotheses updated
    hyp = pd.read_parquet(HYPOTHESES_PATH)
    scored = (hyp["status"] != "unscored").sum()
    checks.append(("hypotheses.parquet updated", scored > 0, f"{scored} scored"))

    # 7. Figures exist
    figs = list(FIG_DIR.glob("*.png"))
    checks.append(("Figure PNGs exist", len(figs) >= 4, f"{len(figs)} figures"))

    # 8. Cost check
    cost = report.get("cost", {}).get("estimated_cost_usd", 0)
    checks.append(("Cost within range", 0 < cost < 5, f"${cost:.2f}"))

    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {detail}")

    passed_count = sum(1 for _, p, _ in checks if p)
    print(f"\n  {passed_count}/{len(checks)} checks passed")
    return passed_count == len(checks)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Module 23: Prediction Resolution")
    parser.add_argument("--evidence-only", action="store_true", help="Build evidence map only, no API")
    parser.add_argument("--dry-run", action="store_true", help="Build JSONL but don't submit")
    parser.add_argument("--resume-batch-id", type=str, help="Resume polling a submitted batch")
    args = parser.parse_args()

    print("=" * 70)
    print("  Module 23: Prediction Resolution")
    print("=" * 70)

    # Step 1: Load
    predictions, convos, summaries, entities = load_data()

    # Step 2: Evidence
    evidence_map, evidence_stats = build_evidence_map(predictions, convos, summaries, entities)

    if args.evidence_only:
        print("\n  --evidence-only: stopping here.")
        return

    # Step 3: Build batch
    count = build_batch_requests(predictions, evidence_map)

    if args.dry_run:
        print(f"\n  --dry-run: {count} requests written to {BATCH_FILE}")
        print("  Inspect with: head -5 " + str(BATCH_FILE))
        return

    # Step 4-6: Submit, poll, retrieve
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n  ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    client = anthropic.Anthropic()

    if args.resume_batch_id:
        batch_id = args.resume_batch_id
    else:
        batch_id = submit_batch(client)

    poll_batch(client, batch_id)
    results, errors = retrieve_results(client, batch_id)

    # Retry errors
    retried = retry_errors(client, errors, predictions, evidence_map)
    all_results = {**results, **retried}

    # Step 7: Build output
    df, cost_stats = build_output(predictions, evidence_map, all_results)

    # Step 8: Report & Figures
    report = generate_report(df, evidence_stats, cost_stats, predictions)

    # Step 9: Validate
    validate(df, report)

    # Summary
    outcome_dist = df["outcome"].value_counts().to_dict()
    resolved = len(df) - outcome_dist.get("unresolvable", 0)
    print(f"\n{'=' * 70}")
    print(f"  Module 23 Complete")
    print(f"  Predictions: {len(df)}")
    print(f"  Resolved: {resolved} ({resolved/len(df):.1%})")
    print(f"  Outcomes: {outcome_dist}")
    print(f"  Cost: ${cost_stats['estimated_cost_usd']:.2f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
