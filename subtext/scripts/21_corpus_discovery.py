"""
Module: Corpus Discovery
Script: 21_corpus_discovery.py

Sends every analysable conversation to the Claude Batch API (Sonnet) and asks
an open-ended question: what is interesting, unusual, or measurable here?
No prior enrichment data is referenced — the model sees only raw conversation
text and knows it is one of ~1,500 from the same user over 11 months.

The output is a dataset of research questions and observations (1-3 per
conversation, ~2,000-3,000 total) that can later be clustered to discover
what the corpus contains that we haven't thought to look for yet.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/21_corpus_discovery.py

    # Dry run (build JSONL, print stats, don't submit):
    python scripts/21_corpus_discovery.py --dry-run

    # Resume a previously submitted batch:
    python scripts/21_corpus_discovery.py --resume-batch-id msgbatch_xxxxx
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

import pandas as pd
import tiktoken
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MSGS_PATH       = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
INTERIM_DIR     = os.path.join(BASE, "data", "interim")
BATCH_FILE      = os.path.join(INTERIM_DIR, "discovery_batch_requests.jsonl")
PROMPTS_DIR     = os.path.join(BASE, "prompts")
SYSTEM_PROMPT_F = os.path.join(PROMPTS_DIR, "discovery_system.txt")

OUT_PARQUET     = os.path.join(BASE, "data", "processed", "corpus_discovery.parquet")
OUT_SAMPLE_CSV  = os.path.join(BASE, "data", "processed", "corpus_discovery_sample.csv")
OUT_REPORT      = os.path.join(BASE, "outputs", "reports", "corpus_discovery_report.json")

MODEL = "claude-sonnet-4-20250514"
MAX_INPUT_TOKENS = 150_000
MAX_OUTPUT_TOKENS = 800

# Batch API pricing (Sonnet, 50% batch discount)
PRICE_INPUT_PER_MTOK  = 1.50   # $3.00 * 0.5
PRICE_OUTPUT_PER_MTOK = 7.50   # $15.00 * 0.5

JSON_FORMAT_INSTRUCTION = (
    '\n\n---\n\n'
    'Respond with a JSON object in this exact format:\n'
    '{"items": [{"type": "question"|"observation", "text": "...", '
    '"evidence": "...", "confidence": "high"|"medium"|"low"}]}\n\n'
    'Return 1-3 items. No other text, only the JSON object.'
)

VALID_TYPES = {"question", "observation"}
VALID_CONFIDENCES = {"high", "medium", "low"}


# ── Step 0: Environment Setup ────────────────────────────────────────────────
def check_api_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print(f"API key found: {api_key[:12]}...{api_key[-4:]}")
    return api_key


# ── Step 1: Build Conversation Texts ─────────────────────────────────────────
def load_data():
    print("Loading data...")
    for path, label in [(MSGS_PATH, "messages"), (CONV_PATH, "conversations")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found at {path}")
            sys.exit(1)

    conversations = pd.read_parquet(CONV_PATH)
    messages = pd.read_parquet(MSGS_PATH)

    conv = conversations[conversations["is_analysable"]].copy()
    msgs = messages[messages["conversation_id"].isin(conv["conversation_id"])].copy()

    print(f"  Total conversations: {len(conversations):,}")
    print(f"  Analysable conversations: {len(conv):,}")
    print(f"  Messages for analysable: {len(msgs):,}")
    return conv, msgs


def build_conversation_text(conv_id, msgs_df):
    """Reconstruct a conversation as a readable User/Assistant dialogue."""
    conv_msgs = msgs_df[msgs_df["conversation_id"] == conv_id].sort_values("msg_index")

    parts = []
    for _, msg in conv_msgs.iterrows():
        if msg["role"] not in ("user", "assistant"):
            continue
        text = msg["text"]
        if pd.isna(text) or str(text).strip() == "":
            continue
        role_label = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role_label}: {str(text).strip()}")

    return "\n\n".join(parts)


def build_all_texts(conv, msgs):
    print("Building conversation texts...")
    enc = tiktoken.get_encoding("cl100k_base")

    conversation_texts = {}
    token_counts = {}

    for conv_id in tqdm(conv["conversation_id"], desc="Building texts"):
        text = build_conversation_text(conv_id, msgs)
        conversation_texts[conv_id] = text
        try:
            token_counts[conv_id] = len(enc.encode(text))
        except Exception:
            token_counts[conv_id] = len(text.split())

    token_series = pd.Series(token_counts)
    estimated_input_cost = token_series.sum() / 1_000_000 * PRICE_INPUT_PER_MTOK
    estimated_output_cost = len(conversation_texts) * 400 / 1_000_000 * PRICE_OUTPUT_PER_MTOK
    estimated_cost = estimated_input_cost + estimated_output_cost

    print(f"  Total conversations: {len(conversation_texts):,}")
    print(f"  Total input tokens: {token_series.sum():,}")
    print(f"  Mean tokens/conversation: {token_series.mean():,.0f}")
    print(f"  Median tokens/conversation: {token_series.median():,.0f}")
    print(f"  Max tokens/conversation: {token_series.max():,}")
    print(f"  Estimated API cost (Sonnet batch): ${estimated_cost:.2f}")
    print(f"    Input:  ${estimated_input_cost:.2f}")
    print(f"    Output: ${estimated_output_cost:.2f} (est. ~400 tokens/response)")

    return conversation_texts, token_counts, enc


def truncate_long_conversations(conversation_texts, token_counts, enc):
    """Truncate very long conversations while preserving beginning and end."""
    truncated_count = 0

    for conv_id in list(conversation_texts.keys()):
        tc = token_counts[conv_id]
        if tc > MAX_INPUT_TOKENS:
            tokens = enc.encode(conversation_texts[conv_id])
            first_chunk = enc.decode(tokens[:100_000])
            last_chunk = enc.decode(tokens[-50_000:])
            omitted = tc - 150_000

            conversation_texts[conv_id] = (
                first_chunk
                + f"\n\n[... middle of conversation omitted for length "
                  f"— {omitted:,} tokens removed ...]\n\n"
                + last_chunk
            )
            truncated_count += 1

    print(f"  Truncated conversations: {truncated_count:,}")
    return truncated_count


# ── Step 2: System Prompt ────────────────────────────────────────────────────
def load_system_prompt():
    if not os.path.exists(SYSTEM_PROMPT_F):
        print(f"ERROR: System prompt not found at {SYSTEM_PROMPT_F}")
        sys.exit(1)
    with open(SYSTEM_PROMPT_F, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    print(f"  System prompt loaded ({len(prompt)} chars)")
    return prompt


# ── Step 3: Batch API ────────────────────────────────────────────────────────
def make_batch_request(conv_id, conversation_text, system_prompt):
    """Create a single batch request object."""
    user_content = conversation_text + JSON_FORMAT_INSTRUCTION

    return {
        "custom_id": conv_id,
        "params": {
            "model": MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    }


def write_batch_file(conversation_texts, system_prompt):
    """Write all requests to a JSONL file."""
    os.makedirs(INTERIM_DIR, exist_ok=True)

    print("Writing batch request file...")
    with open(BATCH_FILE, "w", encoding="utf-8") as f:
        for conv_id, text in conversation_texts.items():
            request = make_batch_request(conv_id, text, system_prompt)
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    line_count = sum(1 for _ in open(BATCH_FILE, encoding="utf-8"))
    print(f"  Batch request file: {BATCH_FILE}")
    print(f"  Total requests: {line_count:,}")
    return line_count


def submit_batch(client):
    """Submit batch to Anthropic Batch API."""
    print("Submitting batch to Anthropic API...")

    requests = []
    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            requests.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests)

    print(f"  ============================================")
    print(f"  BATCH SUBMITTED: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ============================================")
    print(f"  Save this batch ID in case you need to resume!")
    return batch.id


def poll_batch(client, batch_id):
    """Poll for batch completion, printing status each check."""
    print(f"\nPolling batch {batch_id} for completion...")
    print(f"  (Checking every 60 seconds. This may take 1-6 hours.)\n")

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


def retrieve_results(client, batch_id):
    """Retrieve results from completed batch."""
    print(f"Retrieving results for batch {batch_id}...")
    results = {}
    errors = []

    for result in client.messages.batches.results(batch_id):
        conv_id = result.custom_id

        if result.result.type == "succeeded":
            response = result.result.message
            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text += block.text

            results[conv_id] = {
                "raw": raw_text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            errors.append({
                "conversation_id": conv_id,
                "error_type": result.result.type,
                "error": str(result.result.error)
                         if hasattr(result.result, "error") else "unknown",
            })

    print(f"  Succeeded: {len(results):,}")
    print(f"  Errored: {len(errors):,}")
    return results, errors


# ── Step 4: Error Handling & Retries ─────────────────────────────────────────
def retry_errors(client, errors, conversation_texts, system_prompt):
    """Retry failed requests individually via standard API."""
    if not errors:
        return {}

    print(f"\nRetrying {len(errors)} failed requests via standard API...")
    retried = {}

    for error_item in tqdm(errors, desc="Retrying"):
        conv_id = error_item["conversation_id"]
        text = conversation_texts.get(conv_id)
        if not text:
            continue

        try:
            user_content = text + JSON_FORMAT_INSTRUCTION
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )

            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text += block.text

            retried[conv_id] = {
                "raw": raw_text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Retry failed for {conv_id}: {e}")

        time.sleep(1)  # Rate limiting

    print(f"  Retries succeeded: {len(retried):,}")
    return retried


def fill_placeholders(results, conversation_texts):
    """Fill placeholder for any conversations that still have no results."""
    filled = 0
    for conv_id in conversation_texts:
        if conv_id not in results:
            results[conv_id] = {
                "raw": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }
            filled += 1
    if filled:
        print(f"  Filled {filled} placeholders for permanently failed conversations")
    return results


# ── Step 5: Parse Responses ──────────────────────────────────────────────────
def _strip_json_fences(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    return text.strip()


def parse_discovery_response(raw_text):
    """Parse JSON response into validated items.

    Returns (items_list, error_msg).
    items_list is a list of dicts with keys: type, text, evidence, confidence.
    On failure, returns ([], error_message).
    """
    if not raw_text:
        return [], "empty response"

    try:
        text = _strip_json_fences(raw_text)
        parsed = json.loads(text)

        # Handle both {"items": [...]} and bare [...]
        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict):
            raw_items = parsed.get("items", [])
        else:
            return [], f"unexpected JSON type: {type(parsed).__name__}"

        if not isinstance(raw_items, list) or len(raw_items) == 0:
            return [], "empty or missing items array"

        validated = []
        for item in raw_items[:3]:  # Cap at 3 items
            item_type = str(item.get("type", "observation")).lower().strip()
            if item_type not in VALID_TYPES:
                item_type = "observation"

            text_val = str(item.get("text", "")).strip()
            evidence_val = str(item.get("evidence", "")).strip()
            confidence = str(item.get("confidence", "medium")).lower().strip()
            if confidence not in VALID_CONFIDENCES:
                confidence = "medium"

            if not text_val:
                continue

            validated.append({
                "type": item_type,
                "text": text_val,
                "evidence": evidence_val,
                "confidence": confidence,
            })

        if not validated:
            return [], "all items empty after validation"

        return validated, None
    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}"
    except Exception as e:
        return [], f"parse error: {e}"


def build_output_rows(results):
    """Explode per-conversation results into per-item rows."""
    rows = []
    parse_errors = 0
    conversations_with_items = 0

    for conv_id, res in results.items():
        items, err = parse_discovery_response(res.get("raw", ""))

        if err:
            parse_errors += 1
            rows.append({
                "conversation_id": conv_id,
                "item_index": 0,
                "type": "error",
                "text": f"[PARSE FAILED: {err}]",
                "evidence": "",
                "confidence": "low",
                "input_tokens": res.get("input_tokens", 0),
                "output_tokens": res.get("output_tokens", 0),
            })
            continue

        conversations_with_items += 1
        for idx, item in enumerate(items):
            rows.append({
                "conversation_id": conv_id,
                "item_index": idx,
                "type": item["type"],
                "text": item["text"],
                "evidence": item["evidence"],
                "confidence": item["confidence"],
                "input_tokens": res.get("input_tokens", 0) if idx == 0 else 0,
                "output_tokens": res.get("output_tokens", 0) if idx == 0 else 0,
            })

    print(f"  Conversations with items: {conversations_with_items:,}")
    print(f"  Parse errors: {parse_errors:,}")
    print(f"  Total suggestion rows: {len(rows):,}")
    return rows, parse_errors


# ── Step 6: Save Results ─────────────────────────────────────────────────────
def save_results(rows):
    """Build and save the output parquet + CSV sample."""
    print("\nSaving results...")

    df = pd.DataFrame(rows)
    df["item_index"] = df["item_index"].astype("int32")
    df["input_tokens"] = df["input_tokens"].astype("int32")
    df["output_tokens"] = df["output_tokens"].astype("int32")
    df["type"] = df["type"].astype("category")
    df["confidence"] = df["confidence"].astype("category")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET} ({len(df):,} rows)")

    # CSV sample for inspection
    sample = df.head(200)
    sample.to_csv(OUT_SAMPLE_CSV, index=False)
    print(f"  Saved sample: {OUT_SAMPLE_CSV}")

    # Print 10 random good items
    good = df[df["type"] != "error"]
    if len(good) > 0:
        n = min(10, len(good))
        sample_items = good.sample(n=n, random_state=42)

        print("\n" + "=" * 80)
        print("SAMPLE DISCOVERY ITEMS (10 random)")
        print("=" * 80)

        for _, row in sample_items.iterrows():
            print(f"\n  [{row['type'].upper()}] (confidence: {row['confidence']})")
            print(f"  {row['text']}")
            evidence_preview = row['evidence'][:150]
            if len(row['evidence']) > 150:
                evidence_preview += "..."
            print(f"  Evidence: {evidence_preview}")

        print("\n" + "=" * 80)

    return df


# ── Step 7: Generate Report ──────────────────────────────────────────────────
def generate_report(df, results, errors, batch_id,
                    conversation_texts, truncated_count, parse_errors):
    """Write corpus_discovery_report.json."""
    print("\nGenerating report...")

    total_input = int(df["input_tokens"].sum())
    total_output = int(df["output_tokens"].sum())

    input_cost = total_input / 1_000_000 * PRICE_INPUT_PER_MTOK
    output_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_MTOK
    total_cost = input_cost + output_cost

    good = df[df["type"] != "error"]
    n_conversations_succeeded = len(results) - parse_errors

    report = {
        "module": "corpus_discovery",
        "module_version": "1.0",
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": MODEL,
        "batch_id": batch_id,
        "input_data": {
            "conversations_attempted": len(conversation_texts),
            "conversations_succeeded": n_conversations_succeeded,
            "conversations_failed": len(conversation_texts) - len(results),
            "conversations_truncated": truncated_count,
            "conversations_with_parse_errors": parse_errors,
        },
        "token_usage": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "mean_input_tokens_per_conversation": round(
                total_input / max(len(results), 1), 0
            ),
            "mean_output_tokens_per_conversation": round(
                total_output / max(len(results), 1), 0
            ),
        },
        "cost": {
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "pricing_note": (
                f"Sonnet Batch API: ${PRICE_INPUT_PER_MTOK}/MTok input, "
                f"${PRICE_OUTPUT_PER_MTOK}/MTok output (50% discount)"
            ),
        },
        "discovery_statistics": {
            "total_items": len(good),
            "items_per_conversation_mean": round(
                len(good) / max(n_conversations_succeeded, 1), 2
            ),
            "type_distribution": good["type"].value_counts().to_dict()
                                 if len(good) > 0 else {},
            "confidence_distribution": good["confidence"].value_counts().to_dict()
                                       if len(good) > 0 else {},
            "mean_text_length_chars": round(good["text"].str.len().mean(), 0)
                                      if len(good) > 0 else 0,
            "mean_evidence_length_chars": round(good["evidence"].str.len().mean(), 0)
                                          if len(good) > 0 else 0,
        },
        "output_files": [
            "data/processed/corpus_discovery.parquet",
            "data/processed/corpus_discovery_sample.csv",
        ],
    }

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.2f}")
    return report


# ── Validation ───────────────────────────────────────────────────────────────
def run_validation(df, conv, conversation_texts, parse_errors):
    """Run validation checklist and print PASS/FAIL for each check."""
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    all_pass = True

    # 1. Parquet exists and loads
    try:
        test_df = pd.read_parquet(OUT_PARQUET)
        print("  [PASS] corpus_discovery.parquet exists and loads")
    except Exception as e:
        print(f"  [FAIL] corpus_discovery.parquet: {e}")
        all_pass = False

    # 2. Total items between 1x and 3x conversations
    n_conv = len(conversation_texts)
    n_items = len(df[df["type"] != "error"])
    if n_conv <= n_items <= n_conv * 3:
        print(f"  [PASS] Total items: {n_items:,} (in range {n_conv:,}–{n_conv * 3:,})")
    else:
        print(f"  [WARN] Total items: {n_items:,} (expected range {n_conv:,}–{n_conv * 3:,})")
        all_pass = False

    # 3. All conversation_ids exist in conversations_clean
    valid_ids = set(conv["conversation_id"])
    result_ids = set(df["conversation_id"])
    orphans = result_ids - valid_ids
    if len(orphans) == 0:
        print(f"  [PASS] All conversation_ids exist in conversations_clean")
    else:
        print(f"  [FAIL] {len(orphans)} orphan conversation_ids")
        all_pass = False

    # 4. Unique conversation_ids cover >= 99% of input
    coverage = len(result_ids & valid_ids) / max(n_conv, 1) * 100
    if coverage >= 99:
        print(f"  [PASS] Conversation coverage: {coverage:.1f}%")
    else:
        print(f"  [FAIL] Conversation coverage: {coverage:.1f}% (below 99%)")
        all_pass = False

    # 5. Parse error rate < 5%
    parse_pct = parse_errors / max(n_conv, 1) * 100
    if parse_pct < 5:
        print(f"  [PASS] Parse error rate: {parse_pct:.1f}% ({parse_errors:,} errors)")
    else:
        print(f"  [FAIL] Parse error rate: {parse_pct:.1f}% ({parse_errors:,} errors) — exceeds 5%")
        all_pass = False

    # 6. Type values are valid
    good = df[df["type"] != "error"]
    invalid_types = set(good["type"].unique()) - VALID_TYPES
    if len(invalid_types) == 0:
        print(f"  [PASS] All type values are valid")
    else:
        print(f"  [FAIL] Invalid type values: {invalid_types}")
        all_pass = False

    # 7. Confidence values are valid
    invalid_conf = set(good["confidence"].unique()) - VALID_CONFIDENCES
    if len(invalid_conf) == 0:
        print(f"  [PASS] All confidence values are valid")
    else:
        print(f"  [FAIL] Invalid confidence values: {invalid_conf}")
        all_pass = False

    # 8. Mean text length sanity
    if len(good) > 0:
        mean_len = good["text"].str.len().mean()
        if 50 <= mean_len <= 500:
            print(f"  [PASS] Mean text length: {mean_len:.0f} chars (in 50–500 range)")
        else:
            print(f"  [WARN] Mean text length: {mean_len:.0f} chars (outside 50–500 range)")
            all_pass = False

    # 9. Report exists with cost section
    if os.path.exists(OUT_REPORT):
        with open(OUT_REPORT) as f:
            rpt = json.load(f)
        if "cost" in rpt:
            print(f"  [PASS] corpus_discovery_report.json exists with cost section")
            print(f"  [INFO] Total cost: ${rpt['cost']['total_cost_usd']:.2f}")
        else:
            print(f"  [FAIL] corpus_discovery_report.json missing cost section")
            all_pass = False
    else:
        print(f"  [FAIL] corpus_discovery_report.json not found")
        all_pass = False

    # 10. Items per conversation average
    if len(good) > 0:
        n_conv_with_items = good["conversation_id"].nunique()
        items_per = len(good) / max(n_conv_with_items, 1)
        if 1.0 <= items_per <= 3.0:
            print(f"  [PASS] Items per conversation: {items_per:.2f} (in 1.0–3.0 range)")
        else:
            print(f"  [WARN] Items per conversation: {items_per:.2f} (outside 1.0–3.0 range)")
            all_pass = False

    print("=" * 80)
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review above")
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Module: Corpus Discovery via Claude Batch API"
    )
    parser.add_argument(
        "--resume-batch-id", default=None,
        help="Resume from a previously submitted batch ID (skip submission)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build batch JSONL and print stats, but don't submit"
    )
    args = parser.parse_args()

    # Step 0: Check API key
    check_api_key()
    import anthropic
    client = anthropic.Anthropic()

    # Step 1: Load data and build texts
    conv, msgs = load_data()
    conversation_texts, token_counts, enc = build_all_texts(conv, msgs)
    truncated_count = truncate_long_conversations(
        conversation_texts, token_counts, enc
    )

    # Step 2: Load system prompt
    system_prompt = load_system_prompt()

    if args.dry_run:
        # Write batch file and exit
        write_batch_file(conversation_texts, system_prompt)
        print("\n  DRY RUN complete. Batch file written but not submitted.")
        print(f"  Review: {BATCH_FILE}")
        return

    if args.resume_batch_id:
        # Resume mode: skip submission, go straight to polling/retrieval
        batch_id = args.resume_batch_id
        print(f"\nResuming batch: {batch_id}")
    else:
        # Step 3a-3b: Write batch file and submit
        write_batch_file(conversation_texts, system_prompt)
        batch_id = submit_batch(client)

    # Step 3c: Poll for completion
    poll_batch(client, batch_id)

    # Step 3d: Retrieve results
    results, errors = retrieve_results(client, batch_id)

    # Step 4: Retry errors
    retried = retry_errors(client, errors, conversation_texts, system_prompt)
    results.update(retried)

    # Fill placeholders for anything still failed
    fill_placeholders(results, conversation_texts)

    # Step 5: Parse and explode into rows
    print("\nParsing discovery responses...")
    rows, parse_errors = build_output_rows(results)

    # Step 6: Save results
    df = save_results(rows)

    # Step 7: Generate report
    generate_report(
        df, results, errors, batch_id,
        conversation_texts, truncated_count, parse_errors
    )

    # Validation
    run_validation(df, conv, conversation_texts, parse_errors)

    print("\nDone!")


if __name__ == "__main__":
    main()
