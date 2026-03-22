"""
Module: Conversation Summarization
Script: 12_summarize.py

Sends every analysable conversation to the Claude Batch API (Sonnet 4.5) and
retrieves a concise summary. Summaries become the foundation for all downstream
thematic analysis — topic modeling, functional classification, recurring thread
detection, and the eventual cognitive self-model.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/12_summarize.py

    # To resume a previously submitted batch:
    python scripts/12_summarize.py --resume-batch-id msgbatch_xxxxx
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
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
BATCH_FILE      = os.path.join(INTERIM_DIR, "summarize_batch_requests.jsonl")
PROMPTS_DIR     = os.path.join(BASE, "prompts")
SYSTEM_PROMPT_F = os.path.join(PROMPTS_DIR, "summarize_system.txt")

OUT_PARQUET     = os.path.join(BASE, "data", "processed", "conversation_summaries.parquet")
OUT_SAMPLE_CSV  = os.path.join(BASE, "data", "processed", "conversation_summaries_sample.csv")
OUT_REPORT      = os.path.join(BASE, "outputs", "reports", "summarization_report.json")

MODEL = "claude-sonnet-4-20250514"
MAX_INPUT_TOKENS = 150_000


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
    estimated_cost = (
        token_series.sum() / 1_000_000 * 1.5
        + len(conversation_texts) * 75 / 1_000_000 * 7.5
    )

    print(f"  Total conversations: {len(conversation_texts):,}")
    print(f"  Total input tokens: {token_series.sum():,}")
    print(f"  Mean tokens/conversation: {token_series.mean():,.0f}")
    print(f"  Median tokens/conversation: {token_series.median():,.0f}")
    print(f"  Max tokens/conversation: {token_series.max():,}")
    print(f"  Estimated API cost (Sonnet batch): ${estimated_cost:.2f}")

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
    return {
        "custom_id": conv_id,
        "params": {
            "model": MODEL,
            "max_tokens": 300,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": conversation_text
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
            summary_text = ""
            for block in response.content:
                if block.type == "text":
                    summary_text += block.text

            results[conv_id] = {
                "summary": summary_text.strip(),
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
            response = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": text}]
            )

            summary_text = ""
            for block in response.content:
                if block.type == "text":
                    summary_text += block.text

            retried[conv_id] = {
                "summary": summary_text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Retry failed for {conv_id}: {e}")

        time.sleep(1)  # Rate limiting

    print(f"  Retries succeeded: {len(retried):,}")
    return retried


def fill_placeholders(results, conversation_texts):
    """Fill placeholder for any conversations that still couldn't be summarized."""
    filled = 0
    for conv_id in conversation_texts:
        if conv_id not in results:
            results[conv_id] = {
                "summary": "[SUMMARIZATION FAILED]",
                "input_tokens": 0,
                "output_tokens": 0,
            }
            filled += 1
    if filled:
        print(f"  Filled {filled} placeholders for permanently failed conversations")
    return results


# ── Step 5: Save Results ─────────────────────────────────────────────────────
def save_results(results, conv):
    """Build and save the output parquet + CSV sample."""
    print("\nSaving results...")

    summary_rows = []
    for conv_id, result in results.items():
        summary_rows.append({
            "conversation_id": conv_id,
            "summary": result["summary"],
            "summary_input_tokens": result["input_tokens"],
            "summary_output_tokens": result["output_tokens"],
        })

    summaries_df = pd.DataFrame(summary_rows)
    summaries_df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    print(f"  Rows: {len(summaries_df):,}")

    # CSV sample for inspection
    sample = summaries_df.head(200)
    sample.to_csv(OUT_SAMPLE_CSV, index=False)
    print(f"  Saved sample: {OUT_SAMPLE_CSV}")

    # Print 10 random summaries
    good_summaries = summaries_df[
        summaries_df["summary"] != "[SUMMARIZATION FAILED]"
    ]
    if len(good_summaries) > 0:
        n_sample = min(10, len(good_summaries))
        sample_convs = good_summaries.sample(n=n_sample, random_state=42)

        print("\n" + "=" * 80)
        print("SAMPLE SUMMARIES (10 random)")
        print("=" * 80)

        for _, row in sample_convs.iterrows():
            title = conv.loc[
                conv["conversation_id"] == row["conversation_id"], "title"
            ].values
            title_str = (
                title[0]
                if len(title) > 0 and pd.notna(title[0])
                else "[untitled]"
            )
            print(f"\n--- {title_str} ---")
            print(row["summary"])

        print("\n" + "=" * 80)

    return summaries_df


# ── Step 6: Generate Report ──────────────────────────────────────────────────
def generate_report(summaries_df, results, errors, batch_id,
                    conversation_texts, truncated_count):
    """Write summarization_report.json."""
    print("\nGenerating report...")

    total_input = summaries_df["summary_input_tokens"].sum()
    total_output = summaries_df["summary_output_tokens"].sum()

    # Batch API pricing for Sonnet 4.5
    input_cost = total_input / 1_000_000 * 1.5    # $3/MTok * 0.5 batch discount
    output_cost = total_output / 1_000_000 * 7.5   # $15/MTok * 0.5 batch discount
    total_cost = input_cost + output_cost

    succeeded = len([
        r for r in results.values() if r["summary"] != "[SUMMARIZATION FAILED]"
    ])
    failed = len([
        r for r in results.values() if r["summary"] == "[SUMMARIZATION FAILED]"
    ])

    report = {
        "module": "conversation_summarization",
        "module_version": "1.0",
        "generated_at": pd.Timestamp.now().isoformat(),
        "model": MODEL,
        "batch_id": batch_id,
        "input_data": {
            "conversations_attempted": len(conversation_texts),
            "conversations_succeeded": succeeded,
            "conversations_failed": failed,
            "conversations_truncated": truncated_count,
        },
        "token_usage": {
            "total_input_tokens": int(total_input),
            "total_output_tokens": int(total_output),
            "mean_input_tokens_per_conversation": round(
                total_input / max(len(results), 1), 0
            ),
            "mean_output_tokens_per_conversation": round(
                total_output / max(len(results), 1), 0
            ),
            "median_input_tokens_per_conversation": int(
                summaries_df["summary_input_tokens"].median()
            ),
            "max_input_tokens_single_conversation": int(
                summaries_df["summary_input_tokens"].max()
            ),
        },
        "cost": {
            "input_cost_usd": round(input_cost, 2),
            "output_cost_usd": round(output_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "pricing_note": (
                "Sonnet 4.5 Batch API: $1.50/MTok input, "
                "$7.50/MTok output (50% discount)"
            ),
        },
        "summary_statistics": {
            "mean_summary_length_chars": round(
                summaries_df["summary"].str.len().mean(), 0
            ),
            "median_summary_length_chars": round(
                summaries_df["summary"].str.len().median(), 0
            ),
            "mean_summary_length_words": round(
                summaries_df["summary"].str.split().str.len().mean(), 1
            ),
            "failed_count": len(errors),
            "retried_count": len(errors),
        },
        "output_files": [
            "data/processed/conversation_summaries.parquet",
            "data/processed/conversation_summaries_sample.csv",
        ],
    }

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved: {OUT_REPORT}")
    print(f"  Total API cost: ${total_cost:.2f}")
    return report


# ── Validation ───────────────────────────────────────────────────────────────
def run_validation(summaries_df, conv, conversation_texts):
    """Run validation checklist and print PASS/FAIL for each check."""
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)
    all_pass = True

    # 1. Parquet exists and loads
    try:
        test_df = pd.read_parquet(OUT_PARQUET)
        print("  [PASS] conversation_summaries.parquet exists and loads")
    except Exception as e:
        print(f"  [FAIL] conversation_summaries.parquet: {e}")
        all_pass = False

    # 2. Row count matches analysable conversations (within 1%)
    expected = len(conversation_texts)
    actual = len(summaries_df)
    pct_diff = abs(actual - expected) / max(expected, 1) * 100
    if pct_diff <= 1:
        print(f"  [PASS] Row count matches: {actual:,} / {expected:,} ({pct_diff:.1f}% diff)")
    else:
        print(f"  [FAIL] Row count mismatch: {actual:,} / {expected:,} ({pct_diff:.1f}% diff)")
        all_pass = False

    # 3. All conversation_ids exist in conversations_clean
    valid_ids = set(conv["conversation_id"])
    summary_ids = set(summaries_df["conversation_id"])
    orphans = summary_ids - valid_ids
    if len(orphans) == 0:
        print(f"  [PASS] All conversation_ids exist in conversations_clean")
    else:
        print(f"  [FAIL] {len(orphans)} orphan conversation_ids")
        all_pass = False

    # 4. No duplicate conversation_ids
    n_dupes = summaries_df["conversation_id"].duplicated().sum()
    if n_dupes == 0:
        print(f"  [PASS] No duplicate conversation_ids")
    else:
        print(f"  [FAIL] {n_dupes} duplicate conversation_ids")
        all_pass = False

    # 5. Failed count < 1%
    failed = (summaries_df["summary"] == "[SUMMARIZATION FAILED]").sum()
    fail_pct = failed / max(len(summaries_df), 1) * 100
    if fail_pct < 1:
        print(f"  [PASS] Failed summaries: {failed} ({fail_pct:.2f}%)")
    else:
        print(f"  [FAIL] Failed summaries: {failed} ({fail_pct:.2f}%) — exceeds 1%")
        all_pass = False

    # 6. Mean summary length sanity check
    mean_len = summaries_df["summary"].str.len().mean()
    if 30 <= mean_len <= 500:
        print(f"  [PASS] Mean summary length: {mean_len:.0f} chars (in 30-500 range)")
    else:
        print(f"  [FAIL] Mean summary length: {mean_len:.0f} chars (outside 30-500 range)")
        all_pass = False

    # 7. Report exists
    if os.path.exists(OUT_REPORT):
        with open(OUT_REPORT) as f:
            rpt = json.load(f)
        if "cost" in rpt:
            print(f"  [PASS] summarization_report.json exists with cost section")
        else:
            print(f"  [FAIL] summarization_report.json missing cost section")
            all_pass = False
    else:
        print(f"  [FAIL] summarization_report.json not found")
        all_pass = False

    # 8. Total cost range check
    if os.path.exists(OUT_REPORT):
        cost = rpt["cost"]["total_cost_usd"]
        print(f"  [INFO] Total cost: ${cost:.2f}")

    # 9. Sample summaries are coherent (just check they're non-empty strings)
    good = summaries_df[summaries_df["summary"] != "[SUMMARIZATION FAILED]"]
    if len(good) > 0:
        sample = good.sample(n=min(5, len(good)), random_state=42)
        non_empty = all(
            isinstance(s, str) and len(s) > 10
            for s in sample["summary"]
        )
        if non_empty:
            print(f"  [PASS] Sample summaries are coherent non-empty strings")
        else:
            print(f"  [FAIL] Some sample summaries are too short or empty")
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
        description="Module: Conversation Summarization via Claude Batch API"
    )
    parser.add_argument(
        "--resume-batch-id", default=None,
        help="Resume from a previously submitted batch ID (skip submission)"
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

    if args.resume_batch_id:
        # ── Resume mode: skip submission, go straight to polling/retrieval ──
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

    # Step 5: Save results
    summaries_df = save_results(results, conv)

    # Step 6: Generate report
    generate_report(
        summaries_df, results, errors, batch_id,
        conversation_texts, truncated_count
    )

    # Validation
    run_validation(summaries_df, conv, conversation_texts)

    print("\nDone!")


if __name__ == "__main__":
    main()
