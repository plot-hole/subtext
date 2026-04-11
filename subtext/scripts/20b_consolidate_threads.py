"""
Module 20b: Belief Thread Consolidation

Takes fine-grained threads from Module 20 and consolidates them into
30-80 meaningful belief threads using chunked LLM consolidation.

Strategy:
  1. Reload original thread assignments from Pass 1 cache
  2. Group thread names by entity prefix (person_a_, person_b_, user_, etc.)
  3. Chunk large groups (>100 threads) into batches of ~100
  4. Ask LLM to merge similar threads within each chunk → target 5-15 per chunk
  5. Second pass: merge across chunks for the same entity
  6. Remap all hypothesis assignments and recompute trajectories

Usage:
    python scripts/20b_consolidate_threads.py
    python scripts/20b_consolidate_threads.py --dry-run
    python scripts/20b_consolidate_threads.py --resume-batch-id msgbatch_xxxxx
"""

import os, sys, json, re, time, argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
INTERIM_DIR = BASE_DIR / "data" / "interim"
FIG_DIR     = BASE_DIR / "outputs" / "figures"
REPORT_DIR  = BASE_DIR / "outputs" / "reports"

BATCH_FILE     = INTERIM_DIR / "thread_consolidation_batch.jsonl"
BATCH_ID_FILE  = INTERIM_DIR / "thread_consolidation_batch_id.txt"
MAPPING_CACHE  = INTERIM_DIR / "thread_consolidation_mapping.json"
PASS1_CACHE    = INTERIM_DIR / "belief_pass1_results.json"

# ============================================================================
# MODEL CONFIG
# ============================================================================
MODEL              = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS  = 8192
PRICE_INPUT_MTOK   = 0.40
PRICE_OUTPUT_MTOK  = 2.00
CHUNK_SIZE         = 100  # thread names per batch request

# ============================================================================
# PROMPTS
# ============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a research assistant consolidating belief thread labels from a personal conversation corpus.",
    "",
    "You will receive a list of fine-grained thread names with hypothesis counts.",
    "Many are near-duplicates or slight variations of the same belief theme.",
    "",
    "Your task: AGGRESSIVELY merge similar threads into broader categories.",
    "",
    "Rules:",
    "- Threads about the same person's communication style, patterns, responsiveness, etc. → ONE thread",
    "- Threads about the same person's emotional state, feelings, reactions → ONE thread",
    "- Threads about the same person's intentions, motives, strategy → ONE thread",
    "- Threads about the same person's perception of the user → ONE thread",
    "- Threads about the same workplace issue → ONE thread",
    "- DO NOT create catch-all threads like 'person_general' or 'misc'",
    "- DO keep threads analytically meaningful — each should represent a distinct belief TOPIC",
    "- Target: reduce input threads to 5-15 consolidated threads",
    "- Every input thread must map to exactly one output thread",
    "",
    "Respond with ONLY a JSON object:",
    '{"mapping": {"original_thread_name": "consolidated_thread_name", ...}}',
])


def format_message(prefix, thread_names, thread_counts):
    """Format thread names for consolidation."""
    lines = [
        f"Entity/prefix: {prefix}",
        f"Threads to consolidate: {len(thread_names)}",
        f"Target: 5-15 consolidated threads",
        "",
        "Threads (hypothesis count | name):",
    ]
    for name, count in sorted(zip(thread_names, thread_counts), key=lambda x: -x[1]):
        lines.append(f"  {count:3d}  {name}")
    return "\n".join(lines)


# ============================================================================
# BATCH API HELPERS
# ============================================================================
def submit_batch(client, batch_file):
    with open(batch_file, "r") as f:
        requests = [json.loads(line) for line in f]

    batch = client.messages.batches.create(requests=requests)
    print(f"  {'=' * 44}")
    print(f"  BATCH SUBMITTED: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  {'=' * 44}")

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

    return batch.id


def poll_batch(client, batch_id):
    print(f"\n== Poll batch {batch_id} ==")
    print("  (Checking every 60 seconds.)\n")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        rc = batch.request_counts
        total = rc.succeeded + rc.errored + rc.processing + rc.expired + rc.canceled
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] Status: {batch.processing_status} | "
              f"Succeeded: {rc.succeeded:,} | Errored: {rc.errored:,} | "
              f"Processing: {rc.processing:,} | Total: {total:,}")
        if batch.processing_status == "ended":
            print(f"\n  Batch complete!")
            return batch
        time.sleep(60)


def retrieve_results(client, batch_id):
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
            errors.append({"custom_id": cid, "error": str(result.result)})
    return results, errors


def _strip_json_fences(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    return text.strip()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Consolidate belief threads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-batch-id", default=None)
    parser.add_argument("--from-current", action="store_true",
                        help="Consolidate from current thread assignments instead of reloading originals")
    args = parser.parse_args()

    # ---- Load thread assignments ----
    print("\n== Step 0: Load thread assignments =====================================")

    hypotheses = pd.read_parquet(DATA_DIR / "belief_threads.parquet")

    if not args.from_current:
        # Reload original Pass 1 assignments
        if not os.path.exists(PASS1_CACHE):
            print("ERROR: No Pass 1 cache found. Run Module 20 first.")
            sys.exit(1)

        with open(PASS1_CACHE, "r") as f:
            cached = json.load(f)
        original_assignments = cached.get("assignments", {})
        hypotheses["thread"] = hypotheses["hypothesis_id"].map(original_assignments).fillna("unassigned")
        print(f"  Source: original Pass 1 assignments ({len(set(original_assignments.values())):,} threads)")
    else:
        print(f"  Source: current thread assignments ({hypotheses['thread'].nunique():,} threads)")

    thread_counts = hypotheses["thread"].value_counts().to_dict()
    all_threads = list(thread_counts.keys())
    print(f"  Hypotheses          : {len(hypotheses):,}")
    print(f"  Threads to process  : {len(all_threads):,}")

    # ---- Group by prefix ----
    prefix_groups = defaultdict(list)
    for thread_name in all_threads:
        parts = thread_name.split("_")
        prefix = parts[0] if len(parts) > 1 else "misc"
        prefix_groups[prefix].append(thread_name)

    # Sort prefixes by total hypothesis count
    prefix_totals = {}
    for prefix, names in prefix_groups.items():
        prefix_totals[prefix] = sum(thread_counts.get(n, 0) for n in names)

    print(f"\n  Top entity prefixes:")
    for prefix, total in sorted(prefix_totals.items(), key=lambda x: -x[1])[:15]:
        print(f"    {prefix:25s}: {len(prefix_groups[prefix]):,} threads, {total:,} hypotheses")

    # ---- Build batch requests with chunking ----
    print(f"\n== Building consolidation batch ========================================")

    batch_requests = []
    request_id = 0

    for prefix, names in sorted(prefix_groups.items(), key=lambda x: -len(x[1])):
        if len(names) <= 3:
            # Passthrough — too few to consolidate
            continue

        # Sort by count descending for better context
        names_sorted = sorted(names, key=lambda n: -thread_counts.get(n, 0))

        # Chunk large groups
        for chunk_start in range(0, len(names_sorted), CHUNK_SIZE):
            chunk = names_sorted[chunk_start:chunk_start + CHUNK_SIZE]
            counts = [thread_counts.get(n, 0) for n in chunk]
            chunk_idx = chunk_start // CHUNK_SIZE

            custom_id = f"consolidate_{prefix}_{chunk_idx:02d}"
            request = {
                "custom_id": custom_id,
                "params": {
                    "model":      MODEL,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "system":     SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": format_message(prefix, chunk, counts)}],
                },
            }
            batch_requests.append(request)
            request_id += 1

    # Write batch file
    with open(BATCH_FILE, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Batch file    : {BATCH_FILE}")
    print(f"  Total requests: {len(batch_requests)}")

    est_cost = (len(batch_requests) * 1200 * PRICE_INPUT_MTOK / 1e6 +
                len(batch_requests) * 800 * PRICE_OUTPUT_MTOK / 1e6)
    print(f"  Estimated cost: ${est_cost:.2f}")

    if args.dry_run:
        print("\n  DRY RUN — exiting.")
        return

    # ---- Submit or resume ----
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    if args.resume_batch_id:
        batch_id = args.resume_batch_id
        print(f"\n== Resuming batch: {batch_id} ==")
    else:
        batch_id = submit_batch(client, BATCH_FILE)

    batch = poll_batch(client, batch_id)

    # ---- Retrieve and parse ----
    print(f"\n== Retrieve results ====================================================")
    results, errors = retrieve_results(client, batch_id)
    print(f"  Succeeded: {len(results):,}")
    print(f"  Errored  : {len(errors):,}")

    # Build full mapping
    full_mapping = {}
    parse_errors = 0

    for custom_id, res in tqdm(results.items(), desc="Parsing"):
        try:
            text = _strip_json_fences(res["raw"])
            data = json.loads(text)
            mapping = data.get("mapping", {})
            if not isinstance(mapping, dict):
                parse_errors += 1
                continue
            for orig, consolidated in mapping.items():
                consolidated = str(consolidated).strip().lower().replace(" ", "_")
                if orig and consolidated:
                    full_mapping[orig] = consolidated
        except Exception:
            parse_errors += 1

    # Passthrough for tiny groups
    for prefix, names in prefix_groups.items():
        if len(names) <= 3:
            for n in names:
                if n not in full_mapping:
                    full_mapping[n] = n

    # Any unmapped threads pass through
    for t in all_threads:
        if t not in full_mapping:
            full_mapping[t] = t

    new_thread_names = set(full_mapping.values())
    merged_count = sum(1 for k, v in full_mapping.items() if k != v)
    print(f"\n  Total mappings   : {len(full_mapping):,}")
    print(f"  Threads merged   : {merged_count:,}")
    print(f"  New thread count : {len(new_thread_names):,}")
    print(f"  Parse errors     : {parse_errors:,}")

    # Save mapping
    with open(MAPPING_CACHE, "w", encoding="utf-8") as f:
        json.dump(full_mapping, f, indent=2)

    # ---- Apply mapping ----
    print(f"\n== Applying consolidation ==============================================")
    hypotheses["thread"] = hypotheses["thread"].map(full_mapping).fillna(hypotheses["thread"])

    # Merge provenance back in
    provenance_df = pd.read_parquet(DATA_DIR / "belief_provenance.parquet")
    prov_map = provenance_df.set_index("hypothesis_id")[["provenance", "provenance_reasoning"]].to_dict("index")
    hypotheses["provenance"] = hypotheses["hypothesis_id"].map(
        lambda hid: prov_map.get(hid, {}).get("provenance", ""))
    hypotheses["provenance_reasoning"] = hypotheses["hypothesis_id"].map(
        lambda hid: prov_map.get(hid, {}).get("provenance_reasoning", ""))

    # Save
    hypotheses.to_parquet(DATA_DIR / "belief_threads.parquet", index=False)
    final_counts = hypotheses["thread"].value_counts()
    print(f"  Saved: belief_threads.parquet ({len(hypotheses):,} rows, {len(final_counts):,} threads)")

    # ---- Recompute trajectories ----
    print(f"\n== Recomputing trajectories ============================================")
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from importlib import import_module
    mod20 = import_module("20_belief_trajectories")
    summary_df = mod20.compute_trajectories(hypotheses)

    # ---- Regenerate figures ----
    print(f"\n== Regenerating figures ================================================")
    mod20.make_figures(hypotheses, summary_df)

    # ---- Summary ----
    print(f"\n== Summary =============================================================")
    print(f"  Final thread count : {len(final_counts):,}")
    print(f"  Singletons         : {(final_counts == 1).sum():,}")
    print(f"  Largest thread     : {final_counts.index[0]} ({final_counts.iloc[0]})")
    print(f"\n  Top 30 threads:")
    for name, count in final_counts.head(30).items():
        print(f"    {count:4d}  {name}")

    # Save catalog
    catalog = hypotheses.groupby("thread").agg(
        count=("hypothesis_id", "count"),
        types=("type", lambda x: ", ".join(sorted(x.unique()))),
        avg_confidence=("confidence_level", lambda x: x.astype(str).map(
            {"high": 3, "moderate": 2, "low": 1}).mean()),
        first_seen=("message_timestamp", "min"),
        last_seen=("message_timestamp", "max"),
    ).sort_values("count", ascending=False)
    catalog.to_csv(REPORT_DIR / "belief_thread_catalog.csv")
    print(f"\n  Saved: belief_thread_catalog.csv")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
