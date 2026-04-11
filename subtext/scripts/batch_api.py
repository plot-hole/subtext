"""
Shared utilities for the Anthropic Batch API lifecycle.

Consolidates the submit → poll → retrieve → retry pattern used by
12_summarize, 13_functional_classify, 14_emotional_state,
17_frame_adoption, 19_hypothesis_extraction, 21_corpus_discovery,
and 20_belief_trajectories.
"""

import json
import time


def submit_batch(client, batch_file):
    """Submit a JSONL batch file to the Anthropic Batch API.

    Returns the batch ID string.
    """
    print("Submitting batch to Anthropic API...")

    requests = []
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            requests.append(json.loads(line))

    batch = client.messages.batches.create(requests=requests)

    print(f"  ============================================")
    print(f"  BATCH SUBMITTED: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ============================================")
    print(f"  Save this batch ID in case you need to resume!")
    return batch.id


def poll_batch(client, batch_id, interval=60):
    """Poll for batch completion, printing status each check.

    Args:
        client: Anthropic client instance.
        batch_id: ID returned by submit_batch.
        interval: Seconds between polls (default 60).

    Returns the final batch status object.
    """
    print(f"\nPolling batch {batch_id} for completion...")
    print(f"  (Checking every {interval} seconds. This may take 1-6 hours.)\n")

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

        time.sleep(interval)


def retrieve_results(client, batch_id):
    """Retrieve results from a completed batch.

    Returns:
        (results, errors) where results is a dict keyed by custom_id
        mapping to {"text": str, "input_tokens": int, "output_tokens": int},
        and errors is a list of dicts with error details.
    """
    print(f"Retrieving results for batch {batch_id}...")
    results = {}
    errors = []

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            response = result.result.message
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            results[custom_id] = {
                "text": text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            errors.append({
                "custom_id": custom_id,
                "error_type": result.result.type,
                "error": str(result.result.error)
                         if hasattr(result.result, "error") else "unknown",
            })

    print(f"  Succeeded: {len(results):,}")
    print(f"  Errored: {len(errors):,}")
    return results, errors


def retry_errors(client, errors, build_message_fn, model, max_tokens,
                 system_prompt=None, delay=1, max_retries=None):
    """Retry failed batch requests individually via the standard Messages API.

    Args:
        client: Anthropic client instance.
        errors: List of error dicts from retrieve_results (must have "custom_id").
        build_message_fn: Callable(custom_id) -> list of message dicts,
            or None if this request should be skipped.
        model: Model ID string.
        max_tokens: Max output tokens per request.
        system_prompt: Optional system prompt string.
        delay: Seconds between retries (default 1). Uses exponential backoff
            on rate-limit errors (429).
        max_retries: Optional cap on how many errors to retry.

    Returns:
        Dict of retried results keyed by custom_id, same shape as
        retrieve_results output.
    """
    if not errors:
        return {}

    to_retry = errors[:max_retries] if max_retries else errors
    print(f"\nRetrying {len(to_retry)} failed requests via standard API...")
    retried = {}

    for error_item in to_retry:
        custom_id = error_item["custom_id"]
        messages = build_message_fn(custom_id)
        if messages is None:
            continue

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system_prompt:
            kwargs["system"] = system_prompt

        current_delay = delay
        try:
            response = client.messages.create(**kwargs)
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            retried[custom_id] = {
                "text": text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                current_delay = min(current_delay * 2, 60)
            print(f"  Retry failed for {custom_id}: {e}")

        time.sleep(current_delay)

    print(f"  Retries succeeded: {len(retried):,}")
    return retried


def check_api_key():
    """Validate that ANTHROPIC_API_KEY is set. Exits if not.

    Returns the API key string.
    """
    import os
    import sys
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print("API key found: ****")
    return api_key
