"""
Step 2: Parse — Linearize DAG-structured conversations into flat messages and conversations DataFrames.
Saves as Parquet files.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Try tiktoken; fall back to word count
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
    print("tiktoken loaded (cl100k_base encoding)")
except Exception:
    TIKTOKEN_AVAILABLE = False
    print("WARNING: tiktoken not available, falling back to word-count approximation")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "conversations.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Timezone for local time conversions
LOCAL_TZ = "America/Chicago"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            return len(enc.encode(text))
        except Exception:
            return len(text.split())
    return len(text.split())


def linearize_dag(mapping: dict) -> tuple:
    """
    Walk the DAG from root to produce a flat message sequence.
    Returns (messages_list, is_branched).
    Each message is a dict with extracted fields.
    """
    is_branched = False

    # Check for branching: any node with >1 child
    for node in mapping.values():
        children = node.get("children", [])
        if len(children) > 1:
            is_branched = True
            break

    # Find root (parent is None or absent)
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        return [], is_branched

    # Walk: at each node follow the LAST child (most recent path)
    messages = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if node is None:
            break

        msg = node.get("message")
        if msg is not None and msg.get("content") is not None:
            # Extract text from parts
            parts = msg.get("content", {}).get("parts", [])
            text_parts = []
            has_attachment = False
            for p in parts:
                if isinstance(p, str):
                    text_parts.append(p)
                elif isinstance(p, dict):
                    has_attachment = True

            text = "\n".join(text_parts) if text_parts else ""
            role = msg.get("author", {}).get("role", "unknown")
            timestamp = msg.get("create_time")

            # has_code: check for triple backtick
            has_code = "```" in text if text else False

            messages.append({
                "role": role,
                "text": text if text else None,
                "timestamp": timestamp,
                "has_code": has_code,
                "has_attachment": has_attachment,
            })

        # Follow last child
        children = node.get("children", [])
        if children:
            current_id = children[-1]
        else:
            current_id = None

    return messages, is_branched


def epoch_to_datetime(ts):
    """Convert epoch float to UTC datetime, or None."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OSError, ValueError, OverflowError):
        return None


def main():
    print(f"=== Step 2: Parse ===")
    print(f"Loading {RAW_PATH}...")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        conversations_raw = json.load(f)

    total = len(conversations_raw)
    print(f"Loaded {total} conversations")

    all_messages = []
    all_convos = []
    parse_errors = []
    error_count = 0

    for idx, conv in enumerate(tqdm(conversations_raw, desc="Parsing conversations")):
        conv_id = conv.get("id") or conv.get("conversation_id", f"unknown-{idx}")

        try:
            mapping = conv.get("mapping", {})
            if not mapping:
                parse_errors.append({
                    "conversation_id": conv_id,
                    "error": "No mapping field found",
                    "step": "parse"
                })
                error_count += 1
                continue

            messages, is_branched = linearize_dag(mapping)

            # Build message rows
            conv_messages = []
            for msg_idx, msg in enumerate(messages):
                ts = epoch_to_datetime(msg["timestamp"])
                text = msg["text"]
                tok = count_tokens(text) if text else 0
                char_ct = len(text) if text else 0
                word_ct = len(text.split()) if text else 0

                conv_messages.append({
                    "conversation_id": conv_id,
                    "msg_index": msg_idx,
                    "role": msg["role"],
                    "text": text,
                    "timestamp": ts,
                    "token_count": tok,
                    "char_count": char_ct,
                    "word_count": word_ct,
                    "has_code": msg["has_code"],
                    "has_attachment": msg["has_attachment"],
                    "is_branched": is_branched,
                })

            all_messages.extend(conv_messages)

            # Build conversation row
            create_time = epoch_to_datetime(conv.get("create_time"))
            update_time = epoch_to_datetime(conv.get("update_time"))

            # Fallbacks from messages
            msg_timestamps = [m["timestamp"] for m in conv_messages if m["timestamp"] is not None]
            if create_time is None and msg_timestamps:
                create_time = min(msg_timestamps)
            if update_time is None and msg_timestamps:
                update_time = max(msg_timestamps)

            duration_minutes = None
            if create_time and update_time:
                duration_minutes = (update_time - create_time).total_seconds() / 60.0

            msg_count = len(conv_messages)
            user_msgs = [m for m in conv_messages if m["role"] == "user"]
            asst_msgs = [m for m in conv_messages if m["role"] == "assistant"]
            user_msg_count = len(user_msgs)
            assistant_msg_count = len(asst_msgs)
            user_token_total = sum(m["token_count"] for m in user_msgs)
            assistant_token_total = sum(m["token_count"] for m in asst_msgs)
            has_code_conv = any(m["has_code"] for m in conv_messages) if conv_messages else False

            # Hour and day of week in local timezone
            hour_of_day = None
            day_of_week = None
            if create_time:
                local_dt = create_time.astimezone(
                    __import__("zoneinfo").ZoneInfo(LOCAL_TZ)
                )
                hour_of_day = local_dt.hour
                day_of_week = local_dt.weekday()

            all_convos.append({
                "conversation_id": conv_id,
                "title": conv.get("title"),
                "created_at": create_time,
                "updated_at": update_time,
                "duration_minutes": duration_minutes,
                "msg_count": msg_count,
                "user_msg_count": user_msg_count,
                "assistant_msg_count": assistant_msg_count,
                "user_token_total": user_token_total,
                "assistant_token_total": assistant_token_total,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "has_code": has_code_conv,
                "is_branched": is_branched,
            })

        except Exception as e:
            parse_errors.append({
                "conversation_id": conv_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "step": "parse"
            })
            error_count += 1

        # Periodic summary
        if (idx + 1) % 1000 == 0:
            print(f"  [{idx+1}/{total}] Messages extracted: {len(all_messages)}, Errors: {error_count}")

    # Build DataFrames
    print(f"\nBuilding DataFrames...")
    messages_df = pd.DataFrame(all_messages)
    convos_df = pd.DataFrame(all_convos)

    # Set dtypes
    if not messages_df.empty:
        messages_df["role"] = messages_df["role"].astype("category")
        messages_df["msg_index"] = messages_df["msg_index"].astype("int32")
        messages_df["token_count"] = messages_df["token_count"].astype("Int32")
        messages_df["char_count"] = messages_df["char_count"].astype("Int32")
        messages_df["word_count"] = messages_df["word_count"].astype("Int32")
        if messages_df["timestamp"].notna().any():
            messages_df["timestamp"] = pd.to_datetime(messages_df["timestamp"], utc=True)

    if not convos_df.empty:
        if convos_df["created_at"].notna().any():
            convos_df["created_at"] = pd.to_datetime(convos_df["created_at"], utc=True)
        if convos_df["updated_at"].notna().any():
            convos_df["updated_at"] = pd.to_datetime(convos_df["updated_at"], utc=True)
        convos_df["duration_minutes"] = convos_df["duration_minutes"].astype("float32")
        convos_df["msg_count"] = convos_df["msg_count"].astype("int32")
        convos_df["user_msg_count"] = convos_df["user_msg_count"].astype("int32")
        convos_df["assistant_msg_count"] = convos_df["assistant_msg_count"].astype("int32")
        convos_df["user_token_total"] = convos_df["user_token_total"].astype("int32")
        convos_df["assistant_token_total"] = convos_df["assistant_token_total"].astype("int32")
        convos_df["hour_of_day"] = convos_df["hour_of_day"].astype("Int8")
        convos_df["day_of_week"] = convos_df["day_of_week"].astype("Int8")

    # Save
    msg_path = PROCESSED_DIR / "messages.parquet"
    conv_path = PROCESSED_DIR / "conversations.parquet"
    messages_df.to_parquet(msg_path, index=False)
    convos_df.to_parquet(conv_path, index=False)
    print(f"  Saved {msg_path} ({msg_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Saved {conv_path} ({conv_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save parse errors
    errors_path = REPORTS_DIR / "parse_errors.json"
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(parse_errors, f, indent=2, default=str)
    print(f"  Saved {errors_path} ({len(parse_errors)} errors)")

    # Summary
    print(f"\n=== Step 2 Summary ===")
    print(f"  Total conversations:       {len(convos_df)}")
    print(f"  Total messages:            {len(messages_df)}")
    if not messages_df.empty:
        print(f"  Total tokens (all roles):  {messages_df['token_count'].sum():,}")
        print(f"  User messages:             {(messages_df['role'] == 'user').sum():,}")
        print(f"  Assistant messages:        {(messages_df['role'] == 'assistant').sum():,}")
    if not convos_df.empty and convos_df["created_at"].notna().any():
        date_min = convos_df["created_at"].min()
        date_max = convos_df["created_at"].max()
        print(f"  Date range:                {date_min} to {date_max}")
    print(f"  Parse errors:              {error_count}")
    print(f"  Tiktoken available:        {TIKTOKEN_AVAILABLE}")
    print(f"  Timezone:                  {LOCAL_TZ}")
    print(f"\n=== Step 2 Complete ===")


if __name__ == "__main__":
    main()
