"""
Phase 2, Step 1: Edge Case Detection & Tagging.
Scans conversations and messages for edge cases, adds tagging columns.
Does NOT delete any data — tags everything for filtering at analysis time.
"""
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "quality_config.json"

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"

# Error response patterns (case-insensitive substring matching)
ERROR_PATTERNS = [
    "i'm sorry, but i can't",
    "i'm sorry, but i can't",
    "i apologize, but i can't",
    "i apologize, but i'm unable",
    "i'm unable to",
    "network error",
    "something went wrong",
    "an error occurred",
    "i cannot assist with",
    "i'm not able to",
    "as an ai language model, i cannot",
    "i don't have the ability to",
]

# Plugin/tool detection patterns
PLUGIN_PATTERNS = [
    r"\[Searching",
    r"\[Browsing",
    r"Used \[",
    r"I used the \w+ plugin",
    r"I used the \w+ tool",
]


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_code_fraction(text):
    """Return fraction of text that is inside code fences."""
    if not text:
        return 0.0
    pattern = r"```[\s\S]*?```"
    matches = re.findall(pattern, text)
    code_chars = sum(len(m) for m in matches)
    return code_chars / len(text) if len(text) > 0 else 0.0


def main():
    print("=== Phase 2, Step 1: Edge Case Detection & Tagging ===")

    config = load_config()
    bulk_threshold = config["bulk_input_token_threshold"]
    sys_fingerprints = config["default_system_prompts_fingerprints"]

    # Load data
    print("Loading parquet files...")
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)
    print(f"  Messages: {len(messages)}, Conversations: {len(convos)}")

    # ================================================================
    # 1c. Message-level edge case tags (do these first, needed for conv-level)
    # ================================================================
    print("\n--- Message-level edge case tags ---")

    # is_empty_msg
    messages["is_empty_msg"] = (
        messages["text"].isna() |
        (messages["text"].fillna("").str.strip() == "")
    )

    # is_bulk_input
    messages["is_bulk_input"] = (
        (messages["role"] == "user") &
        (messages["token_count"] > bulk_threshold)
    )

    # is_error_response
    def check_error(row):
        if row["role"] != "assistant":
            return False
        text = row["text"]
        if not text or pd.isna(text):
            return False
        text_lower = text.lower()
        for pat in ERROR_PATTERNS:
            if pat in text_lower:
                # For "i'm unable to" pattern, check if message is short
                if "unable to" in pat and row["token_count"] is not None and row["token_count"] > 50:
                    continue
                return True
        return False

    messages["is_error_response"] = messages.apply(check_error, axis=1)

    # Check for duplicate assistant messages (regeneration artifacts)
    messages = messages.sort_values(["conversation_id", "msg_index"]).reset_index(drop=True)
    prev_text = messages.groupby("conversation_id")["text"].shift(1)
    is_dup = (
        (messages["role"] == "assistant") &
        (messages["text"].notna()) &
        (prev_text.notna()) &
        (messages["text"] == prev_text)
    )
    messages["is_error_response"] = messages["is_error_response"] | is_dup

    # is_system_boilerplate
    def check_boilerplate(row):
        if row["role"] != "system":
            return False
        text = row["text"]
        if not text or pd.isna(text):
            return False
        for fp in sys_fingerprints:
            if fp.lower() in text.lower():
                return True
        return False

    messages["is_system_boilerplate"] = messages.apply(check_boilerplate, axis=1)

    # content_type
    def classify_content(row):
        if row["is_empty_msg"]:
            return "empty"
        text = row["text"]
        if not text:
            return "empty"
        code_frac = detect_code_fraction(text)
        if code_frac > 0.8:
            return "code"
        elif code_frac > 0:
            return "mixed"
        else:
            return "text"

    messages["content_type"] = messages.apply(classify_content, axis=1).astype("category")

    print(f"  is_empty_msg:         {messages['is_empty_msg'].sum():,}")
    print(f"  is_bulk_input:        {messages['is_bulk_input'].sum():,}")
    print(f"  is_error_response:    {messages['is_error_response'].sum():,}")
    print(f"  is_system_boilerplate:{messages['is_system_boilerplate'].sum():,}")
    print(f"  content_type dist:    {messages['content_type'].value_counts().to_dict()}")

    # ================================================================
    # 1a. Conversation-level edge case tags
    # ================================================================
    print("\n--- Conversation-level edge case tags ---")

    # Pre-compute per-conversation aggregations from messages
    msg_grouped = messages.groupby("conversation_id")

    # is_empty
    def conv_is_empty(conv_id):
        grp = messages[messages["conversation_id"] == conv_id]
        ua = grp[grp["role"].isin(["user", "assistant"])]
        if len(ua) == 0:
            return True
        has_text = ua["text"].notna() & (ua["text"].str.strip() != "")
        return not has_text.any()

    # Vectorized approach using aggregations
    ua_msgs = messages[messages["role"].isin(["user", "assistant"])]
    ua_has_text = ua_msgs.groupby("conversation_id").apply(
        lambda g: (g["text"].notna() & (g["text"].str.strip() != "")).any()
    )
    convos["is_empty"] = ~convos["conversation_id"].map(ua_has_text).fillna(False)

    # has_tool_use
    tool_convs = set(messages[messages["role"] == "tool"]["conversation_id"].unique())
    convos["has_tool_use"] = convos["conversation_id"].isin(tool_convs)

    # has_plugin — check message text for plugin patterns
    def has_plugin_text(text):
        if not text or pd.isna(text):
            return False
        for pat in PLUGIN_PATTERNS:
            if re.search(pat, text):
                return True
        return False

    plugin_msgs = messages[messages["text"].notna()].copy()
    plugin_msgs["_has_plugin"] = plugin_msgs["text"].apply(has_plugin_text)
    plugin_convs = set(plugin_msgs[plugin_msgs["_has_plugin"]]["conversation_id"].unique())
    convos["has_plugin"] = convos["conversation_id"].isin(plugin_convs)

    # is_custom_gpt — system message doesn't match default fingerprints
    sys_msgs = messages[messages["role"] == "system"].copy()
    if len(sys_msgs) > 0:
        sys_msgs["_is_default"] = sys_msgs.apply(check_boilerplate, axis=1)
        # Conversations with a system message that is NOT default → custom GPT
        has_sys = sys_msgs.groupby("conversation_id")["_is_default"].any()
        # Custom GPT = has system message AND none match defaults
        custom_gpt_ids = set()
        for conv_id, grp in sys_msgs.groupby("conversation_id"):
            has_text = grp["text"].notna() & (grp["text"].str.strip() != "")
            if has_text.any() and not grp["_is_default"].any():
                custom_gpt_ids.add(conv_id)
        convos["is_custom_gpt"] = convos["conversation_id"].isin(custom_gpt_ids)
    else:
        convos["is_custom_gpt"] = False

    # is_multimodal
    attach_convs = set(messages[messages["has_attachment"]]["conversation_id"].unique())
    convos["is_multimodal"] = convos["conversation_id"].isin(attach_convs)

    # has_bulk_input
    bulk_convs = set(messages[messages["is_bulk_input"]]["conversation_id"].unique())
    convos["has_bulk_input"] = convos["conversation_id"].isin(bulk_convs)

    # is_single_turn
    convos["is_single_turn"] = convos["user_msg_count"] <= 1

    print(f"  is_empty:       {convos['is_empty'].sum()}")
    print(f"  has_tool_use:   {convos['has_tool_use'].sum()}")
    print(f"  has_plugin:     {convos['has_plugin'].sum()}")
    print(f"  is_custom_gpt:  {convos['is_custom_gpt'].sum()}")
    print(f"  is_multimodal:  {convos['is_multimodal'].sum()}")
    print(f"  has_bulk_input: {convos['has_bulk_input'].sum()}")
    print(f"  is_single_turn: {convos['is_single_turn'].sum()}")

    # ================================================================
    # 1b. Conversation type classification (priority hierarchy)
    # ================================================================
    print("\n--- Conversation type classification ---")

    # Need code-heavy check: has_code AND >50% of user messages have code fences
    user_msgs_code = messages[(messages["role"] == "user")].copy()
    user_code_frac = user_msgs_code.groupby("conversation_id").apply(
        lambda g: (g["has_code"].sum() / len(g)) if len(g) > 0 else 0.0
    )

    def classify_conv_type(row):
        if row["is_empty"]:
            return "empty"
        if row["is_single_turn"]:
            return "single_turn"
        conv_id = row["conversation_id"]
        code_frac = user_code_frac.get(conv_id, 0.0)
        if row["has_code"] and code_frac > 0.5:
            return "code_heavy"
        if row["has_tool_use"] or row["has_plugin"]:
            return "tool_assisted"
        if row["is_custom_gpt"]:
            return "custom_gpt"
        if row["is_multimodal"] and not (row["has_code"] and code_frac > 0.5):
            return "multimodal"
        return "standard"

    convos["conversation_type"] = convos.apply(classify_conv_type, axis=1).astype("category")
    type_dist = convos["conversation_type"].value_counts().to_dict()
    print(f"  Conversation type distribution: {type_dist}")

    # ================================================================
    # Save
    # ================================================================
    print("\n--- Saving ---")
    messages.to_parquet(MSG_PATH, index=False)
    convos.to_parquet(CONV_PATH, index=False)
    print(f"  Saved {MSG_PATH}")
    print(f"  Saved {CONV_PATH}")

    # Edge case report
    report = {
        "conversations": {
            "total": int(len(convos)),
            "empty": int(convos["is_empty"].sum()),
            "single_turn": int(convos["is_single_turn"].sum()),
            "has_tool_use": int(convos["has_tool_use"].sum()),
            "has_plugin": int(convos["has_plugin"].sum()),
            "is_custom_gpt": int(convos["is_custom_gpt"].sum()),
            "is_multimodal": int(convos["is_multimodal"].sum()),
            "has_bulk_input": int(convos["has_bulk_input"].sum()),
            "conversation_type_distribution": {k: int(v) for k, v in type_dist.items()},
        },
        "messages": {
            "total": int(len(messages)),
            "empty_messages": int(messages["is_empty_msg"].sum()),
            "bulk_inputs": int(messages["is_bulk_input"].sum()),
            "error_responses": int(messages["is_error_response"].sum()),
            "system_boilerplate": int(messages["is_system_boilerplate"].sum()),
            "content_type_distribution": {
                k: int(v) for k, v in messages["content_type"].value_counts().to_dict().items()
            },
        },
    }
    report_path = REPORTS_DIR / "edge_case_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {report_path}")

    print(f"\n=== Step 1 Complete ===")


if __name__ == "__main__":
    main()
