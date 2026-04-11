"""
Step 3: Clean & Validate — Dedup, null audit, timestamp imputation, quality flags, text normalization.
"""
import json
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"


def null_audit(df: pd.DataFrame, name: str) -> dict:
    """Compute null counts and percentages for every column."""
    audit = {}
    for col in df.columns:
        total = len(df)
        null_ct = int(df[col].isna().sum())
        audit[col] = {
            "count": total,
            "null_count": null_ct,
            "null_pct": round(null_ct / total * 100, 2) if total > 0 else 0.0,
        }
    return {name: audit}


def main():
    print("=== Step 3: Clean & Validate ===")

    # Load
    print("Loading parquet files...")
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)
    print(f"  Messages: {len(messages)} rows, Conversations: {len(convos)} rows")

    cleaning_summary = {
        "duplicates_removed_conversations": 0,
        "duplicates_removed_messages": 0,
        "timestamps_imputed_messages": 0,
        "timestamps_imputed_conversations_created": 0,
        "timestamps_imputed_conversations_updated": 0,
        "quality_flags": {},
    }

    # === 3a. Deduplication ===
    print("\n--- 3a. Deduplication ---")
    before = len(convos)
    convos = convos.drop_duplicates(subset=["conversation_id"])
    dup_convos = before - len(convos)
    cleaning_summary["duplicates_removed_conversations"] = dup_convos
    print(f"  Conversations: removed {dup_convos} duplicates")

    before = len(messages)
    messages = messages.drop_duplicates(subset=["conversation_id", "msg_index"])
    dup_msgs = before - len(messages)
    cleaning_summary["duplicates_removed_messages"] = dup_msgs
    print(f"  Messages: removed {dup_msgs} duplicates")

    # === 3b. Null audit ===
    print("\n--- 3b. Null Audit ---")
    audit = {}
    audit.update(null_audit(messages, "messages"))
    audit.update(null_audit(convos, "conversations"))
    audit_path = REPORTS_DIR / "null_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved null audit to {audit_path}")
    # Print summary
    for table_name, cols in audit.items():
        nulls_found = {c: v for c, v in cols.items() if v["null_count"] > 0}
        if nulls_found:
            print(f"  {table_name} columns with nulls:")
            for c, v in nulls_found.items():
                print(f"    {c}: {v['null_count']} ({v['null_pct']}%)")

    # === 3c. Timestamp handling ===
    print("\n--- 3c. Timestamp Imputation ---")

    # Messages: interpolate null timestamps from neighbors
    null_ts_before = int(messages["timestamp"].isna().sum())
    if null_ts_before > 0:
        messages = messages.sort_values(["conversation_id", "msg_index"]).reset_index(drop=True)
        # Convert timestamps to epoch seconds (float) for safe interpolation
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        messages["_ts_seconds"] = messages["timestamp"].apply(
            lambda x: (x - epoch).total_seconds() if pd.notna(x) else np.nan
        )
        # Interpolate within each conversation group
        messages["_ts_seconds"] = messages.groupby("conversation_id")["_ts_seconds"].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )
        # Build a complete timestamp series: keep originals, fill nulls from interpolation
        null_mask = messages["timestamp"].isna() & messages["_ts_seconds"].notna()
        imputed_ts = pd.to_datetime(
            messages.loc[null_mask, "_ts_seconds"], unit="s", utc=True
        ).dt.floor("us")  # match column's us precision
        # Rebuild the column to avoid dtype mismatch
        new_ts = messages["timestamp"].copy()
        new_ts.loc[null_mask] = imputed_ts
        messages["timestamp"] = new_ts
        messages.drop(columns=["_ts_seconds"], inplace=True)

    null_ts_after = int(messages["timestamp"].isna().sum())
    imputed_msg = null_ts_before - null_ts_after
    cleaning_summary["timestamps_imputed_messages"] = imputed_msg
    print(f"  Messages: imputed {imputed_msg} timestamps (remaining null: {null_ts_after})")

    # Conversations: fill from messages
    imputed_created = 0
    imputed_updated = 0
    for idx_row, row in convos[convos["created_at"].isna()].iterrows():
        conv_msgs = messages[messages["conversation_id"] == row["conversation_id"]]
        ts_vals = conv_msgs["timestamp"].dropna()
        if len(ts_vals) > 0:
            convos.loc[idx_row, "created_at"] = ts_vals.min()
            imputed_created += 1
    for idx_row, row in convos[convos["updated_at"].isna()].iterrows():
        conv_msgs = messages[messages["conversation_id"] == row["conversation_id"]]
        ts_vals = conv_msgs["timestamp"].dropna()
        if len(ts_vals) > 0:
            convos.loc[idx_row, "updated_at"] = ts_vals.max()
            imputed_updated += 1

    cleaning_summary["timestamps_imputed_conversations_created"] = imputed_created
    cleaning_summary["timestamps_imputed_conversations_updated"] = imputed_updated
    print(f"  Conversations: imputed {imputed_created} created_at, {imputed_updated} updated_at")

    # === 3d. Quality flags ===
    print("\n--- 3d. Quality Flags ---")

    def assign_quality(row):
        conv_msgs = messages[messages["conversation_id"] == row["conversation_id"]]
        user_asst = conv_msgs[conv_msgs["role"].isin(["user", "assistant"])]

        if len(user_asst) == 0:
            return "empty"

        has_text = user_asst["text"].notna() & (user_asst["text"] != "")
        if not has_text.any():
            return "empty"

        null_ts_pct = conv_msgs["timestamp"].isna().mean()
        null_text_pct = (~has_text).mean() if len(user_asst) > 0 else 0
        if null_ts_pct > 0.5 or null_text_pct > 0.5:
            return "partial"

        return "complete"

    convos["quality_flag"] = convos.apply(assign_quality, axis=1)
    flag_dist = convos["quality_flag"].value_counts().to_dict()
    cleaning_summary["quality_flags"] = flag_dist
    print(f"  Quality flag distribution: {flag_dist}")

    # === 3e. Text normalization ===
    print("\n--- 3e. Text Normalization ---")
    if "text" in messages.columns:
        non_null_mask = messages["text"].notna()
        messages.loc[non_null_mask, "text"] = (
            messages.loc[non_null_mask, "text"]
            .str.strip()
            .apply(lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x)
        )
    print("  Text stripped and NFC-normalized")

    # === 3f. Write cleaned files ===
    print("\n--- 3f. Writing Cleaned Files ---")
    messages.to_parquet(MSG_PATH, index=False)
    convos.to_parquet(CONV_PATH, index=False)
    print(f"  Overwritten {MSG_PATH}")
    print(f"  Overwritten {CONV_PATH}")

    # CSV samples
    csv_msg = PROCESSED_DIR / "messages.csv"
    csv_conv = PROCESSED_DIR / "conversations.csv"
    messages.head(1000).to_csv(csv_msg, index=False)
    convos.to_csv(csv_conv, index=False)
    print(f"  Saved {csv_msg} (first 1000 rows)")
    print(f"  Saved {csv_conv}")

    # Save cleaning summary
    summary_path = REPORTS_DIR / "cleaning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_summary, f, indent=2)
    print(f"  Saved {summary_path}")

    print(f"\n=== Step 3 Summary ===")
    print(f"  Duplicate conversations removed:  {dup_convos}")
    print(f"  Duplicate messages removed:        {dup_msgs}")
    print(f"  Message timestamps imputed:        {imputed_msg}")
    print(f"  Quality flags:                     {flag_dist}")
    print(f"  Final messages count:              {len(messages)}")
    print(f"  Final conversations count:         {len(convos)}")
    print(f"\n=== Step 3 Complete ===")


if __name__ == "__main__":
    main()
