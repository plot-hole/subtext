"""
Phase 2, Step 3: PII Scan.
Detects PII in user messages using spaCy NER and regex patterns.
Saves detailed detections locally (never in outputs/), aggregate stats to reports.
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
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

MSG_PATH = PROCESSED_DIR / "messages.parquet"
CONV_PATH = PROCESSED_DIR / "conversations.parquet"
PII_DETAIL_PATH = INTERIM_DIR / "pii_detections.json"
PII_REPORT_PATH = REPORTS_DIR / "pii_scan_report.json"

# Regex patterns
REGEX_PATTERNS = {
    "email": {
        "pattern": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "priority": "high",
    },
    "phone": {
        "pattern": r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "priority": "high",
    },
    "ssn_pattern": {
        "pattern": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "priority": "critical",
    },
    "url": {
        "pattern": r"https?://[^\s]+",
        "priority": "medium",
    },
    "street_address": {
        "pattern": r"\b\d{1,5}\s+[A-Z][a-z]+\s+(St|Ave|Blvd|Dr|Ln|Rd|Ct|Way|Pl)\b",
        "priority": "medium",
    },
}

# spaCy entity type to priority mapping
NER_PRIORITY = {
    "PERSON": "high",
    "ORG": "medium",
    "GPE": "medium",
    "LOC": "medium",
    "DATE": "low",
    "MONEY": "low",
}

NER_TYPES = set(NER_PRIORITY.keys())

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def main():
    print("=== Phase 2, Step 3: PII Scan ===")

    # Load messages
    print("Loading messages...")
    messages = pd.read_parquet(MSG_PATH)
    convos = pd.read_parquet(CONV_PATH)

    # Filter to user messages with text
    user_msgs = messages[
        (messages["role"] == "user") &
        (messages["text"].notna()) &
        (messages["text"].str.strip() != "")
    ].copy()
    print(f"  User messages to scan: {len(user_msgs)}")

    # Try loading spaCy
    spacy_available = False
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 2_000_000
        spacy_available = True
        print("  spaCy en_core_web_sm loaded")
    except Exception as e:
        print(f"  WARNING: spaCy not available ({e}), running regex-only scan")

    all_detections = []
    ner_errors = 0

    # ================================================================
    # 3a. NER-based detection (if spaCy available)
    # ================================================================
    if spacy_available:
        print("\n--- 3a. spaCy NER Scan ---")
        texts = user_msgs["text"].tolist()
        conv_ids = user_msgs["conversation_id"].tolist()
        msg_indices = user_msgs["msg_index"].tolist()

        batch_size = 1000
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="  NER batches"):
            batch_texts = texts[batch_start:batch_start + batch_size]
            batch_conv_ids = conv_ids[batch_start:batch_start + batch_size]
            batch_msg_idxs = msg_indices[batch_start:batch_start + batch_size]

            try:
                docs = list(nlp.pipe(batch_texts, batch_size=64))
                for doc, cid, midx in zip(docs, batch_conv_ids, batch_msg_idxs):
                    for ent in doc.ents:
                        if ent.label_ in NER_TYPES:
                            all_detections.append({
                                "conversation_id": cid,
                                "msg_index": int(midx),
                                "entity_text": ent.text,
                                "entity_type": ent.label_,
                                "detection_method": "spacy_ner",
                                "char_start": ent.start_char,
                                "char_end": ent.end_char,
                                "priority": NER_PRIORITY[ent.label_],
                            })
            except Exception as e:
                ner_errors += 1
                if ner_errors <= 5:
                    print(f"  NER error in batch starting at {batch_start}: {e}")

        print(f"  NER detections: {len(all_detections)}, errors: {ner_errors}")

    # ================================================================
    # 3b. Regex-based detection
    # ================================================================
    print("\n--- 3b. Regex Scan ---")
    regex_count = 0

    for _, row in tqdm(user_msgs.iterrows(), total=len(user_msgs), desc="  Regex scan"):
        text = row["text"]
        cid = row["conversation_id"]
        midx = row["msg_index"]

        for pattern_name, pinfo in REGEX_PATTERNS.items():
            for match in re.finditer(pinfo["pattern"], text):
                all_detections.append({
                    "conversation_id": cid,
                    "msg_index": int(midx),
                    "entity_text": match.group(),
                    "entity_type": pattern_name,
                    "detection_method": "regex",
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "priority": pinfo["priority"],
                })
                regex_count += 1

    print(f"  Regex detections: {regex_count}")
    print(f"  Total detections: {len(all_detections)}")

    # ================================================================
    # 3c. Save detailed detections to interim (NEVER to outputs)
    # ================================================================
    print("\n--- Saving detections ---")
    with open(PII_DETAIL_PATH, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, default=str)
    print(f"  Saved {PII_DETAIL_PATH} (LOCAL ONLY - {len(all_detections)} detections)")

    # ================================================================
    # Add convenience columns to messages and conversations
    # ================================================================
    print("\n--- Adding PII columns ---")

    # Build lookup: (conversation_id, msg_index) -> highest priority
    pii_by_msg = {}
    for det in all_detections:
        key = (det["conversation_id"], det["msg_index"])
        existing = pii_by_msg.get(key)
        if existing is None or PRIORITY_ORDER.get(det["priority"], 99) < PRIORITY_ORDER.get(existing, 99):
            pii_by_msg[key] = det["priority"]

    messages["has_pii"] = messages.apply(
        lambda r: (r["conversation_id"], r["msg_index"]) in pii_by_msg, axis=1
    )
    messages["pii_priority"] = messages.apply(
        lambda r: pii_by_msg.get((r["conversation_id"], r["msg_index"])), axis=1
    ).astype("category")

    # Conversation-level
    pii_by_conv = {}
    for det in all_detections:
        cid = det["conversation_id"]
        existing = pii_by_conv.get(cid)
        if existing is None or PRIORITY_ORDER.get(det["priority"], 99) < PRIORITY_ORDER.get(existing, 99):
            pii_by_conv[cid] = det["priority"]

    pii_conv_ids = set(pii_by_conv.keys())
    convos["has_pii"] = convos["conversation_id"].isin(pii_conv_ids)
    convos["max_pii_priority"] = convos["conversation_id"].map(pii_by_conv).astype("category")

    print(f"  Messages with PII:       {messages['has_pii'].sum():,}")
    print(f"  Conversations with PII:  {convos['has_pii'].sum():,}")

    # Save updated parquets
    messages.to_parquet(MSG_PATH, index=False)
    convos.to_parquet(CONV_PATH, index=False)
    print(f"  Saved updated parquet files")

    # ================================================================
    # 3c. Aggregate report (safe to share — no actual PII)
    # ================================================================
    print("\n--- Generating aggregate report ---")

    entity_type_counts = {}
    priority_counts = {}
    critical_conv_ids = []
    for det in all_detections:
        et = det["entity_type"]
        entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
        pr = det["priority"]
        priority_counts[pr] = priority_counts.get(pr, 0) + 1
        if pr == "critical" and det["conversation_id"] not in critical_conv_ids:
            critical_conv_ids.append(det["conversation_id"])

    report = {
        "total_user_messages_scanned": int(len(user_msgs)),
        "messages_with_pii": int(messages["has_pii"].sum()),
        "conversations_with_pii": int(convos["has_pii"].sum()),
        "entity_type_counts": entity_type_counts,
        "priority_distribution": priority_counts,
        "top_10_person_names_redacted": "REDACTED - see data/interim/pii_detections.json locally",
        "conversations_with_critical_pii": critical_conv_ids,
        "spacy_available": spacy_available,
        "ner_errors": ner_errors,
    }

    with open(PII_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {PII_REPORT_PATH}")

    print(f"\n=== Step 3 Summary ===")
    print(f"  Messages scanned:        {len(user_msgs):,}")
    print(f"  Total PII detections:    {len(all_detections):,}")
    print(f"  Messages with PII:       {messages['has_pii'].sum():,}")
    print(f"  Conversations with PII:  {convos['has_pii'].sum():,}")
    print(f"  Critical PII convos:     {len(critical_conv_ids)}")
    print(f"  Entity type counts:      {entity_type_counts}")
    print(f"\n=== Step 3 Complete ===")


if __name__ == "__main__":
    main()
