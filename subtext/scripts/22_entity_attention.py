"""
Module 22: Entity Attention Over Time
Tracks mentions of specific people across analysable conversations over 11 months.
Combines spaCy NER with regex patterns for known names to produce per-conversation
and weekly time series of entity attention. Changepoint detection (PELT) identifies
inflection points in top entity series.

Zero API cost — local computation only.

Usage:
    python scripts/22_entity_attention.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
CONV_PATH = PROJECT_ROOT / "data" / "processed" / "conversations_clean.parquet"
MSGS_PATH = PROJECT_ROOT / "data" / "processed" / "messages_clean.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "entity_attention.parquet"
OUT_WEEKLY = PROJECT_ROOT / "data" / "processed" / "entity_attention_weekly.parquet"
OUT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "entity_attention_report.json"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "entity_attention"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "outputs" / "reports").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (14, 6)
DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10

ENTITY_PALETTE = [
    "#2E75B6", "#C55A11", "#E91E90", "#E15759",
    "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2",
]
COLOR_OTHER = "#BAB0AC"

# ---------------------------------------------------------------------------
# Known entities (seed list — expanded by NER discovery)
# ---------------------------------------------------------------------------
# Seed list of known entities — loaded from config to keep names out of source.
# Format: {"CanonicalName": ["alias1", "alias2"]}
# Fall back to empty if config not found (NER discovery still works).
_entity_config = PROJECT_ROOT / "config" / "known_entities.json"
if _entity_config.exists():
    import json as _json
    with open(_entity_config, encoding="utf-8") as _f:
        _cfg = _json.load(_f)
    KNOWN_ENTITIES = _cfg.get("known_entities", {})
    EXCLUDE_ENTITIES = set(_cfg.get("exclude_entities", []))
else:
    KNOWN_ENTITIES = {}
    EXCLUDE_ENTITIES = set()

# Words that spaCy often tags as PERSON but are common English words
AMBIGUITY_STOPLIST = {
    # Pronouns and fragments spaCy misclassifies as PERSON
    "i", "s", "t", "m", "re", "d", "ll", "ve",
    # Common words that are also names
    "will", "grace", "art", "mark", "hope", "faith", "joy", "may",
    "dawn", "chance", "bill", "pat", "rob", "bob", "frank", "gene",
    "crystal", "amber", "ivy", "penny", "chase", "wade", "lance",
    "cliff", "glen", "dale", "brook", "reed", "drew", "grant",
    # Tech terms spaCy misclassifies
    "ai", "api", "sql", "rag", "llm", "mcp", "etl",
    "jira", "snowflake", "cyera", "metadata", "warehouse",
    # Other non-person noise
    "ill", "ya", "idk", "ngl", "kindof", "kinda", "emoji",
    "dick",  # unless a known person, this is noise
    "context", "lmk", "idfk", "tprm",
    # Common words spaCy tags as PERSON in conversational text
    "people", "lol", "lmao", "god", "ok", "okay", "hey", "hi", "thanks",
    "sorry", "yes", "no", "oh", "ah", "ugh", "hmm", "haha", "hahaha",
    "gonna", "wanna", "gotta", "kk", "omg", "smh", "tbh", "imo",
    # More false positives discovered in first runs
    "false", "null", "true", "girl", "scared", "happen", "rewrite",
    "unclear", "legal", "dot", "mac", "top datastores",
    "skull emoji", "dick mentioned", "mitm", "supernova",
    "copilot", "gemini", "openai", "instagram",
}

ROLE_FILTER = "user"
MIN_NER_MENTIONS_FOR_REGEX = 3
MIN_MENTIONS_FOR_TIMESERIES = 10
MIN_MENTIONS_FOR_CHANGEPOINT = 50
MULTI_WORD_COLLAPSE_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    return str(FIG_DIR / name)


def build_alias_lookup():
    """Build reverse-lookup: lowercase alias -> canonical name."""
    lookup = {}
    for canonical, aliases in KNOWN_ENTITIES.items():
        lookup[canonical.lower()] = canonical
        for alias in aliases:
            lookup[alias.lower()] = canonical
    return lookup


def normalize_entity_name(name, alias_lookup):
    """Normalize an entity name: strip possessives, title-case, alias lookup."""
    name = name.strip()
    # If name contains newlines or control chars, take only the first line
    if "\n" in name or "\r" in name:
        name = re.split(r"[\r\n]+", name)[0].strip()
    # Strip emoji and non-ASCII symbols
    name = re.sub(r"[^\w\s\-'.]+", "", name).strip()
    # Strip possessives
    if name.endswith("'s") or name.endswith("\u2019s"):
        name = name[:-2]
    # Strip trailing punctuation
    name = name.rstrip(".,;:!?-")
    name = name.strip()
    if not name or len(name) < 2:
        return ""
    # Check alias lookup first
    key = name.lower()
    if key in alias_lookup:
        return alias_lookup[key]
    # Title-case
    return name.title()


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data():
    print("Loading data...")
    conversations = pd.read_parquet(CONV_PATH)
    messages = pd.read_parquet(MSGS_PATH)

    conv = conversations[conversations["is_analysable"]].copy()
    msgs = messages[messages["conversation_id"].isin(conv["conversation_id"])].copy()

    # Filter to user messages only
    user_msgs = msgs[
        (msgs["role"] == ROLE_FILTER)
        & (msgs["text"].notna())
        & (msgs["text"].str.strip() != "")
    ].copy()

    print(f"  Conversations: {len(conv):,}")
    print(f"  User messages to scan: {len(user_msgs):,}")
    return conv, user_msgs


# ---------------------------------------------------------------------------
# Step 2: spaCy NER scan
# ---------------------------------------------------------------------------
def run_ner_scan(user_msgs):
    """Extract PERSON entities using spaCy NER."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 2_000_000
        print("  spaCy en_core_web_sm loaded")
    except Exception as e:
        print(f"  WARNING: spaCy not available ({e}), skipping NER scan")
        return [], False

    detections = []
    ner_errors = 0
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
                    if ent.label_ == "PERSON":
                        detections.append({
                            "conversation_id": cid,
                            "msg_index": int(midx),
                            "entity_raw": ent.text,
                            "char_start": ent.start_char,
                            "char_end": ent.end_char,
                            "method": "spacy_ner",
                        })
        except Exception as e:
            ner_errors += 1
            if ner_errors <= 5:
                print(f"  NER error in batch starting at {batch_start}: {e}")

    print(f"  NER PERSON detections: {len(detections):,}, errors: {ner_errors}")
    return detections, True


# ---------------------------------------------------------------------------
# Step 3: Discover entities & run regex scan
# ---------------------------------------------------------------------------
def discover_entities(ner_detections, alias_lookup):
    """Find all normalized entity names with enough NER mentions."""
    name_counts = Counter()
    for det in ner_detections:
        normalized = normalize_entity_name(det["entity_raw"], alias_lookup)
        if normalized and normalized.lower() not in AMBIGUITY_STOPLIST:
            name_counts[normalized] += 1

    # Keep entities with MIN_NER_MENTIONS_FOR_REGEX+ mentions
    discovered = {
        name for name, count in name_counts.items()
        if count >= MIN_NER_MENTIONS_FOR_REGEX
    }
    # Always include known entities
    for canonical in KNOWN_ENTITIES:
        discovered.add(canonical)

    print(f"  Discovered {len(discovered)} entities for regex pass")
    return discovered


def run_regex_scan(user_msgs, entity_names, alias_lookup):
    """Scan user messages with word-boundary regex for each entity name."""
    detections = []

    # Build compiled patterns
    patterns = {}
    for name in entity_names:
        # For multi-word names, match the full name
        escaped = re.escape(name)
        patterns[name] = re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    texts = user_msgs["text"].tolist()
    conv_ids = user_msgs["conversation_id"].tolist()
    msg_indices = user_msgs["msg_index"].tolist()

    for i in tqdm(range(len(texts)), desc="  Regex scan"):
        text = texts[i]
        cid = conv_ids[i]
        midx = msg_indices[i]
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                detections.append({
                    "conversation_id": cid,
                    "msg_index": int(midx),
                    "entity_raw": match.group(),
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "method": "regex",
                })

    print(f"  Regex detections: {len(detections):,}")
    return detections


# ---------------------------------------------------------------------------
# Step 4: Deduplicate & normalize
# ---------------------------------------------------------------------------
def deduplicate_and_normalize(ner_detections, regex_detections, alias_lookup):
    """Merge NER + regex, deduplicate overlapping spans, normalize names."""
    all_dets = []

    # Index NER detections by (cid, midx, start, end) for dedup
    ner_spans = set()
    for det in ner_detections:
        key = (det["conversation_id"], det["msg_index"], det["char_start"], det["char_end"])
        ner_spans.add(key)
        normalized = normalize_entity_name(det["entity_raw"], alias_lookup)
        if (normalized and normalized.lower() not in AMBIGUITY_STOPLIST
                and normalized.lower() not in EXCLUDE_ENTITIES):
            all_dets.append({
                "conversation_id": det["conversation_id"],
                "msg_index": det["msg_index"],
                "entity": normalized,
                "method": "spacy_ner",
            })

    # Add regex detections only if they don't overlap with NER
    regex_added = 0
    regex_skipped = 0
    for det in regex_detections:
        key = (det["conversation_id"], det["msg_index"], det["char_start"], det["char_end"])
        if key in ner_spans:
            regex_skipped += 1
            continue
        normalized = normalize_entity_name(det["entity_raw"], alias_lookup)
        if (normalized and normalized.lower() not in AMBIGUITY_STOPLIST
                and normalized.lower() not in EXCLUDE_ENTITIES):
            all_dets.append({
                "conversation_id": det["conversation_id"],
                "msg_index": det["msg_index"],
                "entity": normalized,
                "method": "regex",
            })
            regex_added += 1

    print(f"  After dedup: {len(all_dets):,} detections "
          f"(regex added: {regex_added:,}, regex overlaps skipped: {regex_skipped:,})")

    # Collapse multi-word variants to first name when the first name
    # exists as a standalone entity with significantly more mentions
    if all_dets:
        entity_counts = Counter(d["entity"] for d in all_dets)
        collapse_map = {}
        for name, count in entity_counts.items():
            parts = name.split()
            if len(parts) > 1:
                first = parts[0]
                first_count = entity_counts.get(first, 0)
                # Collapse if: first name has more mentions, OR multi-word has < 5
                if first_count > 0 and (first_count > count or count < MULTI_WORD_COLLAPSE_THRESHOLD):
                    collapse_map[name] = first
        if collapse_map:
            print(f"  Collapsing {len(collapse_map)} multi-word names")
            for det in all_dets:
                if det["entity"] in collapse_map:
                    det["entity"] = collapse_map[det["entity"]]

    return all_dets


# ---------------------------------------------------------------------------
# Step 5: Aggregate
# ---------------------------------------------------------------------------
def aggregate(detections, conv_df):
    """Build per-conversation and weekly/monthly aggregates."""
    if not detections:
        print("  WARNING: No detections to aggregate")
        empty_attention = pd.DataFrame(columns=[
            "conversation_id", "entity", "mention_count", "role_filter"])
        empty_weekly = pd.DataFrame(columns=[
            "year_week", "entity", "mention_count", "conversation_count"])
        return empty_attention, empty_weekly

    det_df = pd.DataFrame(detections)

    # Per-conversation
    attention = (
        det_df.groupby(["conversation_id", "entity"])
        .size()
        .reset_index(name="mention_count")
    )
    attention["mention_count"] = attention["mention_count"].astype("int32")
    attention["role_filter"] = ROLE_FILTER

    print(f"  Per-conversation rows: {len(attention):,}")
    print(f"  Unique entities: {attention['entity'].nunique()}")

    # Join with conversation timestamps
    conv_time = conv_df[["conversation_id", "year_week", "year_month"]].copy()
    att_with_time = attention.merge(conv_time, on="conversation_id", how="left")

    # Weekly
    weekly = (
        att_with_time.groupby(["year_week", "entity"])
        .agg(
            mention_count=("mention_count", "sum"),
            conversation_count=("conversation_id", "nunique"),
        )
        .reset_index()
    )
    weekly["mention_count"] = weekly["mention_count"].astype("int32")
    weekly["conversation_count"] = weekly["conversation_count"].astype("int32")

    # Monthly
    monthly = (
        att_with_time.groupby(["year_month", "entity"])
        .agg(
            mention_count=("mention_count", "sum"),
            conversation_count=("conversation_id", "nunique"),
        )
        .reset_index()
    )

    print(f"  Weekly rows: {len(weekly):,}, Monthly rows: {len(monthly):,}")
    return attention, weekly, monthly


# ---------------------------------------------------------------------------
# Step 6: Changepoint detection
# ---------------------------------------------------------------------------
def detect_changepoints(weekly_df, conv_df):
    """Run PELT changepoint detection on top entity weekly series."""
    try:
        import ruptures as rpt
        print("  ruptures loaded")
    except ImportError:
        print("  WARNING: ruptures not installed, skipping changepoint detection")
        return {}

    # Build full week range
    all_weeks = sorted(conv_df["year_week"].dropna().unique())
    if len(all_weeks) < 10:
        print("  WARNING: fewer than 10 weeks in data, skipping changepoints")
        return {}

    # Find entities with enough mentions
    entity_totals = weekly_df.groupby("entity")["mention_count"].sum()
    eligible = entity_totals[entity_totals >= MIN_MENTIONS_FOR_CHANGEPOINT].index.tolist()
    print(f"  Entities eligible for changepoint detection: {len(eligible)}")
    print(f"  Top 5 eligible: {eligible[:5]}")

    changepoints = {}
    for entity in eligible:
        entity_weekly = weekly_df[weekly_df["entity"] == entity].set_index("year_week")["mention_count"]
        # Reindex to full week range, fill missing with 0
        signal = entity_weekly.reindex(all_weeks, fill_value=0).values.astype(float)

        if len(signal) < 6:
            continue

        try:
            # Normalize signal to zero mean, unit variance for consistent penalty
            sig_std = signal.std()
            if sig_std > 0:
                normalized = ((signal - signal.mean()) / sig_std).reshape(-1, 1)
            else:
                continue
            algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(normalized)
            result = algo.predict(pen=3)
            # result always includes len(signal) as last element — remove it
            cps = [cp for cp in result if cp < len(signal)]

            entity_cps = []
            for cp in cps:
                before = signal[max(0, cp - 4):cp]
                after = signal[cp:min(len(signal), cp + 4)]
                before_mean = float(np.mean(before)) if len(before) > 0 else 0
                after_mean = float(np.mean(after)) if len(after) > 0 else 0
                direction = "increase" if after_mean > before_mean else "decrease"
                entity_cps.append({
                    "year_week": all_weeks[cp] if cp < len(all_weeks) else all_weeks[-1],
                    "index": int(cp),
                    "direction": direction,
                    "before_mean": round(before_mean, 2),
                    "after_mean": round(after_mean, 2),
                })

            if entity_cps:
                changepoints[entity] = entity_cps
                print(f"    {entity}: {len(entity_cps)} changepoint(s)")
            elif entity in top_entities[:3]:
                print(f"    {entity}: raw PELT result={result}, signal len={len(signal)}, "
                      f"max={signal.max():.0f}, mean={signal.mean():.1f}")
        except Exception as e:
            print(f"    {entity}: changepoint error: {e}")

    return changepoints


# ---------------------------------------------------------------------------
# Step 7: Figures
# ---------------------------------------------------------------------------
def generate_figures(attention_df, weekly_df, monthly_df, changepoints, conv_df):
    """Generate 7 analysis figures."""
    if attention_df.empty:
        print("  No data for figures, skipping")
        return []

    generated = []

    # Top entities by total mentions
    entity_totals = (
        attention_df.groupby("entity")["mention_count"]
        .sum()
        .sort_values(ascending=False)
    )
    top20 = entity_totals.head(20)
    top8_names = entity_totals.head(8).index.tolist()

    # --- Figure 1: Top entities bar chart ---
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        colors = [ENTITY_PALETTE[i % len(ENTITY_PALETTE)] if name in top8_names
                  else COLOR_OTHER for i, name in enumerate(top20.index)]
        ax.barh(range(len(top20)), top20.values, color=colors[:len(top20)])
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=TICK_SIZE)
        ax.invert_yaxis()
        ax.set_xlabel("Total Mentions", fontsize=LABEL_SIZE)
        ax.set_title("Top 20 Entities by Total Mentions", fontsize=TITLE_SIZE)
        plt.tight_layout()
        path = figpath("01_top_entities_bar.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append("01_top_entities_bar.png")
    except Exception as e:
        print(f"  Figure 1 error: {e}")

    # --- Figure 2: Weekly attention time series ---
    try:
        all_weeks = sorted(conv_df["year_week"].dropna().unique())
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for i, entity in enumerate(top8_names):
            ew = weekly_df[weekly_df["entity"] == entity].set_index("year_week")["mention_count"]
            ew = ew.reindex(all_weeks, fill_value=0)
            ax.plot(range(len(all_weeks)), ew.values,
                    label=entity, color=ENTITY_PALETTE[i % len(ENTITY_PALETTE)],
                    linewidth=1.5, alpha=0.85)
        # Show month labels instead of week codes
        from datetime import datetime as _dt
        tick_positions = []
        tick_labels = []
        seen_months = set()
        for idx, wk in enumerate(all_weeks):
            try:
                # Parse ISO week to get the month
                dt = _dt.strptime(wk + "-1", "%Y-W%W-%w")
                month_key = dt.strftime("%Y-%m")
                if month_key not in seen_months:
                    seen_months.add(month_key)
                    tick_positions.append(idx)
                    tick_labels.append(dt.strftime("%b %Y"))
            except ValueError:
                pass
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=TICK_SIZE - 1, ha="right")
        ax.set_ylabel("Weekly Mentions", fontsize=LABEL_SIZE)
        ax.set_title("Entity Attention Over Time (Weekly)", fontsize=TITLE_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper right")
        plt.tight_layout()
        path = figpath("02_weekly_attention_timeseries.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append("02_weekly_attention_timeseries.png")
    except Exception as e:
        print(f"  Figure 2 error: {e}")

    # --- Figure 3: Monthly attention heatmap ---
    try:
        top10_names = entity_totals.head(10).index.tolist()
        all_months = sorted(conv_df["year_month"].dropna().unique())
        heatmap_data = []
        for entity in top10_names:
            em = monthly_df[monthly_df["entity"] == entity].set_index("year_month")["mention_count"]
            em = em.reindex(all_months, fill_value=0)
            heatmap_data.append(em.values)
        heatmap_arr = np.array(heatmap_data)

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        sns.heatmap(heatmap_arr, ax=ax, cmap="YlOrRd",
                    xticklabels=all_months, yticklabels=top10_names,
                    annot=True, fmt="g", linewidths=0.5)
        ax.set_xlabel("Month", fontsize=LABEL_SIZE)
        ax.set_title("Monthly Entity Attention Heatmap", fontsize=TITLE_SIZE)
        plt.xticks(rotation=45, ha="right", fontsize=TICK_SIZE - 1)
        plt.yticks(fontsize=TICK_SIZE)
        plt.tight_layout()
        path = figpath("03_monthly_attention_heatmap.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append("03_monthly_attention_heatmap.png")
    except Exception as e:
        print(f"  Figure 3 error: {e}")

    # --- Figure 4: Top entity changepoints ---
    try:
        # Pick the top entity (by total mentions) that has changepoints
        top_entity_key = None
        for ent in entity_totals.head(5).index:
            if ent in changepoints:
                top_entity_key = ent
                break
        if top_entity_key is None and len(entity_totals) > 0:
            top_entity_key = entity_totals.index[0]

        if top_entity_key:
            all_weeks = sorted(conv_df["year_week"].dropna().unique())
            bw = weekly_df[weekly_df["entity"] == top_entity_key].set_index("year_week")["mention_count"]
            bw = bw.reindex(all_weeks, fill_value=0)

            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            ax.plot(range(len(all_weeks)), bw.values, color=ENTITY_PALETTE[0],
                    linewidth=1.5, label=top_entity_key)
            ax.fill_between(range(len(all_weeks)), bw.values,
                            alpha=0.15, color=ENTITY_PALETTE[0])

            # Mark changepoints
            if top_entity_key in changepoints:
                for cp in changepoints[top_entity_key]:
                    idx = cp["index"]
                    ax.axvline(x=idx, color="#E15759", linestyle="--", alpha=0.7)
                    ax.annotate(
                        f"{cp['direction']}\n{cp['before_mean']:.1f} -> {cp['after_mean']:.1f}",
                        xy=(idx, bw.values[min(idx, len(bw) - 1)]),
                        xytext=(10, 20), textcoords="offset points",
                        fontsize=8, color="#E15759",
                        arrowprops=dict(arrowstyle="->", color="#E15759", alpha=0.5),
                    )

            tick_step = max(1, len(all_weeks) // 12)
            ax.set_xticks(range(0, len(all_weeks), tick_step))
            ax.set_xticklabels([all_weeks[i] for i in range(0, len(all_weeks), tick_step)],
                               rotation=45, fontsize=TICK_SIZE - 1, ha="right")
            ax.set_ylabel("Weekly Mentions", fontsize=LABEL_SIZE)
            ax.set_title(f"{top_entity_key} Attention with Changepoints (PELT, l2, pen=3)",
                         fontsize=TITLE_SIZE)
            ax.legend(fontsize=TICK_SIZE)
            plt.tight_layout()
            path = figpath("04_top_entity_changepoints.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            generated.append("04_top_entity_changepoints.png")
    except Exception as e:
        print(f"  Figure 4 error: {e}")

    # --- Figure 5: Attention share stacked area ---
    try:
        all_months = sorted(conv_df["year_month"].dropna().unique())
        top5_names = entity_totals.head(5).index.tolist()

        stacked = {}
        for entity in top5_names:
            em = monthly_df[monthly_df["entity"] == entity].set_index("year_month")["mention_count"]
            stacked[entity] = em.reindex(all_months, fill_value=0).values.astype(float)

        # "Other" bucket
        all_monthly_totals = monthly_df.groupby("year_month")["mention_count"].sum()
        all_totals = all_monthly_totals.reindex(all_months, fill_value=0).values.astype(float)
        top5_sum = sum(stacked.values())
        stacked["Other"] = np.maximum(all_totals - top5_sum, 0)

        # Convert to proportions
        total_per_month = sum(stacked.values())
        total_per_month = np.where(total_per_month == 0, 1, total_per_month)  # avoid /0
        proportions = {k: v / total_per_month for k, v in stacked.items()}

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        bottom = np.zeros(len(all_months))
        labels = list(proportions.keys())
        for i, label in enumerate(labels):
            color = ENTITY_PALETTE[i % len(ENTITY_PALETTE)] if label != "Other" else COLOR_OTHER
            ax.fill_between(range(len(all_months)), bottom, bottom + proportions[label],
                            label=label, color=color, alpha=0.8)
            bottom += proportions[label]

        ax.set_xticks(range(len(all_months)))
        ax.set_xticklabels(all_months, rotation=45, fontsize=TICK_SIZE - 1, ha="right")
        ax.set_ylabel("Share of Mentions", fontsize=LABEL_SIZE)
        ax.set_ylim(0, 1)
        ax.set_title("Entity Attention Share Over Time", fontsize=TITLE_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper right")
        plt.tight_layout()
        path = figpath("05_attention_share_stacked.png")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        generated.append("05_attention_share_stacked.png")
    except Exception as e:
        print(f"  Figure 5 error: {e}")

    # --- Figure 6: Entity discovery timeline ---
    try:
        att_with_time = attention_df.merge(
            conv_df[["conversation_id", "created_at"]], on="conversation_id", how="left"
        )
        top12_names = entity_totals.head(12).index.tolist()
        timeline_data = []
        for entity in top12_names:
            ent_times = att_with_time[att_with_time["entity"] == entity]["created_at"]
            if len(ent_times) > 0:
                timeline_data.append({
                    "entity": entity,
                    "first": ent_times.min(),
                    "last": ent_times.max(),
                    "mentions": int(entity_totals[entity]),
                })

        if timeline_data:
            fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
            for i, td in enumerate(timeline_data):
                color = ENTITY_PALETTE[i % len(ENTITY_PALETTE)]
                ax.barh(i, (td["last"] - td["first"]).total_seconds() / 86400,
                        left=td["first"], height=0.6, color=color, alpha=0.7)
                ax.text(td["last"], i, f"  {td['mentions']}", va="center",
                        fontsize=TICK_SIZE - 1, color=color)
            ax.set_yticks(range(len(timeline_data)))
            ax.set_yticklabels([td["entity"] for td in timeline_data], fontsize=TICK_SIZE)
            ax.invert_yaxis()
            ax.set_xlabel("Date", fontsize=LABEL_SIZE)
            ax.set_title("Entity Active Windows (First to Last Mention)", fontsize=TITLE_SIZE)
            fig.autofmt_xdate()
            plt.tight_layout()
            path = figpath("06_entity_discovery_timeline.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            generated.append("06_entity_discovery_timeline.png")
    except Exception as e:
        print(f"  Figure 6 error: {e}")

    # --- Figure 7: Changepoints for all eligible entities ---
    try:
        cp_entities = list(changepoints.keys())[:6]
        if cp_entities:
            n_panels = len(cp_entities)
            ncols = min(3, n_panels)
            nrows = (n_panels + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            if n_panels == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            all_weeks = sorted(conv_df["year_week"].dropna().unique())
            for i, entity in enumerate(cp_entities):
                ax = axes[i]
                ew = weekly_df[weekly_df["entity"] == entity].set_index("year_week")["mention_count"]
                ew = ew.reindex(all_weeks, fill_value=0)
                ax.plot(range(len(all_weeks)), ew.values,
                        color=ENTITY_PALETTE[i % len(ENTITY_PALETTE)], linewidth=1.2)
                for cp in changepoints[entity]:
                    ax.axvline(x=cp["index"], color="#E15759", linestyle="--", alpha=0.6)
                ax.set_title(entity, fontsize=LABEL_SIZE)
                ax.tick_params(labelsize=TICK_SIZE - 2)
                # Sparse x labels
                tick_step = max(1, len(all_weeks) // 6)
                ax.set_xticks(range(0, len(all_weeks), tick_step))
                ax.set_xticklabels([all_weeks[j] for j in range(0, len(all_weeks), tick_step)],
                                   rotation=45, fontsize=7, ha="right")

            # Hide unused panels
            for j in range(len(cp_entities), len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("Changepoint Detection (PELT) — Top Entities", fontsize=TITLE_SIZE)
            plt.tight_layout()
            path = figpath("07_changepoints_all_entities.png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            generated.append("07_changepoints_all_entities.png")
    except Exception as e:
        print(f"  Figure 7 error: {e}")

    return generated


# ---------------------------------------------------------------------------
# Step 8: Report
# ---------------------------------------------------------------------------
def build_report(attention_df, weekly_df, conv_df, changepoints,
                 ner_count, regex_count, spacy_available, figures):
    """Build and save JSON report."""
    if attention_df.empty:
        entity_totals = pd.Series(dtype="int64")
    else:
        entity_totals = attention_df.groupby("entity")["mention_count"].sum().sort_values(ascending=False)

    # Top entities detail
    top_entities = {}
    for entity in entity_totals.head(10).index:
        ew = weekly_df[weekly_df["entity"] == entity]
        att = attention_df[attention_df["entity"] == entity]
        att_with_time = att.merge(conv_df[["conversation_id", "year_week"]], on="conversation_id", how="left")
        weeks_present = att_with_time["year_week"].dropna()
        top_entities[entity] = {
            "total_mentions": int(entity_totals[entity]),
            "conversations": int(att["conversation_id"].nunique()),
            "first_week": str(weeks_present.min()) if len(weeks_present) > 0 else None,
            "last_week": str(weeks_present.max()) if len(weeks_present) > 0 else None,
            "peak_week": str(ew.loc[ew["mention_count"].idxmax(), "year_week"]) if len(ew) > 0 else None,
            "peak_mentions": int(ew["mention_count"].max()) if len(ew) > 0 else 0,
        }

    # Attention dynamics
    total_mentions = int(entity_totals.sum()) if len(entity_totals) > 0 else 0
    top1_entity = entity_totals.index[0] if len(entity_totals) > 0 else None
    top1_mentions = int(entity_totals.iloc[0]) if len(entity_totals) > 0 else 0

    report = {
        "module": "entity_attention",
        "module_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_data": {
            "conversations_analysed": int(len(conv_df)),
            "messages_scanned": int(attention_df["conversation_id"].nunique()) if not attention_df.empty else 0,
            "role_filter": ROLE_FILTER,
            "date_range": [
                str(conv_df["created_at"].min()),
                str(conv_df["created_at"].max()),
            ],
            "spacy_available": spacy_available,
            "ruptures_available": "ruptures" in sys.modules,
        },
        "entity_discovery": {
            "total_unique_entities": int(entity_totals.nunique()) if len(entity_totals) > 0 else 0,
            "entities_with_10plus_mentions": int((entity_totals >= 10).sum()),
            "total_detections": total_mentions,
            "ner_detections": ner_count,
            "regex_detections": regex_count,
        },
        "top_entities": top_entities,
        "changepoints": clean_dict(changepoints),
        "attention_dynamics": {
            "top1_entity": top1_entity,
            "top1_share_overall": round(top1_mentions / total_mentions, 3) if total_mentions > 0 else 0,
            "top5_share_overall": round(
                int(entity_totals.head(5).sum()) / total_mentions, 3
            ) if total_mentions > 0 else 0,
        },
        "figures_generated": figures,
    }

    report = clean_dict(report)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    return report


# ---------------------------------------------------------------------------
# Step 9: Validation
# ---------------------------------------------------------------------------
def run_validation(attention_df, weekly_df, conv_df, changepoints, figures):
    """Run validation checks and print results."""
    print("\n=== Validation ===")
    checks = []

    # 1. Parquet exists and loads
    ok = OUT_PARQUET.exists()
    checks.append(("entity_attention.parquet exists", ok))

    # 2. Weekly parquet exists
    ok = OUT_WEEKLY.exists()
    checks.append(("entity_attention_weekly.parquet exists", ok))

    # 3. All conversation_ids exist in conversations_clean
    if not attention_df.empty:
        valid_ids = set(conv_df["conversation_id"])
        orphans = set(attention_df["conversation_id"]) - valid_ids
        checks.append(("All conversation_ids valid", len(orphans) == 0))
    else:
        checks.append(("All conversation_ids valid", True))

    # 4. Top entity from config appears in top 3
    if not attention_df.empty:
        entity_totals = attention_df.groupby("entity")["mention_count"].sum().sort_values(ascending=False)
        top3 = entity_totals.head(3).index.tolist()
        top_known = list(KNOWN_ENTITIES.keys())[:1]
        if top_known:
            checks.append((f"Top known entity in top 3", top_known[0] in top3))
        else:
            checks.append(("At least 3 entities detected", len(entity_totals) >= 3))
    else:
        checks.append(("Top known entity in top 3", False))

    # 5. Unique entities > 10
    if not attention_df.empty:
        n_entities = attention_df["entity"].nunique()
        checks.append((f"Unique entities > 10 (got {n_entities})", n_entities > 10))
    else:
        checks.append(("Unique entities > 10", False))

    # 6. Weekly series covers 40+ weeks
    if not weekly_df.empty:
        n_weeks = weekly_df["year_week"].nunique()
        checks.append((f"Weekly series >= 40 weeks (got {n_weeks})", n_weeks >= 40))
    else:
        checks.append(("Weekly series >= 40 weeks", False))

    # 7. No NaN in mention_count
    if not attention_df.empty:
        ok = attention_df["mention_count"].isna().sum() == 0
        checks.append(("No NaN in mention_count", ok))
    else:
        checks.append(("No NaN in mention_count", True))

    # 8. All figure PNGs exist
    ok = all((FIG_DIR / f).exists() for f in figures) if figures else False
    checks.append((f"All {len(figures)} figures exist", ok))

    # 9. Changepoint detection ran for top entity
    top_known = list(KNOWN_ENTITIES.keys())[:1]
    if top_known:
        ok = top_known[0] in changepoints
        checks.append(("Changepoint detection ran for top entity", ok))
    else:
        ok = len(changepoints) > 0
        checks.append(("Changepoint detection ran", ok))

    # 10. Report exists
    ok = OUT_REPORT.exists()
    checks.append(("Report JSON exists", ok))

    passed = 0
    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if ok:
            passed += 1
    print(f"\n  {passed}/{len(checks)} checks passed")
    return passed, len(checks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Module 22: Entity Attention Over Time")
    print("=" * 60)

    # Step 0: Load
    conv_df, user_msgs = load_data()

    # Step 1: Build alias lookup
    alias_lookup = build_alias_lookup()

    # Step 2: NER scan
    print("\n--- Step 2: spaCy NER Scan ---")
    ner_detections, spacy_available = run_ner_scan(user_msgs)

    # Step 3: Discover entities & regex scan
    print("\n--- Step 3: Entity Discovery & Regex Scan ---")
    discovered_entities = discover_entities(ner_detections, alias_lookup)
    regex_detections = run_regex_scan(user_msgs, discovered_entities, alias_lookup)

    # Step 4: Deduplicate & normalize
    print("\n--- Step 4: Deduplicate & Normalize ---")
    all_detections = deduplicate_and_normalize(ner_detections, regex_detections, alias_lookup)

    # Step 5: Aggregate
    print("\n--- Step 5: Aggregate ---")
    attention_df, weekly_df, monthly_df = aggregate(all_detections, conv_df)

    # Save parquets
    attention_df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET}")
    weekly_df.to_parquet(OUT_WEEKLY, index=False)
    print(f"  Saved: {OUT_WEEKLY}")

    # Step 6: Changepoint detection
    print("\n--- Step 6: Changepoint Detection (PELT) ---")
    changepoints = detect_changepoints(weekly_df, conv_df)

    # Step 7: Figures
    print("\n--- Step 7: Generating Figures ---")
    figures = generate_figures(attention_df, weekly_df, monthly_df, changepoints, conv_df)
    print(f"  Generated {len(figures)} figures")

    # Step 8: Report
    print("\n--- Step 8: Report ---")
    ner_count = len(ner_detections)
    regex_count = len(regex_detections)
    report = build_report(attention_df, weekly_df, conv_df, changepoints,
                          ner_count, regex_count, spacy_available, figures)

    # Print summary
    print("\n=== Summary ===")
    if not attention_df.empty:
        entity_totals = attention_df.groupby("entity")["mention_count"].sum().sort_values(ascending=False)
        print(f"  Total unique entities: {attention_df['entity'].nunique()}")
        print(f"  Total mentions: {entity_totals.sum():,}")
        print(f"  Top 10 entities:")
        for name, count in entity_totals.head(10).items():
            pct = count / entity_totals.sum() * 100
            print(f"    {name:20s} {count:6,} ({pct:.1f}%)")

    # Step 9: Validation
    run_validation(attention_df, weekly_df, conv_df, changepoints, figures)

    print("\n=== Module 22 Complete ===")


if __name__ == "__main__":
    main()
