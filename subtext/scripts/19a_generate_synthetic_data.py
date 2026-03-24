"""
Module 19a: Synthetic Data Generator for Influence Prediction Pipeline

Generates realistic synthetic data for all upstream parquet files so that
Module 19 (influence_prediction.py) can train and evaluate end-to-end
without requiring actual LLM classification runs.

The generator embeds realistic correlations:
  - Longer conversations drift toward more "adopt" frames
  - Emotional processing conversations have higher reject/redirect rates
  - Late-night conversations show more vulnerability → higher adopt rates
  - Code conversations have near-zero emotional engagement → more steer/ignore
  - Within-conversation frame sequences are auto-correlated (momentum)
  - Token ratios vary by functional category (code = long assistant replies)

Usage:
    python scripts/19a_generate_synthetic_data.py

    # Control scale:
    python scripts/19a_generate_synthetic_data.py --conversations 5000
    python scripts/19a_generate_synthetic_data.py --conversations 20000

    # Seed for reproducibility:
    python scripts/19a_generate_synthetic_data.py --seed 42
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# -- Constants ---------------------------------------------------------------
VALID_FRAMES = ["adopt", "extend", "redirect", "reject", "ignore", "steer"]

FUNCTIONS = [
    "interpersonal_analysis", "emotional_processing", "creative_expression",
    "career_strategy", "self_modeling", "practical", "learning",
    "problem_solving", "coding", "social_rehearsal", "work_professional",
    "planning",
]

EMOTIONS = [
    "analytical", "anxious", "curious", "frustrated", "grieving", "playful",
    "reflective", "strategic", "vulnerable", "energized", "numb", "determined",
]

OPENING_TYPES = ["question", "command", "greeting", "statement", "code", "fragment"]

SHAPE_ARCHETYPES = [
    "front_loaded", "back_loaded", "balanced", "spike", "plateau", "fade_out",
]

MODEL_ERAS = [
    "gpt35_era", "gpt4_era", "gpt4_turbo_era", "gpt4o_era", "o1_era",
]

TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]

# -- Frame probability profiles per function ---------------------------------
# Each function has a different "personality" for frame adoption
FRAME_PROFILES = {
    #                      adopt  extend redirect reject ignore steer
    "interpersonal_analysis": [0.30, 0.15, 0.15,   0.10,  0.05,  0.25],
    "emotional_processing":   [0.25, 0.10, 0.20,   0.15,  0.05,  0.25],
    "creative_expression":    [0.20, 0.25, 0.15,   0.05,  0.05,  0.30],
    "career_strategy":        [0.35, 0.15, 0.15,   0.05,  0.05,  0.25],
    "self_modeling":          [0.30, 0.20, 0.15,   0.10,  0.05,  0.20],
    "practical":              [0.35, 0.10, 0.10,   0.05,  0.10,  0.30],
    "learning":               [0.40, 0.15, 0.10,   0.05,  0.05,  0.25],
    "problem_solving":        [0.30, 0.15, 0.15,   0.10,  0.05,  0.25],
    "coding":                 [0.25, 0.10, 0.05,   0.10,  0.15,  0.35],
    "social_rehearsal":       [0.25, 0.15, 0.20,   0.10,  0.05,  0.25],
    "work_professional":      [0.35, 0.10, 0.10,   0.05,  0.10,  0.30],
    "planning":               [0.30, 0.15, 0.15,   0.05,  0.05,  0.30],
}

# Emotion modifiers: these shift frame probabilities
# positive = more adopt/extend, negative = more redirect/reject
EMOTION_INFLUENCE = {
    "analytical":  0.0,
    "anxious":    -0.10,
    "curious":     0.15,
    "frustrated": -0.20,
    "grieving":   -0.05,
    "playful":     0.10,
    "reflective":  0.05,
    "strategic":   0.10,
    "vulnerable":  0.05,
    "energized":   0.10,
    "numb":       -0.15,
    "determined":  0.05,
}


def generate_conversations(n_conv, rng):
    """Generate conversations_clean.parquet with realistic distributions."""
    print(f"\n  Generating {n_conv:,} conversations...")

    # Conversation lengths follow a power-law-ish distribution
    # Most conversations are short, some are very long
    msg_counts = rng.lognormal(mean=2.5, sigma=0.8, size=n_conv).astype(int).clip(2, 200)

    # Timestamps spanning ~2 years
    start = datetime(2023, 3, 1, tzinfo=timezone.utc)
    offsets = rng.uniform(0, 365 * 2, size=n_conv)
    created_at = [start + timedelta(days=d, hours=rng.uniform(0, 24)) for d in offsets]

    # Function assignment (weighted — more practical/learning, less grieving)
    func_weights = np.array([0.12, 0.08, 0.07, 0.08, 0.06, 0.15,
                              0.10, 0.10, 0.08, 0.04, 0.07, 0.05])
    func_weights /= func_weights.sum()
    functions = rng.choice(FUNCTIONS, size=n_conv, p=func_weights)

    # Emotion assignment (correlated with function)
    emotions = []
    for func in functions:
        if func == "coding":
            probs = [0.40, 0.05, 0.10, 0.15, 0.00, 0.02, 0.03, 0.10, 0.00, 0.10, 0.00, 0.05]
        elif func == "emotional_processing":
            probs = [0.05, 0.20, 0.05, 0.10, 0.15, 0.02, 0.20, 0.03, 0.15, 0.02, 0.02, 0.01]
        elif func == "creative_expression":
            probs = [0.05, 0.02, 0.20, 0.02, 0.01, 0.30, 0.05, 0.05, 0.03, 0.20, 0.02, 0.05]
        elif func in ("career_strategy", "work_professional"):
            probs = [0.15, 0.10, 0.10, 0.05, 0.01, 0.02, 0.10, 0.25, 0.05, 0.10, 0.02, 0.05]
        else:
            probs = [0.15, 0.08, 0.15, 0.08, 0.03, 0.08, 0.12, 0.10, 0.05, 0.08, 0.03, 0.05]
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()
        emotions.append(rng.choice(EMOTIONS, p=probs))

    # Hour of day — bimodal (morning peak, evening peak)
    hours = np.concatenate([
        rng.normal(10, 2, size=n_conv // 2),
        rng.normal(21, 2, size=n_conv - n_conv // 2),
    ])
    rng.shuffle(hours)
    hours = np.clip(hours, 0, 23).astype(int)

    days = rng.randint(0, 7, size=n_conv)

    # Code presence — correlated with function
    has_code = np.array([
        rng.random() < (0.85 if f == "coding" else 0.30 if f == "problem_solving" else 0.05)
        for f in functions
    ])

    # Duration — correlated with message count
    duration = msg_counts * rng.uniform(0.5, 5.0, size=n_conv)

    # User/assistant token totals
    user_tok_per_msg = np.where(
        np.isin(functions, ["coding", "problem_solving"]),
        rng.lognormal(5.0, 0.6, size=n_conv),
        rng.lognormal(4.0, 0.7, size=n_conv),
    )
    asst_tok_per_msg = np.where(
        np.isin(functions, ["coding", "creative_expression"]),
        rng.lognormal(5.5, 0.5, size=n_conv),
        rng.lognormal(4.8, 0.6, size=n_conv),
    )

    user_msg_counts = (msg_counts / 2).astype(int).clip(1)
    asst_msg_counts = msg_counts - user_msg_counts

    user_token_total = (user_tok_per_msg * user_msg_counts).astype(int)
    asst_token_total = (asst_tok_per_msg * asst_msg_counts).astype(int)

    # Model era — time-based
    era_idx = np.digitize(offsets, [0, 180, 300, 450, 600]) - 1
    era_idx = np.clip(era_idx, 0, len(MODEL_ERAS) - 1)
    model_eras = [MODEL_ERAS[i] for i in era_idx]

    # Time of day category
    tod = np.array([
        "morning" if 6 <= h < 12 else "afternoon" if 12 <= h < 17
        else "evening" if 17 <= h < 22 else "night"
        for h in hours
    ])

    conv_ids = [f"conv_{i:06d}" for i in range(n_conv)]

    # Opening types
    openings = rng.choice(OPENING_TYPES, size=n_conv, p=[0.30, 0.25, 0.10, 0.15, 0.10, 0.10])

    # Topics (integer clusters 0-14)
    n_topics = 15
    topics = rng.randint(0, n_topics, size=n_conv)

    # Shape archetypes
    shapes = rng.choice(SHAPE_ARCHETYPES, size=n_conv)

    conv_df = pd.DataFrame({
        "conversation_id":     conv_ids,
        "created_at":          created_at,
        "msg_count":           msg_counts,
        "user_msg_count":      user_msg_counts,
        "assistant_msg_count": asst_msg_counts,
        "user_token_total":    user_token_total,
        "assistant_token_total": asst_token_total,
        "user_token_ratio":    user_token_total / np.clip(asst_token_total, 1, None),
        "duration_minutes":    duration.round(1),
        "hour_of_day":         hours,
        "day_of_week":         days,
        "has_code":            has_code,
        "is_branched":         rng.random(n_conv) < 0.05,
        "is_analysable":       True,
        "model_era":           model_eras,
        "time_of_day":         tod,
        "turns":               msg_counts,
        "year_month":          [d.strftime("%Y-%m") for d in created_at],
        "conversation_type":   "standard",
        "first_user_message_type": openings,
    })

    # Metadata for downstream use
    meta = {
        "functions": dict(zip(conv_ids, functions)),
        "emotions":  dict(zip(conv_ids, emotions)),
        "topics":    dict(zip(conv_ids, topics.tolist())),
        "shapes":    dict(zip(conv_ids, shapes)),
        "openings":  dict(zip(conv_ids, openings)),
    }

    return conv_df, meta


def generate_messages(conv_df, rng):
    """Generate messages_clean.parquet."""
    print("  Generating messages...")
    rows = []
    for _, conv in conv_df.iterrows():
        cid = conv["conversation_id"]
        n_msgs = conv["msg_count"]
        user_toks_avg = conv["user_token_total"] / max(conv["user_msg_count"], 1)
        asst_toks_avg = conv["assistant_token_total"] / max(conv["assistant_msg_count"], 1)

        for i in range(n_msgs):
            role = "user" if i % 2 == 0 else "assistant"
            tok_avg = user_toks_avg if role == "user" else asst_toks_avg
            token_count = max(1, int(rng.lognormal(np.log(max(tok_avg, 1)), 0.4)))

            rows.append({
                "conversation_id":     cid,
                "msg_index":           i,
                "role":                role,
                "text":                f"[synthetic {role} message {i}]",
                "token_count":         token_count,
                "position_in_conversation": i / max(n_msgs - 1, 1),
                "inter_msg_seconds":   float(rng.exponential(60)) if i > 0 else 0.0,
                "is_first_user_msg":   (i == 0 and role == "user"),
            })

    return pd.DataFrame(rows)


def generate_frame_adoption(conv_df, msgs_df, meta, rng):
    """
    Generate frame_adoption.parquet with realistic correlated labels.

    Key correlations embedded:
    - First user message → always "steer" (rule-based)
    - Function profile shapes base probabilities
    - Emotion modifies adopt/reject balance
    - Late night → slightly more adopt (vulnerability)
    - Within-conversation momentum (frames are auto-correlated)
    - Longer conversations → drift toward adopt (rapport building)
    - Code conversations → more steer/ignore
    """
    print("  Generating frame adoption labels...")

    user_msgs = msgs_df[msgs_df["role"] == "user"].copy()
    rows = []

    for cid, group in user_msgs.groupby("conversation_id"):
        func = meta["functions"].get(cid, "practical")
        emotion = meta["emotions"].get(cid, "analytical")
        conv_row = conv_df[conv_df["conversation_id"] == cid].iloc[0]
        hour = conv_row["hour_of_day"]
        n_total = conv_row["msg_count"]

        base_probs = np.array(FRAME_PROFILES.get(func, FRAME_PROFILES["practical"]),
                              dtype=float)

        # Emotion modifier: shift adopt/extend up or redirect/reject up
        em_shift = EMOTION_INFLUENCE.get(emotion, 0.0)
        base_probs[0] += em_shift * 0.5   # adopt
        base_probs[1] += em_shift * 0.3   # extend
        base_probs[2] -= em_shift * 0.3   # redirect
        base_probs[3] -= em_shift * 0.4   # reject

        # Late night modifier
        if hour >= 22 or hour <= 4:
            base_probs[0] += 0.05  # more adopt
            base_probs[1] += 0.03  # more extend
            base_probs[3] -= 0.05  # less reject

        # Ensure valid
        base_probs = np.clip(base_probs, 0.01, None)
        base_probs /= base_probs.sum()

        prev_frame = None
        frames_so_far = []

        for _, msg_row in group.iterrows():
            msg_idx = msg_row["msg_index"]
            is_first = msg_row.get("is_first_user_msg", msg_idx == 0)

            if is_first:
                # Rule: conversation opener → steer
                rows.append({
                    "conversation_id":           cid,
                    "message_index":             msg_idx,
                    "frame_adoption":            "steer",
                    "frame_confidence":          float("nan"),
                    "classification_method":     "rule",
                    "is_conversation_opener":    True,
                    "user_tokens":               msg_row["token_count"],
                    "assistant_tokens_preceding": 0,
                    "input_tokens":              0,
                    "output_tokens":             0,
                })
                prev_frame = "steer"
                frames_so_far.append("steer")
                continue

            # Dynamic probability adjustments
            probs = base_probs.copy()

            # Momentum: if previous frame was adopt/extend, boost adopt/extend
            if prev_frame in ("adopt", "extend"):
                probs[0] += 0.08  # adopt momentum
                probs[1] += 0.04  # extend momentum
            elif prev_frame in ("redirect", "reject"):
                probs[2] += 0.06  # redirect momentum
                probs[3] += 0.04  # reject momentum

            # Conversation depth: later turns → slightly more adopt (rapport)
            position = msg_idx / max(n_total - 1, 1)
            probs[0] += position * 0.10   # adopt grows
            probs[5] -= position * 0.05   # steer shrinks

            # Accumulated adopt rate matters (entrainment effect)
            if len(frames_so_far) >= 3:
                adopt_rate = sum(1 for f in frames_so_far if f in ("adopt", "extend")) / len(frames_so_far)
                probs[0] += adopt_rate * 0.08
                probs[1] += adopt_rate * 0.04

            # Ensure valid
            probs = np.clip(probs, 0.01, None)
            probs /= probs.sum()

            frame = rng.choice(VALID_FRAMES, p=probs)
            confidence = float(np.clip(rng.beta(8, 2), 0.5, 1.0))

            # Find preceding assistant message
            preceding_msgs = msgs_df[
                (msgs_df["conversation_id"] == cid) &
                (msgs_df["msg_index"] < msg_idx) &
                (msgs_df["role"] == "assistant")
            ]
            asst_tokens = int(preceding_msgs["token_count"].iloc[-1]) if len(preceding_msgs) > 0 else 0

            rows.append({
                "conversation_id":           cid,
                "message_index":             msg_idx,
                "frame_adoption":            frame,
                "frame_confidence":          confidence,
                "classification_method":     "llm",
                "is_conversation_opener":    False,
                "user_tokens":               msg_row["token_count"],
                "assistant_tokens_preceding": asst_tokens,
                "input_tokens":              rng.randint(200, 800),
                "output_tokens":             rng.randint(15, 40),
            })

            prev_frame = frame
            frames_so_far.append(frame)

    df = pd.DataFrame(rows)
    df["frame_adoption"] = df["frame_adoption"].astype("category")
    df["classification_method"] = df["classification_method"].astype("category")
    return df


def generate_auxiliary_parquets(conv_df, meta):
    """Generate the optional upstream parquet files."""
    print("  Generating auxiliary classification files...")
    conv_ids = conv_df["conversation_id"].tolist()

    # functional_classifications.parquet
    func_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "function":        [meta["functions"][c] for c in conv_ids],
        "function_primary": [meta["functions"][c] for c in conv_ids],
    })

    # emotional_states.parquet
    emot_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "emotion":         [meta["emotions"][c] for c in conv_ids],
        "emotion_primary": [meta["emotions"][c] for c in conv_ids],
    })

    # topic_assignments.parquet
    topic_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "topic":           [meta["topics"][c] for c in conv_ids],
    })

    # shape_clusters.parquet
    shape_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "cluster_label":   [meta["shapes"][c] for c in conv_ids],
        "shape_archetype": [meta["shapes"][c] for c in conv_ids],
    })

    # opening_classifications.parquet
    open_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "opening_l1":      [meta["openings"][c] for c in conv_ids],
    })

    return func_df, emot_df, topic_df, shape_df, open_df


def save_all(conv_df, msgs_df, frames_df, func_df, emot_df, topic_df, shape_df, open_df):
    """Save all parquet files."""
    print("\n  Saving parquet files...")

    files = {
        "conversations_clean.parquet":       conv_df,
        "messages_clean.parquet":            msgs_df,
        "frame_adoption.parquet":            frames_df,
        "functional_classifications.parquet": func_df,
        "emotional_states.parquet":          emot_df,
        "topic_assignments.parquet":         topic_df,
        "shape_clusters.parquet":            shape_df,
        "opening_classifications.parquet":   open_df,
    }

    for name, df in files.items():
        path = os.path.join(OUT_DIR, name)
        df.to_parquet(path, index=False)
        print(f"    {name:45s} {len(df):>8,} rows, {len(df.columns):>2} cols")


def print_summary(conv_df, msgs_df, frames_df):
    """Print distribution summaries."""
    print("\n== Distribution Summary =================================================")

    # Frame distribution
    frame_counts = frames_df["frame_adoption"].value_counts()
    print("\n  Frame adoption distribution:")
    for frame in VALID_FRAMES:
        count = frame_counts.get(frame, 0)
        pct = count / len(frames_df) * 100
        bar = "#" * int(pct)
        print(f"    {frame:12s} {count:>7,}  ({pct:5.1f}%)  {bar}")

    # Method distribution
    method_counts = frames_df["classification_method"].value_counts()
    print(f"\n  Classification method:")
    for method, count in method_counts.items():
        print(f"    {method:12s} {count:>7,}")

    # Conversation length stats
    print(f"\n  Conversation length:")
    mc = conv_df["msg_count"]
    print(f"    Mean: {mc.mean():.1f}, Median: {mc.median():.0f}, "
          f"Max: {mc.max()}, Min: {mc.min()}")

    # Function distribution
    func_counts = pd.Series([v for v in frames_df.merge(
        conv_df[["conversation_id"]], on="conversation_id"
    )["frame_adoption"]])
    print(f"\n  Total messages: {len(msgs_df):,}")
    print(f"  Total user messages with labels: {len(frames_df):,}")
    print(f"  Conversations: {len(conv_df):,}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for influence prediction")
    parser.add_argument("--conversations", type=int, default=10000,
                        help="Number of conversations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("MODULE 19a: SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    print(f"  Conversations: {args.conversations:,}")
    print(f"  Seed: {args.seed}")

    rng = np.random.RandomState(args.seed)

    conv_df, meta = generate_conversations(args.conversations, rng)
    msgs_df = generate_messages(conv_df, rng)
    frames_df = generate_frame_adoption(conv_df, msgs_df, meta, rng)
    func_df, emot_df, topic_df, shape_df, open_df = generate_auxiliary_parquets(conv_df, meta)

    save_all(conv_df, msgs_df, frames_df, func_df, emot_df, topic_df, shape_df, open_df)
    print_summary(conv_df, msgs_df, frames_df)

    print("\nDone! Run Module 19 now:")
    print("  python scripts/19_influence_prediction.py")
    print("  python scripts/19_influence_prediction.py --binary")


if __name__ == "__main__":
    main()
