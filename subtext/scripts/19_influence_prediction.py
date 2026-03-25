"""
Module 4.1: Influence Prediction via Machine Learning
Script: 19_influence_prediction.py

Trains a classifier to predict how a user will respond to an AI frame
(adopt/extend/redirect/reject/ignore/steer) using features derived from
upstream modules — turn dynamics, conversation context, emotional state,
functional category, temporal patterns, and within-conversation frame history.

This module answers: "Given everything we know about this conversation so far,
can we predict whether the user will accept, push back against, or redirect
the AI's framing?"

Features (4 groups):
  1. Message-level: user token count, preceding assistant token count,
     turn position (msg_index / total turns), relative position in conversation
  2. Conversation-level: msg_count, token ratios, duration, hour_of_day,
     day_of_week, has_code, functional category, emotional state, topic
  3. Sequence: prior frame history within conversation (rolling adopt rate,
     last N frames one-hot encoded, streak length of current frame pattern)
  4. Interaction: user_tokens / assistant_tokens ratio, position × function

Models:
  - Baseline: Stratified dummy (majority class)
  - Primary: LightGBM gradient-boosted classifier
  - Evaluation: Stratified 5-fold CV, classification report, confusion matrix,
    SHAP feature importance, per-class precision/recall

Outputs:
  - influence_prediction_report.json (metrics, feature importance, per-class)
  - confusion_matrix.png, feature_importance.png, roc_curves.png
  - class_probability_calibration.png
  - Serialized model (joblib) for downstream use

Usage:
    python scripts/19_influence_prediction.py

    # Use fewer CV folds for speed:
    python scripts/19_influence_prediction.py --folds 3

    # Binary mode (adopt+extend vs redirect+reject+ignore+steer):
    python scripts/19_influence_prediction.py --binary

    # Skip SHAP (faster):
    python scripts/19_influence_prediction.py --skip-shap
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
    accuracy_score,
    f1_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import joblib

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH       = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
FRAME_PATH      = os.path.join(BASE, "data", "processed", "frame_adoption.parquet")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
TOPIC_PATH      = os.path.join(BASE, "data", "processed", "topic_assignments.parquet")
SHAPE_PATH      = os.path.join(BASE, "data", "processed", "shape_clusters.parquet")
OPEN_PATH       = os.path.join(BASE, "data", "processed", "opening_classifications.parquet")
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")

OUT_REPORT  = os.path.join(BASE, "outputs", "reports", "influence_prediction_report.json")
FIG_DIR     = os.path.join(BASE, "outputs", "figures", "influence_prediction")
MODEL_DIR   = os.path.join(BASE, "outputs", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "influence_predictor.joblib")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -- Style -------------------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
DPI = 150

FRAME_COLORS = {
    "adopt":    "#59A14F",
    "extend":   "#8CD17D",
    "redirect": "#F28E2B",
    "reject":   "#E15759",
    "ignore":   "#BAB0AC",
    "steer":    "#4E79A7",
}

VALID_FRAMES = ["adopt", "extend", "redirect", "reject", "ignore", "steer"]

# -- Helpers -----------------------------------------------------------------
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
    return clean(v=d)


def figpath(name):
    return os.path.join(FIG_DIR, name)


# ── Step 0: Load Data ───────────────────────────────────────────────────────
def load_data():
    print("\n== Step 0: Load data ===================================================")

    for path, label in [
        (CONV_PATH, "conversations_clean.parquet"),
        (MSGS_PATH, "messages_clean.parquet"),
        (FRAME_PATH, "frame_adoption.parquet"),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found at {path}")
            sys.exit(1)

    conv   = pd.read_parquet(CONV_PATH)
    msgs   = pd.read_parquet(MSGS_PATH)
    frames = pd.read_parquet(FRAME_PATH)

    print(f"  Conversations : {len(conv):,}")
    print(f"  Messages      : {len(msgs):,}")
    print(f"  Frame labels  : {len(frames):,}")

    # Optional upstream data
    func_class = _load_optional(FUNC_CLASS_PATH, "Functional classifications")
    emot       = _load_optional(EMOT_PATH, "Emotional states")
    topics     = _load_optional(TOPIC_PATH, "Topic assignments")
    shapes     = _load_optional(SHAPE_PATH, "Shape clusters")
    openings   = _load_optional(OPEN_PATH, "Opening classifications")

    return conv, msgs, frames, func_class, emot, topics, shapes, openings


def _load_optional(path, label):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  {label}: {len(df):,}")
        return df
    print(f"  {label}: not found (skipping)")
    return None


# ── Step 1: Feature Engineering ─────────────────────────────────────────────
def build_features(frames, conv, msgs, func_class, emot, topics, shapes, openings):
    """
    Build feature matrix from frame adoption labels + upstream data.
    Each row = one user message with a frame adoption label.
    """
    print("\n== Step 1: Feature engineering =========================================")

    df = frames.copy()

    # Filter to valid labels only (exclude 'error')
    df = df[df["frame_adoption"].isin(VALID_FRAMES)].copy()
    print(f"  Valid labeled messages: {len(df):,}")

    # --- Message-level features ---
    df["user_tokens"] = df["user_tokens"].astype(float)
    df["assistant_tokens_preceding"] = df["assistant_tokens_preceding"].astype(float)
    df["token_ratio"] = (
        df["user_tokens"] / df["assistant_tokens_preceding"].clip(lower=1)
    )
    df["log_user_tokens"] = np.log1p(df["user_tokens"])
    df["log_asst_tokens"] = np.log1p(df["assistant_tokens_preceding"])

    # --- Conversation-level features (merge from conv) ---
    conv_features = conv[["conversation_id"]].copy()
    for col in ["msg_count", "user_msg_count", "assistant_msg_count",
                 "user_token_total", "assistant_token_total",
                 "duration_minutes", "hour_of_day", "day_of_week",
                 "has_code", "is_branched"]:
        if col in conv.columns:
            conv_features[col] = conv[col].values

    df = df.merge(conv_features, on="conversation_id", how="left")

    # Turn position within conversation
    if "msg_count" in df.columns:
        df["turn_position"] = df["message_index"] / df["msg_count"].clip(lower=1)
    else:
        df["turn_position"] = 0.0

    # Log conversation length
    if "msg_count" in df.columns:
        df["log_msg_count"] = np.log1p(df["msg_count"])

    # Conversation token ratio (overall)
    if "user_token_total" in df.columns and "assistant_token_total" in df.columns:
        df["conv_token_ratio"] = (
            df["user_token_total"] / df["assistant_token_total"].clip(lower=1)
        )

    # --- Functional category (encoded) ---
    if func_class is not None and "function" in func_class.columns:
        fmap = func_class.set_index("conversation_id")["function"]
        df["function"] = df["conversation_id"].map(fmap).fillna("unknown")
    else:
        df["function"] = "unknown"
    df["function_encoded"] = LabelEncoder().fit_transform(df["function"].astype(str))

    # --- Emotional state (encoded) ---
    if emot is not None and "emotion" in emot.columns:
        emap = emot.set_index("conversation_id")["emotion"]
        df["emotion"] = df["conversation_id"].map(emap).fillna("unknown")
    else:
        df["emotion"] = "unknown"
    df["emotion_encoded"] = LabelEncoder().fit_transform(df["emotion"].astype(str))

    # --- Topic cluster ---
    if topics is not None and "topic" in topics.columns:
        tmap = topics.set_index("conversation_id")["topic"]
        df["topic"] = df["conversation_id"].map(tmap).fillna(-1).astype(int)
    else:
        df["topic"] = -1

    # --- Shape cluster ---
    if shapes is not None:
        shape_col = "cluster_label" if "cluster_label" in shapes.columns else "shape_archetype"
        if shape_col in shapes.columns:
            smap = shapes.set_index("conversation_id")[shape_col]
            df["shape"] = df["conversation_id"].map(smap).fillna("unknown")
        else:
            df["shape"] = "unknown"
    else:
        df["shape"] = "unknown"
    df["shape_encoded"] = LabelEncoder().fit_transform(df["shape"].astype(str))

    # --- Opening taxonomy ---
    if openings is not None and "opening_l1" in openings.columns:
        omap = openings.set_index("conversation_id")["opening_l1"]
        df["opening_type"] = df["conversation_id"].map(omap).fillna("unknown")
    else:
        df["opening_type"] = "unknown"
    df["opening_encoded"] = LabelEncoder().fit_transform(df["opening_type"].astype(str))

    # --- Sequence features (within-conversation frame history) ---
    print("  Building sequence features...")
    df = df.sort_values(["conversation_id", "message_index"]).reset_index(drop=True)
    df = _add_sequence_features(df)

    # --- Boolean features as int ---
    for col in ["has_code", "is_branched", "is_conversation_opener"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Weekend indicator
    if "day_of_week" in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Late night indicator
    if "hour_of_day" in df.columns:
        df["is_late_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 4)).astype(int)

    print(f"  Final feature matrix: {df.shape}")
    return df


def _add_sequence_features(df):
    """Add within-conversation frame history features."""
    # For each message, compute features based on prior frames in same conversation
    adopt_running = []
    extend_running = []
    redirect_running = []
    streak_len = []
    prev_frame_encoded = []
    n_prior_messages = []

    frame_to_int = {f: i for i, f in enumerate(VALID_FRAMES)}

    for _, group in df.groupby("conversation_id", sort=False):
        frames_so_far = []
        for idx in group.index:
            current_frame = df.loc[idx, "frame_adoption"]
            n_prior = len(frames_so_far)
            n_prior_messages.append(n_prior)

            if n_prior == 0:
                adopt_running.append(0.0)
                extend_running.append(0.0)
                redirect_running.append(0.0)
                streak_len.append(0)
                prev_frame_encoded.append(-1)
            else:
                adopt_running.append(sum(1 for f in frames_so_far if f == "adopt") / n_prior)
                extend_running.append(sum(1 for f in frames_so_far if f == "extend") / n_prior)
                redirect_running.append(sum(1 for f in frames_so_far if f == "redirect") / n_prior)
                prev_frame_encoded.append(frame_to_int.get(frames_so_far[-1], -1))

                # Streak: how many consecutive same frames before this one
                s = 0
                if len(frames_so_far) >= 2:
                    last = frames_so_far[-1]
                    for f in reversed(frames_so_far[:-1]):
                        if f == last:
                            s += 1
                        else:
                            break
                streak_len.append(s)

            frames_so_far.append(current_frame)

    df["adopt_rate_prior"] = adopt_running
    df["extend_rate_prior"] = extend_running
    df["redirect_rate_prior"] = redirect_running
    df["prev_frame_encoded"] = prev_frame_encoded
    df["frame_streak_len"] = streak_len
    df["n_prior_user_msgs"] = n_prior_messages

    return df


# ── Step 2: Define Feature Columns ──────────────────────────────────────────
def get_feature_cols(df):
    """Return list of feature column names that exist in df."""
    candidates = [
        # Message-level
        "user_tokens", "assistant_tokens_preceding", "token_ratio",
        "log_user_tokens", "log_asst_tokens",
        "message_index", "turn_position", "is_conversation_opener",
        # Conversation-level
        "msg_count", "log_msg_count", "user_msg_count", "assistant_msg_count",
        "user_token_total", "assistant_token_total", "conv_token_ratio",
        "duration_minutes", "hour_of_day", "day_of_week",
        "has_code", "is_branched", "is_weekend", "is_late_night",
        # Categorical (encoded)
        "function_encoded", "emotion_encoded", "topic",
        "shape_encoded", "opening_encoded",
        # Sequence
        "adopt_rate_prior", "extend_rate_prior", "redirect_rate_prior",
        "prev_frame_encoded", "frame_streak_len", "n_prior_user_msgs",
    ]
    return [c for c in candidates if c in df.columns]


# ── Step 3: Train & Evaluate ────────────────────────────────────────────────
def train_and_evaluate(df, feature_cols, n_folds=5, binary_mode=False):
    """Stratified K-fold cross-validation with LightGBM."""
    print(f"\n== Step 2: Train & evaluate ({'binary' if binary_mode else '6-class'}, "
          f"{n_folds}-fold CV) ===")

    target_col = "frame_adoption"

    if binary_mode:
        # Binary: influenced (adopt+extend) vs independent (redirect+reject+ignore+steer)
        df["target"] = (df[target_col].isin(["adopt", "extend"])).astype(int)
        class_names = ["independent", "influenced"]
    else:
        le = LabelEncoder()
        le.fit(VALID_FRAMES)
        df["target"] = le.transform(df[target_col])
        class_names = list(le.classes_)

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values
    n_classes = len(class_names)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Classes: {class_names}")
    print(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # LightGBM parameters
    params = {
        "objective": "binary" if binary_mode else "multiclass",
        "metric": "binary_logloss" if binary_mode else "multi_logloss",
        "num_class": n_classes if not binary_mode else 1,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if not binary_mode:
        del params["num_class"]  # LightGBM infers from data

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Storage for CV results
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    feature_importances = np.zeros(len(feature_cols))
    best_model = None
    best_score = -1

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")

        fold_metrics.append({"fold": fold_i, "accuracy": acc, "f1_macro": f1_macro})
        print(f"  Fold {fold_i}: accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        feature_importances += model.feature_importances_

        if f1_macro > best_score:
            best_score = f1_macro
            best_model = model

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    feature_importances /= n_folds

    # Baseline: stratified dummy
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X, y)
    dummy_pred = dummy.predict(X)
    dummy_acc = accuracy_score(y, dummy_pred)
    dummy_f1 = f1_score(y, dummy_pred, average="macro")

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    overall_loss = log_loss(all_y_true, all_y_proba)

    print(f"\n  === Overall CV Results ===")
    print(f"  Accuracy:      {overall_acc:.4f}  (baseline: {dummy_acc:.4f})")
    print(f"  F1 Macro:      {overall_f1:.4f}  (baseline: {dummy_f1:.4f})")
    print(f"  Log Loss:      {overall_loss:.4f}")
    print(f"  Lift over baseline: {overall_f1 / max(dummy_f1, 0.001):.2f}x")

    # Classification report
    cls_report = classification_report(
        all_y_true, all_y_pred,
        target_names=class_names,
        output_dict=True,
    )
    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    # Feature importance ranking
    fi_sorted = sorted(
        zip(feature_cols, feature_importances),
        key=lambda x: x[1], reverse=True,
    )
    print("  Top 15 features:")
    for fname, fimp in fi_sorted[:15]:
        print(f"    {fname:35s} {fimp:.1f}")

    results = {
        "binary_mode": binary_mode,
        "n_folds": n_folds,
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "class_names": class_names,
        "overall": {
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "log_loss": overall_loss,
        },
        "baseline": {
            "accuracy": dummy_acc,
            "f1_macro": dummy_f1,
        },
        "lift": overall_f1 / max(dummy_f1, 0.001),
        "fold_metrics": fold_metrics,
        "classification_report": cls_report,
        "feature_importance": fi_sorted,
    }

    return results, best_model, all_y_true, all_y_pred, all_y_proba, class_names


# ── Step 4: Figures ─────────────────────────────────────────────────────────
def make_figures(results, y_true, y_pred, y_proba, class_names, feature_cols):
    print("\n== Step 3: Generate figures =============================================")

    _fig_confusion_matrix(y_true, y_pred, class_names)
    _fig_feature_importance(results["feature_importance"])
    _fig_roc_curves(y_true, y_proba, class_names)
    _fig_calibration(y_true, y_proba, class_names)


def _fig_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Frame Adoption Prediction", fontsize=14, fontweight="bold")

    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(class_names, fontsize=9)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Counts")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Normalized
    im2 = ax2.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Normalized (row)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    path = figpath("confusion_matrix.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def _fig_feature_importance(fi_sorted):
    top_n = min(25, len(fi_sorted))
    names = [f[0] for f in fi_sorted[:top_n]][::-1]
    values = [f[1] for f in fi_sorted[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = [COLOR_PRIMARY if "prior" in n or "streak" in n or "prev_frame" in n
              else COLOR_ACCENT if "token" in n
              else COLOR_SECONDARY for n in names]
    ax.barh(range(len(names)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance (LightGBM split gain)")
    ax.set_title("Top Feature Importances — Influence Prediction", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_PRIMARY, label="Sequence features"),
        Patch(facecolor=COLOR_ACCENT, label="Token features"),
        Patch(facecolor=COLOR_SECONDARY, label="Other features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = figpath("feature_importance.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def _fig_roc_curves(y_true, y_proba, class_names):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(class_names):
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        auc = roc_auc_score(y_binary, y_proba[:, i])
        color = FRAME_COLORS.get(cls, COLOR_SECONDARY)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest per Frame Category", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = figpath("roc_curves.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def _fig_calibration(y_true, y_proba, class_names):
    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Probability Calibration — Per Class", fontsize=13, fontweight="bold")
    axes = np.atleast_2d(axes)

    for i, cls in enumerate(class_names):
        ax = axes[i // n_cols, i % n_cols]
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() < 10:
            ax.text(0.5, 0.5, "Too few\nsamples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(cls)
            continue
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, i], n_bins=10)
        color = FRAME_COLORS.get(cls, COLOR_SECONDARY)
        ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=cls)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(cls, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for i in range(n_classes, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    path = figpath("calibration.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ── Step 5: SHAP ────────────────────────────────────────────────────────────
def compute_shap(model, X, feature_cols, class_names):
    """Compute SHAP values if shap is available."""
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis")
        return None

    print("\n== Step 4: SHAP analysis ===============================================")
    # Sample for speed
    n_sample = min(2000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_cols,
        class_names=class_names, show=False, max_display=20,
    )
    path = figpath("shap_summary.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    return shap_values


# ── Step 6: Save Model & Report ─────────────────────────────────────────────
def save_model(model, feature_cols, class_names):
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "class_names": class_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n  Model saved: {MODEL_PATH}")


def generate_report(results):
    report = {
        "module": "19_influence_prediction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "ML model predicting user frame adoption response to AI output, "
            "trained on features from upstream pipeline modules"
        ),
        **results,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(clean_dict(report), f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report saved: {OUT_REPORT}")
    return report


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train influence prediction model")
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--binary", action="store_true",
                        help="Binary mode: influenced vs independent")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("MODULE 19: INFLUENCE PREDICTION VIA MACHINE LEARNING")
    print("=" * 70)

    conv, msgs, frames, func_class, emot, topics, shapes, openings = load_data()

    df = build_features(frames, conv, msgs, func_class, emot, topics, shapes, openings)
    feature_cols = get_feature_cols(df)

    results, best_model, y_true, y_pred, y_proba, class_names = train_and_evaluate(
        df, feature_cols, n_folds=args.folds, binary_mode=args.binary,
    )

    make_figures(results, y_true, y_pred, y_proba, class_names, feature_cols)

    if not args.skip_shap:
        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        compute_shap(best_model, X, feature_cols, class_names)

    save_model(best_model, feature_cols, class_names)
    report = generate_report(results)

    # Print highlights
    print("\n" + "=" * 70)
    print("HIGHLIGHTS")
    print("=" * 70)
    r = results["overall"]
    b = results["baseline"]
    print(f"  Accuracy:  {r['accuracy']:.4f}  (baseline: {b['accuracy']:.4f})")
    print(f"  F1 Macro:  {r['f1_macro']:.4f}  (baseline: {b['f1_macro']:.4f})")
    print(f"  Lift:      {results['lift']:.2f}x over stratified random")
    print(f"\n  Top 5 predictive features:")
    for fname, fimp in results["feature_importance"][:5]:
        print(f"    {fname}")

    print("\nDone!")


if __name__ == "__main__":
    main()
