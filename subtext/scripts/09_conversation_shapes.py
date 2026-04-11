"""
Module 3.1: Structural Analysis - Conversation Shapes
Reads clean Parquet files, clusters conversation shape vectors, produces 12 figures + JSON report.
"""

import os
import sys
import json
import warnings
import traceback
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, entropy as scipy_entropy
from sklearn.metrics import silhouette_score as sk_silhouette

warnings.filterwarnings("ignore")

# --- tslearn import (DTW) ---------------------------------------------------
DTW_AVAILABLE = False
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.clustering import silhouette_score as ts_silhouette
    from tslearn.utils import to_time_series_dataset
    DTW_AVAILABLE = True
    print("tslearn loaded - DTW clustering enabled")
except ImportError:
    from sklearn.cluster import KMeans
    print("tslearn not found - falling back to Euclidean k-means")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Paths ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_PATH    = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
MSGS_PATH    = os.path.join(BASE, "data", "processed", "messages_clean.parquet")
CLUSTER_PATH = os.path.join(BASE, "data", "processed", "shape_clusters.parquet")
FIGS_DIR     = os.path.join(BASE, "outputs", "figures", "conversation_shapes")
REPORT_PATH  = os.path.join(BASE, "outputs", "reports", "conversation_shapes_report.json")
CONFIG_PATH  = os.path.join(BASE, "config", "quality_config.json")

os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# --- Style constants --------------------------------------------------------
COLOR_PRIMARY   = "#2E75B6"
COLOR_SECONDARY = "#666666"
COLOR_ACCENT    = "#C55A11"
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE     = (14, 6)
FIGSIZE_TALL     = (10, 8)
FIGSIZE_GALLERY  = (16, 10)
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE  = 10
DPI = 150

# --- Helpers ----------------------------------------------------------------
def clean(v):
    if isinstance(v, (float, np.floating)):
        val = float(v)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def clean_dict(d):
    if isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict(v) for v in d]
    return clean(d)

def figpath(name):
    return os.path.join(FIGS_DIR, name)

def save_fig(fig, fname):
    path = figpath(fname)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures_generated.append(f"outputs/figures/conversation_shapes/{fname}")
    print(f"  Saved: {fname}")

report = {
    "module": "conversation_shapes",
    "module_version": "1.0",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "warnings": [],
}
figures_generated = []

# load era boundaries
try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    era_boundaries = config.get("model_era_boundaries", {})
except Exception:
    era_boundaries = {}

# ===========================================================================
# STEP 0 - Load, filter, set minimum length
# ===========================================================================
print("\n-- Step 0: Load & filter -----------------------------------------------")

conversations = pd.read_parquet(CONV_PATH)
messages      = pd.read_parquet(MSGS_PATH)

conv = conversations[conversations["is_analysable"]].copy()
msgs = messages[messages["conversation_id"].isin(conv["conversation_id"])].copy()

MIN_MESSAGES = 6
shape_conv = conv[conv["msg_count"] >= MIN_MESSAGES].copy()
shape_msgs = msgs[msgs["conversation_id"].isin(shape_conv["conversation_id"])].copy()

print(f"  Analysable conversations       : {len(conv):,}")
print(f"  Shape-eligible (>={MIN_MESSAGES} messages): {len(shape_conv):,}")
print(f"  Excluded (too short)           : {len(conv) - len(shape_conv):,}")
print(f"  Messages in scope              : {len(shape_msgs):,}")

report["input_data"] = {
    "analysable_conversations": len(conv),
    "shape_eligible": len(shape_conv),
    "excluded_too_short": len(conv) - len(shape_conv),
    "min_messages_threshold": MIN_MESSAGES,
}

# ===========================================================================
# STEP 1 - Shape vector construction
# ===========================================================================
print("\n-- Step 1: Shape vector construction ----------------------------------")

N_BINS = 10

def build_shape_vector(conv_msgs, n_bins=N_BINS):
    conv_msgs = conv_msgs.sort_values("msg_index").copy()
    bins = np.linspace(0, 1, n_bins + 1)
    conv_msgs["shape_bin"] = pd.cut(
        conv_msgs["position_in_conversation"],
        bins=bins, labels=range(n_bins), include_lowest=True,
    )
    bin_means = conv_msgs.groupby("shape_bin", observed=True)["token_count"].mean()
    vector = bin_means.reindex(range(n_bins)).interpolate(method="linear").ffill().bfill()
    return vector.values

def build_role_shape_vector(conv_msgs, role, n_bins=N_BINS):
    role_msgs = conv_msgs[conv_msgs["role"] == role].copy()
    if len(role_msgs) < 3:
        return None
    role_msgs = role_msgs.sort_values("msg_index").copy()
    role_msgs["role_position"] = np.linspace(0, 1, len(role_msgs))
    bins = np.linspace(0, 1, n_bins + 1)
    role_msgs["shape_bin"] = pd.cut(
        role_msgs["role_position"],
        bins=bins, labels=range(n_bins), include_lowest=True,
    )
    bin_means = role_msgs.groupby("shape_bin", observed=True)["token_count"].mean()
    vector = bin_means.reindex(range(n_bins)).interpolate(method="linear").ffill().bfill()
    return vector.values

def normalize_vector(v):
    v_min, v_max = v.min(), v.max()
    if v_max - v_min < 1e-6:
        return np.zeros_like(v, dtype=float)
    return (v - v_min) / (v_max - v_min)

def extract_shape_features(vec):
    n = len(vec)
    x = np.arange(n)
    slope     = float(np.polyfit(x, vec, 1)[0])
    curvature = float(np.mean(np.diff(vec, n=2))) if n >= 3 else 0.0
    first_diff = np.diff(vec)
    volatility = float(np.std(first_diff)) if len(first_diff) > 0 else 0.0
    peak_position  = float(np.argmax(vec) / (n - 1)) if n > 1 else 0.5
    peak_prominence = float((vec.max() - vec.mean()) / (vec.max() + 1e-6))
    half = n // 2
    if half >= 2:
        fh = vec[:half]; sh_rev = vec[-half:][::-1]
        if np.std(fh) > 1e-6 and np.std(sh_rev) > 1e-6:
            symmetry = float(np.corrcoef(fh, sh_rev)[0, 1])
        else:
            symmetry = 1.0
    else:
        symmetry = 0.0
    flatness = float(1.0 - np.std(vec))
    total = vec.sum()
    front_half_mass = float(vec[:half].sum() / total) if total > 1e-6 else 0.5
    return {
        "slope": slope, "curvature": curvature, "volatility": volatility,
        "peak_position": peak_position, "peak_prominence": peak_prominence,
        "symmetry": symmetry, "flatness": flatness, "front_half_mass": front_half_mass,
    }

FEATURE_COLS = ["slope", "curvature", "volatility", "peak_position",
                "peak_prominence", "symmetry", "flatness", "front_half_mass"]

# Build vectors
shape_vectors = {}
user_vectors  = {}
asst_vectors  = {}

grouped = list(shape_msgs.groupby("conversation_id"))
total = len(grouped)
for i, (cid, grp) in enumerate(grouped):
    if i % 200 == 0:
        print(f"  Building vectors... {i}/{total}", end="\r")
    vec = build_shape_vector(grp)
    if vec is not None and not np.isnan(vec).any():
        shape_vectors[cid] = normalize_vector(vec.astype(float))
    u_vec = build_role_shape_vector(grp, "user")
    if u_vec is not None and not np.isnan(u_vec).any():
        user_vectors[cid] = normalize_vector(u_vec.astype(float))
    a_vec = build_role_shape_vector(grp, "assistant")
    if a_vec is not None and not np.isnan(a_vec).any():
        asst_vectors[cid] = normalize_vector(a_vec.astype(float))

conv_ids     = list(shape_vectors.keys())
shape_matrix = np.array([shape_vectors[cid] for cid in conv_ids])

print(f"\n  Shape vectors built: {len(conv_ids):,}")
print(f"  Matrix shape       : {shape_matrix.shape}")
print(f"  User vectors       : {len(user_vectors):,}")
print(f"  Asst vectors       : {len(asst_vectors):,}")

# Shape features
features_list = []
for cid in conv_ids:
    feats = extract_shape_features(shape_vectors[cid])
    feats["conversation_id"] = cid
    features_list.append(feats)
shape_features = pd.DataFrame(features_list).set_index("conversation_id")

print("  Step 1 complete.")

# ===========================================================================
# STEP 2 - Clustering
# ===========================================================================
print("\n-- Step 2: Clustering --------------------------------------------------")

def name_cluster(centroid_vector):
    feats = extract_shape_features(centroid_vector)
    if feats["flatness"] > 0.7:
        return "plateau"
    if feats["volatility"] > 0.2:
        return "oscillating"
    if feats["peak_prominence"] > 0.5 and 0.3 <= feats["peak_position"] <= 0.7:
        return "spike"
    if feats["peak_position"] < 0.3 and feats["peak_prominence"] > 0.3:
        return "front_loaded"
    if feats["peak_position"] > 0.7 and feats["peak_prominence"] > 0.3:
        return "back_loaded"
    if feats["slope"] > 0.03 and feats["volatility"] < 0.15:
        return "deepening"
    if feats["slope"] < -0.03 and feats["volatility"] < 0.15:
        return "tapering"
    if feats["front_half_mass"] < 0.4 and feats["symmetry"] > 0.5 and feats["curvature"] > 0.01:
        return "u_shaped"
    if (0.35 < feats["front_half_mass"] < 0.65 and feats["curvature"] < -0.01
            and 0.3 < feats["peak_position"] < 0.7):
        return "arch"
    return "irregular"

clustering_method = "unknown"
silhouette_scores_tested = {}
best_k = 5
cluster_labels = None
cluster_centers = None

try:
    if DTW_AVAILABLE and len(conv_ids) <= 3000:
        print("  Using DTW k-means (tslearn)...")
        clustering_method = "dtw_kmeans"
        ts_data = to_time_series_dataset(shape_matrix)

        k_range = [4, 5, 6, 7]
        dtw_results = {}
        for k in k_range:
            print(f"  Fitting k={k}...", end=" ", flush=True)
            model = TimeSeriesKMeans(
                n_clusters=k, metric="dtw", max_iter=30,
                random_state=42, n_jobs=1, verbose=0,
            )
            labels = model.fit_predict(ts_data)
            score  = ts_silhouette(ts_data, labels, metric="dtw")
            silhouette_scores_tested[str(k)] = clean(score)
            dtw_results[k] = {"model": model, "labels": labels, "silhouette": float(score)}
            print(f"silhouette={score:.3f}")

        best_k = max(dtw_results, key=lambda k: dtw_results[k]["silhouette"])
        best_model    = dtw_results[best_k]["model"]
        cluster_labels  = dtw_results[best_k]["labels"]
        cluster_centers = best_model.cluster_centers_.squeeze()  # (k, 10)
        best_silhouette = dtw_results[best_k]["silhouette"]

    else:
        print("  Using Euclidean k-means (sklearn)...")
        clustering_method = "euclidean_kmeans"
        from sklearn.cluster import KMeans

        k_range = [4, 5, 6, 7, 8]
        euc_results = {}
        for k in k_range:
            print(f"  Fitting k={k}...", end=" ", flush=True)
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(shape_matrix)
            score  = sk_silhouette(shape_matrix, labels,
                                   sample_size=min(5000, len(shape_matrix)))
            silhouette_scores_tested[str(k)] = clean(score)
            euc_results[k] = {"model": km, "labels": labels, "silhouette": float(score)}
            print(f"silhouette={score:.3f}")

        best_k = max(euc_results, key=lambda k: euc_results[k]["silhouette"])
        cluster_labels  = euc_results[best_k]["labels"]
        cluster_centers = euc_results[best_k]["model"].cluster_centers_
        best_silhouette = euc_results[best_k]["silhouette"]

    print(f"\n  Best k={best_k}, silhouette={best_silhouette:.3f}")

    # Name clusters; deduplicate
    raw_names = [name_cluster(cluster_centers[i]) for i in range(best_k)]
    name_count = {}
    cluster_names = []
    suffixes = {
        "deepening":   ["_gradual", "_steep"],
        "tapering":    ["_gradual", "_steep"],
        "plateau":     ["_a", "_b"],
        "oscillating": ["_a", "_b"],
        "irregular":   ["_a", "_b", "_c", "_d"],
        "front_loaded":["_a", "_b"],
        "back_loaded": ["_a", "_b"],
        "spike":       ["_a", "_b"],
        "u_shaped":    ["_a", "_b"],
        "arch":        ["_a", "_b"],
    }
    for i, raw in enumerate(raw_names):
        count = name_count.get(raw, 0)
        if count == 0:
            cluster_names.append(raw)
        else:
            sfx_list = suffixes.get(raw, [f"_{j}" for j in range(10)])
            # try to use slope to differentiate deepening/tapering
            if raw in ("deepening", "tapering"):
                feats = extract_shape_features(cluster_centers[i])
                qualifier = "_steep" if abs(feats["slope"]) > 0.06 else "_gradual"
                cluster_names.append(raw + qualifier)
            else:
                sfx = sfx_list[count - 1] if count - 1 < len(sfx_list) else f"_{count}"
                cluster_names.append(raw + sfx)
        name_count[raw] = name_count.get(raw, 0) + 1

    # Ensure unique names (if still duplicates after disambiguation)
    seen = {}
    final_names = []
    for nm in cluster_names:
        if nm in seen:
            seen[nm] += 1
            final_names.append(f"{nm}_{seen[nm]}")
        else:
            seen[nm] = 0
            final_names.append(nm)
    cluster_names = final_names

    print(f"  Cluster names: {cluster_names}")

    # 2e. Save cluster assignments
    cluster_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "shape_cluster_id": cluster_labels.astype(int),
        "shape_cluster_name": [cluster_names[l] for l in cluster_labels],
    }).set_index("conversation_id")
    cluster_df = cluster_df.join(shape_features)
    cluster_df.to_parquet(CLUSTER_PATH)
    print(f"  shape_clusters.parquet saved ({len(cluster_df):,} rows)")
    print("  Step 2 complete.")

except Exception as e:
    report["warnings"].append(f"Step 2 error: {traceback.format_exc()}")
    print(f"  Step 2 ERROR: {e}")
    # fallback
    best_k = 1
    cluster_labels = np.zeros(len(conv_ids), dtype=int)
    cluster_centers = shape_matrix.mean(axis=0, keepdims=True)
    cluster_names = ["irregular"]
    best_silhouette = 0.0
    cluster_df = pd.DataFrame({
        "conversation_id": conv_ids,
        "shape_cluster_id": 0,
        "shape_cluster_name": "irregular",
    }).set_index("conversation_id").join(shape_features)

# Cluster color palette
CLUSTER_COLORS = {
    name: matplotlib.cm.tab10(i / max(best_k, 1))
    for i, name in enumerate(cluster_names)
}
CLUSTER_COLORS_HEX = {
    name: "#{:02x}{:02x}{:02x}".format(
        int(matplotlib.cm.tab10(i / max(best_k, 1))[0] * 255),
        int(matplotlib.cm.tab10(i / max(best_k, 1))[1] * 255),
        int(matplotlib.cm.tab10(i / max(best_k, 1))[2] * 255),
    )
    for i, name in enumerate(cluster_names)
}

# ===========================================================================
# STEP 3 - Cluster analysis
# ===========================================================================
print("\n-- Step 3: Cluster analysis --------------------------------------------")

cluster_analysis_report = {}
try:
    # Join cluster df with conversation metadata
    meta_cols = ["conversation_type", "model_era", "year_month", "time_of_day",
                 "is_weekend", "turns", "user_token_ratio", "duration_minutes",
                 "msg_count", "first_user_message_type", "has_code", "quality_score"]
    analysis = cluster_df.join(
        shape_conv.set_index("conversation_id")[meta_cols]
    )

    # 3a. Distribution
    cluster_dist = cluster_df["shape_cluster_name"].value_counts().to_dict()
    cluster_pcts = cluster_df["shape_cluster_name"].value_counts(normalize=True).round(3).to_dict()

    # 3b. Cross-tabulations
    def safe_crosstab(index_series, columns_series, **kwargs):
        try:
            return pd.crosstab(index_series, columns_series, **kwargs)
        except Exception:
            return pd.DataFrame()

    shape_by_type = safe_crosstab(
        analysis["shape_cluster_name"], analysis["conversation_type"], normalize="columns"
    ).round(3)
    shape_by_era = safe_crosstab(
        analysis["shape_cluster_name"], analysis["model_era"], normalize="columns"
    ).round(3)
    shape_by_tod = safe_crosstab(
        analysis["shape_cluster_name"], analysis["time_of_day"], normalize="columns"
    ).round(3)
    shape_by_opening = safe_crosstab(
        analysis["shape_cluster_name"], analysis["first_user_message_type"], normalize="columns"
    ).round(3)

    # 3c. Cluster profiles
    cluster_profiles = {}
    for i, name in enumerate(cluster_names):
        cluster_data = analysis[analysis["shape_cluster_name"] == name]
        centroid = cluster_centers[i]
        centroid_feats = extract_shape_features(centroid)
        cluster_profiles[name] = {
            "count": int(len(cluster_data)),
            "pct_of_total": clean(len(cluster_data) / len(analysis) * 100),
            "centroid": [clean(x) for x in centroid.tolist()],
            "centroid_features": clean_dict(centroid_feats),
            "median_turns": clean(cluster_data["turns"].median()),
            "median_msg_count": clean(cluster_data["msg_count"].median()),
            "median_user_token_ratio": clean(cluster_data["user_token_ratio"].median()),
            "median_duration_minutes": clean(cluster_data["duration_minutes"].median()),
            "top_conversation_type": (
                str(cluster_data["conversation_type"].mode().iloc[0])
                if len(cluster_data) > 0 else None
            ),
            "top_opening_type": (
                str(cluster_data["first_user_message_type"].mode().iloc[0])
                if len(cluster_data) > 0 else None
            ),
            "pct_code": clean(
                cluster_data["has_code"].sum() / len(cluster_data) * 100
                if len(cluster_data) > 0 else 0
            ),
            "shape_features_mean": {
                k: clean(cluster_data[k].mean())
                for k in FEATURE_COLS if k in cluster_data.columns
            },
        }

    # 3d. Longitudinal shape distribution
    monthly_shapes = safe_crosstab(
        analysis["year_month"], analysis["shape_cluster_name"], normalize="index"
    ).round(3)

    # 3e. Statistical tests
    stat_tests = {}
    try:
        contingency_type = pd.crosstab(
            analysis["shape_cluster_name"], analysis["conversation_type"]
        )
        if contingency_type.shape[1] >= 2:
            chi2_type, p_type, dof_type, _ = chi2_contingency(contingency_type)
            def cramers_v(ct):
                chi2 = chi2_contingency(ct)[0]
                n    = ct.values.sum()
                min_dim = min(ct.shape) - 1
                return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0
            v_type = cramers_v(contingency_type)
            stat_tests["shape_vs_conversation_type"] = {
                "chi2": clean(chi2_type), "p_value": clean(p_type),
                "cramers_v": clean(v_type),
                "interpretation": (
                    "Significant association" if p_type < 0.05 else "No significant association"
                ) + f" (V={v_type:.3f})"
            }
        else:
            stat_tests["shape_vs_conversation_type"] = {
                "chi2": None, "p_value": None, "cramers_v": None,
                "interpretation": "Only one conversation type - test not applicable"
            }
    except Exception as ex:
        stat_tests["shape_vs_conversation_type"] = {
            "error": str(ex), "chi2": None, "p_value": None, "cramers_v": None
        }

    try:
        contingency_era = pd.crosstab(
            analysis["shape_cluster_name"], analysis["model_era"]
        )
        if contingency_era.shape[1] >= 2:
            chi2_era, p_era, dof_era, _ = chi2_contingency(contingency_era)
            v_era = cramers_v(contingency_era)
            stat_tests["shape_vs_model_era"] = {
                "chi2": clean(chi2_era), "p_value": clean(p_era),
                "cramers_v": clean(v_era),
                "interpretation": (
                    "Significant association" if p_era < 0.05 else "No significant association"
                ) + f" (V={v_era:.3f})"
            }
        else:
            stat_tests["shape_vs_model_era"] = {
                "chi2": None, "p_value": None, "cramers_v": None,
                "interpretation": "Only one model era in dataset - chi2 test not applicable"
            }
            report["warnings"].append("shape_vs_model_era: only one era in data; chi2 skipped.")
    except Exception as ex:
        stat_tests["shape_vs_model_era"] = {"error": str(ex), "chi2": None, "p_value": None}

    cluster_analysis_report = {
        "distribution": {k: int(v) for k, v in cluster_dist.items()},
        "distribution_pct": {k: clean(v * 100) for k, v in cluster_pcts.items()},
        "profiles": cluster_profiles,
        "cross_tabulations": {
            "shape_by_conversation_type": clean_dict(shape_by_type.to_dict()),
            "shape_by_model_era": clean_dict(shape_by_era.to_dict()),
            "shape_by_time_of_day": clean_dict(shape_by_tod.to_dict()),
            "shape_by_opening_type": clean_dict(shape_by_opening.to_dict()),
        },
        "statistical_tests": stat_tests,
        "monthly_shapes": clean_dict(monthly_shapes.to_dict(orient="index")),
    }

    dominant_shape = max(cluster_dist, key=cluster_dist.get)
    print(f"  Dominant shape: {dominant_shape} ({cluster_dist[dominant_shape]} / {len(cluster_df)})")
    print("  Step 3 complete.")

except Exception as e:
    report["warnings"].append(f"Step 3 error: {traceback.format_exc()}")
    print(f"  Step 3 ERROR: {e}")
    analysis = cluster_df.copy()
    stat_tests = {}
    monthly_shapes = pd.DataFrame()
    dominant_shape = cluster_names[0] if cluster_names else "unknown"
    cluster_dist   = {}
    cluster_pcts   = {}
    cluster_profiles = {}

# ===========================================================================
# STEP 4 - User vs. assistant shape comparison
# ===========================================================================
print("\n-- Step 4: User vs. assistant shapes ----------------------------------")

uva_report = {}
try:
    def cosine_similarity(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0

    agreement_scores = {
        cid: cosine_similarity(user_vectors[cid], asst_vectors[cid])
        for cid in conv_ids
        if cid in user_vectors and cid in asst_vectors
    }

    # Divergence classification
    divergence = {}
    for cid, score in agreement_scores.items():
        u_feats = extract_shape_features(user_vectors[cid])
        a_feats = extract_shape_features(asst_vectors[cid])
        if score > 0.8:
            pattern = "synchronized"
        elif score < -0.3:
            pattern = "inverse"
        elif u_feats["peak_position"] < a_feats["peak_position"] - 0.2:
            pattern = "user_leads"
        elif a_feats["peak_position"] < u_feats["peak_position"] - 0.2:
            pattern = "assistant_leads"
        else:
            pattern = "independent"
        divergence[cid] = pattern

    div_counts = pd.Series(divergence).value_counts()
    total_div  = len(divergence)

    # Average shapes
    all_u_vecs = np.array([user_vectors[cid] for cid in user_vectors])
    all_a_vecs = np.array([asst_vectors[cid] for cid in asst_vectors])
    avg_user_shape = all_u_vecs.mean(axis=0)
    avg_asst_shape = all_a_vecs.mean(axis=0)
    std_user_shape = all_u_vecs.std(axis=0)
    std_asst_shape = all_a_vecs.std(axis=0)

    overall_cosine = cosine_similarity(avg_user_shape, avg_asst_shape)

    uva_report = {
        "avg_user_shape": [clean(x) for x in avg_user_shape.tolist()],
        "avg_assistant_shape": [clean(x) for x in avg_asst_shape.tolist()],
        "overall_cosine_similarity": clean(overall_cosine),
        "agreement_distribution": {
            f"{p}_pct": clean(div_counts.get(p, 0) / total_div * 100)
            for p in ["synchronized", "user_leads", "assistant_leads", "inverse", "independent"]
        },
    }
    agg_scores = list(agreement_scores.values())
    print(f"  Pairs analysed: {len(agreement_scores):,}")
    print(f"  Overall cosine: {overall_cosine:.3f}")
    print(f"  Divergence distribution: {div_counts.to_dict()}")
    print("  Step 4 complete.")
except Exception as e:
    report["warnings"].append(f"Step 4 error: {traceback.format_exc()}")
    print(f"  Step 4 ERROR: {e}")
    agg_scores = []
    avg_user_shape = np.zeros(N_BINS)
    avg_asst_shape = np.zeros(N_BINS)
    std_user_shape = np.zeros(N_BINS)
    std_asst_shape = np.zeros(N_BINS)

# ===========================================================================
# STEP 5 - Shape stability
# ===========================================================================
print("\n-- Step 5: Shape stability --------------------------------------------")

stability_report = {}
try:
    shape_counts_norm = cluster_df["shape_cluster_name"].value_counts(normalize=True)
    shape_ent = float(scipy_entropy(shape_counts_norm))
    max_ent   = float(np.log(best_k)) if best_k > 1 else 1.0
    norm_ent  = shape_ent / max_ent if max_ent > 0 else 0.0

    stability_by_type = {}
    for ctype in analysis["conversation_type"].dropna().unique():
        subset = analysis[analysis["conversation_type"] == ctype]
        if len(subset) < 2:
            continue
        tc = subset["shape_cluster_name"].value_counts(normalize=True)
        stability_by_type[str(ctype)] = {
            "dominant_shape": str(tc.index[0]),
            "dominant_pct": clean(tc.iloc[0] * 100),
            "entropy": clean(float(scipy_entropy(tc))),
            "count": int(len(subset)),
        }

    # Shape streaks
    anal_sorted = analysis.dropna(subset=["year_month"]).sort_values("year_month").copy()
    shape_runs = (
        anal_sorted["shape_cluster_name"]
        != anal_sorted["shape_cluster_name"].shift()
    ).cumsum()
    run_lengths = anal_sorted.groupby(shape_runs)["shape_cluster_name"].agg(["first", "count"])
    run_lengths.columns = ["shape", "streak_length"]

    long_streaks = run_lengths[run_lengths["streak_length"] >= 5]
    most_common_streak = (
        long_streaks["shape"].value_counts().index[0]
        if len(long_streaks) > 0 else str(cluster_names[0])
    )

    stability_report = {
        "overall_entropy": clean(shape_ent),
        "normalized_entropy": clean(norm_ent),
        "dominant_shape": str(shape_counts_norm.index[0]),
        "dominant_shape_pct": clean(float(shape_counts_norm.iloc[0]) * 100),
        "stability_by_type": clean_dict(stability_by_type),
        "streak_analysis": {
            "max_streak": int(run_lengths["streak_length"].max()),
            "mean_streak": clean(run_lengths["streak_length"].mean()),
            "median_streak": int(run_lengths["streak_length"].median()),
            "most_common_streak_shape": str(most_common_streak),
        },
    }
    print(f"  Normalized entropy: {norm_ent:.3f}  (0=rigid, 1=uniform)")
    print(f"  Longest streak: {run_lengths['streak_length'].max()}")
    print("  Step 5 complete.")
except Exception as e:
    report["warnings"].append(f"Step 5 error: {traceback.format_exc()}")
    print(f"  Step 5 ERROR: {e}")
    run_lengths = pd.DataFrame(columns=["shape", "streak_length"])
    norm_ent = 0.5

# ===========================================================================
# STEP 6 - Extreme shape identification
# ===========================================================================
print("\n-- Step 6: Extreme shapes ---------------------------------------------")

exemplars = {}
outliers  = {}
try:
    for cluster_idx, cname in enumerate(cluster_names):
        mask = cluster_labels == cluster_idx
        c_ids = [conv_ids[i] for i, m in enumerate(mask) if m]
        c_vecs = shape_matrix[mask]
        if len(c_vecs) == 0:
            exemplars[cname] = {"conversation_ids": [], "distances": []}
            continue
        centroid   = cluster_centers[cluster_idx]
        distances  = np.linalg.norm(c_vecs - centroid, axis=1)
        n_ex = min(3, len(c_ids))
        closest    = np.argsort(distances)[:n_ex]
        exemplars[cname] = {
            "conversation_ids": [c_ids[i] for i in closest],
            "distances": [clean(distances[i]) for i in closest],
        }

    dist_to_center = np.array([
        np.linalg.norm(shape_matrix[i] - cluster_centers[cluster_labels[i]])
        for i in range(len(conv_ids))
    ])
    outlier_idx = np.argsort(dist_to_center)[-10:]
    outliers = {
        "conversation_ids": [conv_ids[i] for i in outlier_idx],
        "distances":        [clean(dist_to_center[i]) for i in outlier_idx],
        "assigned_clusters":[cluster_names[cluster_labels[i]] for i in outlier_idx],
    }
    print(f"  Exemplars computed for {len(exemplars)} clusters")
    print(f"  Top outlier distance: {dist_to_center[outlier_idx[-1]]:.3f}")
    print("  Step 6 complete.")
except Exception as e:
    report["warnings"].append(f"Step 6 error: {traceback.format_exc()}")
    print(f"  Step 6 ERROR: {e}")

# ===========================================================================
# FIGURES
# ===========================================================================
print("\n-- Figures -------------------------------------------------------------")

BIN_LABELS = [f"{i*10}%-{(i+1)*10}%" for i in range(N_BINS)]
X_BINS = np.arange(N_BINS)

# -- Figure 01: Shape vector method illustration ----------------------------
try:
    # Pick an exemplar conversation with >=15 messages
    ex_cid = None
    for cname, ex_data in exemplars.items():
        for cid in ex_data.get("conversation_ids", []):
            cid_msgs = shape_msgs[shape_msgs["conversation_id"] == cid]
            if len(cid_msgs) >= 15:
                ex_cid = cid
                break
        if ex_cid:
            break
    if ex_cid is None:
        cid_lengths = shape_msgs.groupby("conversation_id").size()
        ex_cid = cid_lengths[cid_lengths >= 15].index[0] if (cid_lengths >= 15).any() else conv_ids[0]

    ex_msgs = shape_msgs[shape_msgs["conversation_id"] == ex_cid].sort_values("msg_index")
    ex_vec  = shape_vectors[ex_cid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    colors = [COLOR_PRIMARY if r == "user" else "#A9D18E" for r in ex_msgs["role"]]
    ax1.bar(ex_msgs["msg_index"], ex_msgs["token_count"].fillna(0),
            color=colors, alpha=0.8, width=0.85)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=COLOR_PRIMARY, label="User"),
                         Patch(color="#A9D18E", label="Assistant")],
               fontsize=TICK_SIZE)
    ax1.set_xlabel("Message Index", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Token Count", fontsize=LABEL_SIZE)
    ax1.set_title("Raw Message Lengths\n(one real conversation)", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax1)

    ax2.plot(X_BINS, ex_vec, color=COLOR_ACCENT, linewidth=2.5, marker="o", markersize=6)
    ax2.fill_between(X_BINS, ex_vec, alpha=0.15, color=COLOR_ACCENT)
    ax2.set_xticks(X_BINS)
    ax2.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Normalized Token Density", fontsize=LABEL_SIZE)
    ax2.set_title("10-Bin Normalized Shape Vector\n(min-max normalized)", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax2)

    plt.suptitle("How Conversations Become Shape Vectors",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "01_shape_vector_method.png")
except Exception as e:
    report["warnings"].append(f"Fig 01 error: {e}")
    print(f"  Fig 01 ERROR: {e}")

# -- Figure 02: Cluster gallery ---------------------------------------------
try:
    ncols = min(3, best_k)
    nrows = int(np.ceil(best_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5),
                              constrained_layout=True)
    if best_k == 1:
        axes = np.array([[axes]])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    rng = np.random.default_rng(42)
    for idx, cname in enumerate(cluster_names):
        ax = axes_flat[idx]
        color = CLUSTER_COLORS[cname]
        mask  = cluster_labels == idx
        member_vecs = shape_matrix[mask]
        n_members = len(member_vecs)
        n_show = min(20, n_members)
        sample_idx = rng.choice(n_members, n_show, replace=False) if n_members > n_show else np.arange(n_members)
        for si in sample_idx:
            ax.plot(X_BINS, member_vecs[si], color=color, alpha=0.18, linewidth=0.8)
        ax.plot(X_BINS, cluster_centers[idx], color=color, linewidth=3,
                label="Centroid", zorder=5)
        ax.set_title(f"{cname.replace('_', ' ').title()}\n(n={n_members:,})",
                     fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xticks(X_BINS[::2])
        ax.set_xticklabels([BIN_LABELS[i] for i in range(0, N_BINS, 2)], rotation=35,
                            ha="right", fontsize=7)
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    # Hide empty subplots
    for idx in range(best_k, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Conversation Shape Cluster Gallery",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    save_fig(fig, "02_cluster_gallery.png")
except Exception as e:
    report["warnings"].append(f"Fig 02 error: {e}")
    print(f"  Fig 02 ERROR: {e}")

# -- Figure 03: Cluster distribution ----------------------------------------
try:
    sorted_names = sorted(cluster_dist.keys(), key=lambda n: cluster_dist[n], reverse=True)
    counts = [cluster_dist[n] for n in sorted_names]
    colors = [CLUSTER_COLORS[n] for n in sorted_names]
    total  = sum(counts)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars = ax.barh(sorted_names[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({cnt/total*100:.1f}%)", va="center", fontsize=TICK_SIZE)
    ax.set_xlabel("Number of Conversations", fontsize=LABEL_SIZE)
    ax.set_title("Conversation Shape Distribution", fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_yticklabels([n.replace("_", " ").title() for n in sorted_names[::-1]], fontsize=TICK_SIZE)
    ax.set_xlim(0, max(counts) * 1.22)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_fig(fig, "03_cluster_distribution.png")
except Exception as e:
    report["warnings"].append(f"Fig 03 error: {e}")
    print(f"  Fig 03 ERROR: {e}")

# -- Figure 04: Cluster by conversation type --------------------------------
try:
    if not shape_by_type.empty:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        bottom = np.zeros(len(shape_by_type.columns))
        ctypes = shape_by_type.columns.tolist()
        for cname in cluster_names:
            if cname in shape_by_type.index:
                vals = shape_by_type.loc[cname, ctypes].fillna(0).values
                ax.bar(ctypes, vals * 100, bottom=bottom * 100,
                       color=CLUSTER_COLORS[cname], label=cname.replace("_", " ").title(),
                       alpha=0.85)
                bottom += vals
        ax.set_ylabel("Proportion (%)", fontsize=LABEL_SIZE)
        ax.set_xlabel("Conversation Type", fontsize=LABEL_SIZE)
        ax.set_title("Shape Distribution by Conversation Type\n(normalized to 100% per type)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=TICK_SIZE - 1)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_ylim(0, 110)
        sns.despine(ax=ax)
        plt.tight_layout()
        save_fig(fig, "04_cluster_by_type.png")
    else:
        raise ValueError("shape_by_type is empty")
except Exception as e:
    report["warnings"].append(f"Fig 04 error: {e}")
    print(f"  Fig 04 ERROR: {e}")

# -- Figure 05: Cluster by model era ----------------------------------------
try:
    if not shape_by_era.empty and shape_by_era.shape[1] >= 1:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        bottom = np.zeros(len(shape_by_era.columns))
        eras   = shape_by_era.columns.tolist()
        for cname in cluster_names:
            if cname in shape_by_era.index:
                vals = shape_by_era.loc[cname, eras].fillna(0).values
                ax.bar([str(e) for e in eras], vals * 100, bottom=bottom * 100,
                       color=CLUSTER_COLORS[cname], label=cname.replace("_", " ").title(),
                       alpha=0.85)
                bottom += vals
        ax.set_ylabel("Proportion (%)", fontsize=LABEL_SIZE)
        ax.set_xlabel("Model Era", fontsize=LABEL_SIZE)
        ax.set_title("Shape Distribution by Model Era\n(normalized to 100% per era)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=TICK_SIZE - 1)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.set_ylim(0, 110)
        sns.despine(ax=ax)
        plt.tight_layout()
        save_fig(fig, "05_cluster_by_era.png")
    else:
        raise ValueError("Only one model era - creating informational figure")
except Exception as e:
    # Create a minimal figure since only one era exists
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        labels = [n.replace("_", " ").title() for n in sorted_names]
        vals   = [cluster_dist[n] for n in sorted_names]
        ax.bar(labels, vals,
               color=[CLUSTER_COLORS[n] for n in sorted_names], alpha=0.85)
        ax.set_xlabel("Shape Cluster", fontsize=LABEL_SIZE)
        ax.set_ylabel("Count", fontsize=LABEL_SIZE)
        ax.set_title("Shape Distribution\n(single model era: o3 — era breakdown not applicable)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        sns.despine(ax=ax)
        plt.tight_layout()
        save_fig(fig, "05_cluster_by_era.png")
    except Exception as e2:
        report["warnings"].append(f"Fig 05 fallback error: {e2}")
        print(f"  Fig 05 ERROR: {e2}")

# -- Figure 06: Cluster trend (HTML) ----------------------------------------
try:
    if not monthly_shapes.empty:
        fig6 = go.Figure()
        ym_vals = monthly_shapes.index.tolist()
        for cname in cluster_names:
            if cname in monthly_shapes.columns:
                ys = (monthly_shapes[cname] * 100).tolist()
                fig6.add_trace(go.Scatter(
                    x=ym_vals, y=ys, mode="lines", stackgroup="one",
                    name=cname.replace("_", " ").title(),
                    fillcolor=CLUSTER_COLORS_HEX.get(cname, "#999999"),
                    line=dict(color=CLUSTER_COLORS_HEX.get(cname, "#999999")),
                    hovertemplate=f"{cname}: %{{y:.1f}}%<extra></extra>",
                ))
        # Era boundaries
        ym_min = min(ym_vals) if ym_vals else "2000-01"
        ym_max = max(ym_vals) if ym_vals else "2099-12"
        for ename, edate in era_boundaries.items():
            eym = edate[:7]
            if ym_min <= eym <= ym_max:
                try:
                    fig6.add_vline(x=eym, line_dash="dot", line_color="gray",
                                   annotation_text=ename.replace("_", " "),
                                   annotation_position="top left", annotation_font_size=9)
                except Exception:
                    pass
        fig6.update_layout(
            title="Conversation Shape Distribution Over Time",
            xaxis_title="Month", yaxis_title="% of Conversations",
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig6.write_html(figpath("06_cluster_trend.html"), include_plotlyjs=True)
        figures_generated.append("outputs/figures/conversation_shapes/06_cluster_trend.html")
        print("  Saved: 06_cluster_trend.html")
    else:
        raise ValueError("monthly_shapes is empty")
except Exception as e:
    report["warnings"].append(f"Fig 06 error: {e}")
    print(f"  Fig 06 ERROR: {e}")

# -- Figure 07: User vs. assistant shapes -----------------------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: avg user + asst shapes with std band
    ax1.fill_between(X_BINS, avg_user_shape - std_user_shape,
                     avg_user_shape + std_user_shape,
                     color=COLOR_PRIMARY, alpha=0.12)
    ax1.fill_between(X_BINS, avg_asst_shape - std_asst_shape,
                     avg_asst_shape + std_asst_shape,
                     color="#A9D18E", alpha=0.18)
    ax1.plot(X_BINS, avg_user_shape, color=COLOR_PRIMARY, linewidth=2.5,
             marker="o", markersize=5, label="User (avg)")
    ax1.plot(X_BINS, avg_asst_shape, color="#52973E", linewidth=2.5,
             marker="s", markersize=5, label="Assistant (avg)")
    ax1.set_xticks(X_BINS[::2])
    ax1.set_xticklabels([BIN_LABELS[i] for i in range(0, N_BINS, 2)], rotation=35,
                        ha="right", fontsize=8)
    ax1.set_ylabel("Normalized Token Density", fontsize=LABEL_SIZE)
    ax1.set_title("Average Shape Trajectory\nby Role", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.legend(fontsize=TICK_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax1)

    # Right: histogram of agreement scores
    if agg_scores:
        ax2.hist(agg_scores, bins=40, color=COLOR_PRIMARY, alpha=0.75,
                 edgecolor="white", linewidth=0.4)
        ax2.axvline(0.8,  color=COLOR_ACCENT,    linestyle="--", linewidth=1.5, label="Sync threshold (0.8)")
        ax2.axvline(-0.3, color=COLOR_SECONDARY, linestyle=":",  linewidth=1.5, label="Inverse threshold (-0.3)")
        n_total  = len(agg_scores)
        n_sync   = sum(s > 0.8 for s in agg_scores)
        n_inv    = sum(s < -0.3 for s in agg_scores)
        n_indep  = n_total - n_sync - n_inv
        ymax = ax2.get_ylim()[1]
        ax2.text(0.85,  0.92, f"Sync:\n{n_sync/n_total*100:.0f}%",
                 transform=ax2.transAxes, ha="center", fontsize=9, color=COLOR_ACCENT)
        ax2.text(0.05,  0.92, f"Inv:\n{n_inv/n_total*100:.0f}%",
                 transform=ax2.transAxes, ha="center", fontsize=9, color=COLOR_SECONDARY)
        ax2.legend(fontsize=TICK_SIZE - 1)
    ax2.set_xlabel("Cosine Similarity (User vs. Assistant Shape)", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax2.set_title("User-Assistant Shape Agreement\nDistribution", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax2)

    plt.suptitle("User vs. Assistant Conversation Shapes",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "07_user_vs_assistant_shapes.png")
except Exception as e:
    report["warnings"].append(f"Fig 07 error: {e}")
    print(f"  Fig 07 ERROR: {e}")

# -- Figure 08: Shape features scatter (HTML) --------------------------------
try:
    feat_df = shape_features.reset_index().merge(
        cluster_df[["shape_cluster_name"]].reset_index(), on="conversation_id"
    ).merge(
        shape_conv[["conversation_id", "msg_count", "turns", "duration_minutes"]],
        on="conversation_id", how="left"
    )
    if len(feat_df) > 5000:
        feat_df = feat_df.sample(5000, random_state=42)

    fig8 = go.Figure()
    for cname in cluster_names:
        sub = feat_df[feat_df["shape_cluster_name"] == cname]
        if len(sub) == 0:
            continue
        sizes = np.clip(sub["msg_count"].fillna(10) / 3, 4, 20).tolist()
        fig8.add_trace(go.Scatter(
            x=sub["slope"], y=sub["volatility"],
            mode="markers",
            marker=dict(color=CLUSTER_COLORS_HEX.get(cname, "#999999"),
                        size=sizes, opacity=0.65,
                        line=dict(width=0.3, color="white")),
            name=cname.replace("_", " ").title(),
            customdata=sub[["conversation_id", "turns", "duration_minutes"]].values,
            hovertemplate=(
                "ID: %{customdata[0]}<br>"
                "Cluster: " + cname + "<br>"
                "Slope: %{x:.3f}<br>Volatility: %{y:.3f}<br>"
                "Turns: %{customdata[1]:.0f}<br>"
                "Duration: %{customdata[2]:.1f}m<extra></extra>"
            ),
        ))
    fig8.update_layout(
        title="Shape Feature Space: Slope vs. Volatility",
        xaxis_title="Slope (negative=tapering, positive=deepening)",
        yaxis_title="Volatility (higher=more zig-zag)",
        hovermode="closest", template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=0.5),
    )
    fig8.write_html(figpath("08_shape_features_scatter.html"), include_plotlyjs=True)
    figures_generated.append("outputs/figures/conversation_shapes/08_shape_features_scatter.html")
    print("  Saved: 08_shape_features_scatter.html")
except Exception as e:
    report["warnings"].append(f"Fig 08 error: {e}")
    print(f"  Fig 08 ERROR: {e}")

# -- Figure 09: Shape stability ----------------------------------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: dominant shape % per conversation type
    if stability_by_type:
        ctypes_st = list(stability_by_type.keys())
        dom_pcts  = [stability_by_type[c]["dominant_pct"] for c in ctypes_st]
        dom_names = [stability_by_type[c]["dominant_shape"] for c in ctypes_st]
        sort_idx  = np.argsort(dom_pcts)[::-1]
        ctypes_sorted   = [ctypes_st[i] for i in sort_idx]
        dom_pcts_sorted = [dom_pcts[i] for i in sort_idx]
        bars = ax1.bar(ctypes_sorted, dom_pcts_sorted,
                       color=[CLUSTER_COLORS.get(stability_by_type[c]["dominant_shape"],
                               (0.5, 0.5, 0.5, 1.0)) for c in ctypes_sorted],
                       alpha=0.85)
        ax1.axhline(100 / best_k, color="gray", linestyle="--",
                    linewidth=1.2, label=f"Chance (1/k={100/best_k:.0f}%)")
        ax1.set_ylabel("Dominant Shape %", fontsize=LABEL_SIZE)
        ax1.set_xlabel("Conversation Type", fontsize=LABEL_SIZE)
        ax1.set_title("Shape Concentration\nby Conversation Type",
                      fontsize=LABEL_SIZE, fontweight="bold")
        ax1.legend(fontsize=TICK_SIZE - 1)
        ax1.tick_params(labelsize=TICK_SIZE)
        plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")
    sns.despine(ax=ax1)

    # Right: streak length histogram
    if len(run_lengths) > 0:
        ax2.hist(run_lengths["streak_length"], bins=min(30, run_lengths["streak_length"].max()),
                 color=COLOR_PRIMARY, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax2.axvline(run_lengths["streak_length"].median(), color=COLOR_ACCENT, linestyle="--",
                    linewidth=1.8, label=f"Median ({run_lengths['streak_length'].median():.0f})")
        ax2.axvline(run_lengths["streak_length"].max(), color="darkred", linestyle=":",
                    linewidth=1.5, label=f"Max ({int(run_lengths['streak_length'].max())})")
        ax2.set_xlabel("Consecutive Same-Shape Streak Length", fontsize=LABEL_SIZE)
        ax2.set_ylabel("Count", fontsize=LABEL_SIZE)
        ax2.set_title("Shape Streak Distribution",
                      fontsize=LABEL_SIZE, fontweight="bold")
        ax2.legend(fontsize=TICK_SIZE - 1)
        ax2.tick_params(labelsize=TICK_SIZE)
    sns.despine(ax=ax2)

    plt.suptitle("Conversation Shape Stability", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "09_shape_stability.png")
except Exception as e:
    report["warnings"].append(f"Fig 09 error: {e}")
    print(f"  Fig 09 ERROR: {e}")

# -- Figure 10: Extreme shapes ----------------------------------------------
try:
    n_ex_clusters = min(6, best_k)
    n_outlier_show = min(4, len(outliers.get("conversation_ids", [])))
    total_subplots = n_ex_clusters + n_outlier_show
    if total_subplots == 0:
        raise ValueError("No subplots to show")

    ncols = min(4, total_subplots)
    nrows = int(np.ceil(total_subplots / ncols))
    nrows = max(2, nrows)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 3),
                              constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    ax_idx = 0
    for ci, cname in enumerate(cluster_names[:n_ex_clusters]):
        ex_ids = exemplars.get(cname, {}).get("conversation_ids", [])
        if not ex_ids:
            continue
        cid_ex = ex_ids[0]
        ex_raw = shape_msgs[shape_msgs["conversation_id"] == cid_ex].sort_values("msg_index")
        ax = axes_flat[ax_idx]
        colors = [COLOR_PRIMARY if r == "user" else "#A9D18E"
                  for r in ex_raw["role"]]
        ax.bar(ex_raw["msg_index"], ex_raw["token_count"].fillna(0),
               color=colors, alpha=0.8, width=0.85)
        ax.set_title(f"Most Typical:\n{cname.replace('_', ' ').title()}",
                     fontsize=9, fontweight="bold",
                     color=CLUSTER_COLORS[cname])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Msg Index", fontsize=8)
        ax.set_ylabel("Tokens", fontsize=8)
        sns.despine(ax=ax)
        ax_idx += 1

    for oi in range(n_outlier_show):
        if ax_idx >= len(axes_flat):
            break
        ocid = outliers["conversation_ids"][oi]
        odist = outliers["distances"][oi]
        ocluster = outliers["assigned_clusters"][oi]
        ex_raw = shape_msgs[shape_msgs["conversation_id"] == ocid].sort_values("msg_index")
        ax = axes_flat[ax_idx]
        colors = [COLOR_ACCENT if r == "user" else "#F4B183" for r in ex_raw["role"]]
        ax.bar(ex_raw["msg_index"], ex_raw["token_count"].fillna(0),
               color=colors, alpha=0.8, width=0.85)
        ax.set_title(f"Outlier (d={odist:.2f})\nAssigned: {ocluster.replace('_', ' ').title()}",
                     fontsize=9, fontweight="bold", color=COLOR_ACCENT)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Msg Index", fontsize=8)
        ax.set_ylabel("Tokens", fontsize=8)
        sns.despine(ax=ax)
        ax_idx += 1

    for idx in range(ax_idx, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Extreme Shapes: Most Typical Exemplars & Outliers",
                 fontsize=TITLE_SIZE, fontweight="bold")
    save_fig(fig, "10_extreme_shapes.png")
except Exception as e:
    report["warnings"].append(f"Fig 10 error: {e}")
    print(f"  Fig 10 ERROR: {e}")

# -- Figure 11: Shape signature summary -------------------------------------
try:
    fig = plt.figure(figsize=FIGSIZE_TALL)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.7, wspace=0.5)
    ax_title   = fig.add_subplot(gs[0, :])        # title + key stats
    ax_entropy = fig.add_subplot(gs[1, 0])         # entropy gauge
    ax_c1 = fig.add_subplot(gs[1, 1])
    ax_c2 = fig.add_subplot(gs[1, 2])
    ax_c3 = fig.add_subplot(gs[2, :])              # 3rd centroid + monthly sparkline

    # Title panel
    ax_title.axis("off")
    sorted_names_fig11 = sorted(cluster_dist.keys(), key=cluster_dist.get, reverse=True)
    dom = sorted_names_fig11[0]
    dom_pct = cluster_pcts.get(dom, 0) * 100

    # key callouts from analysis
    try:
        code_shapes = {n: cluster_profiles[n]["pct_code"] for n in cluster_names if n in cluster_profiles}
        top_code_shape = max(code_shapes, key=code_shapes.get) if code_shapes else "N/A"
    except Exception:
        top_code_shape = "N/A"

    v_val = stat_tests.get("shape_vs_model_era", {}).get("cramers_v")
    era_note = f"Cramer's V = {v_val:.3f}" if v_val is not None else "single era"

    norm_ent_val = stability_report.get("normalized_entropy", 0.5)
    flex_label = "rigid" if norm_ent_val < 0.33 else ("flexible" if norm_ent_val > 0.66 else "moderate")

    callouts = (
        f"Dominant shape: {dom.replace('_', ' ').upper()}  ({dom_pct:.1f}% of conversations)\n"
        f"Shape diversity: {flex_label} (normalized entropy = {norm_ent_val:.2f})\n"
        f"Top code conversation shape: {top_code_shape.replace('_', ' ').title()}\n"
        f"Shape vs. model eras: {era_note}"
    )
    ax_title.text(0.5, 0.5, callouts, transform=ax_title.transAxes,
                  ha="center", va="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8",
                             edgecolor="#BDD7EE", linewidth=1.5),
                  family="monospace")

    # Entropy gauge
    ax_entropy.axis("off")
    gauge_val = float(np.clip(norm_ent_val, 0, 1))
    gauge_color = plt.cm.RdYlGn(gauge_val)
    ax_entropy.barh([0], [gauge_val], height=0.5, color=gauge_color, alpha=0.9)
    ax_entropy.barh([0], [1 - gauge_val], left=[gauge_val], height=0.5,
                    color="lightgray", alpha=0.5)
    ax_entropy.set_xlim(0, 1)
    ax_entropy.set_ylim(-0.5, 1.0)
    ax_entropy.text(0.5, 0.75, "Shape Diversity", ha="center", fontsize=9, fontweight="bold",
                    transform=ax_entropy.transAxes)
    ax_entropy.text(0.0, -0.4, "Rigid", ha="left", fontsize=8, color="gray")
    ax_entropy.text(1.0, -0.4, "Flexible", ha="right", fontsize=8, color="gray")
    ax_entropy.text(gauge_val, 0.5, f" {gauge_val:.2f}", ha="left" if gauge_val < 0.85 else "right",
                    va="center", fontsize=9, fontweight="bold")

    # Top 3 cluster centroids
    for panel_ax, nm in zip([ax_c1, ax_c2, ax_c3], sorted_names_fig11[:3]):
        cidx = cluster_names.index(nm) if nm in cluster_names else 0
        cnt  = cluster_dist.get(nm, 0)
        panel_ax.plot(X_BINS, cluster_centers[cidx],
                      color=CLUSTER_COLORS[nm], linewidth=2, marker="o", markersize=4)
        panel_ax.fill_between(X_BINS, cluster_centers[cidx], alpha=0.15, color=CLUSTER_COLORS[nm])
        panel_ax.set_title(f"{nm.replace('_', ' ').title()}\n(n={cnt})",
                            fontsize=9, fontweight="bold")
        panel_ax.set_xticks([])
        panel_ax.set_yticks([0, 0.5, 1])
        panel_ax.set_ylim(-0.05, 1.1)
        panel_ax.tick_params(labelsize=7)
        sns.despine(ax=panel_ax)

    plt.suptitle("Conversation Shapes — Cognitive Signature Fragment",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    save_fig(fig, "11_shape_signature.png")
except Exception as e:
    report["warnings"].append(f"Fig 11 error: {e}")
    print(f"  Fig 11 ERROR: {e}")

# -- Figure 12: Cluster explorer (HTML) -------------------------------------
try:
    feat_df_ex = shape_features.reset_index().merge(
        cluster_df[["shape_cluster_name"]].reset_index(), on="conversation_id"
    ).merge(
        shape_conv[["conversation_id", "msg_count", "turns"]],
        on="conversation_id", how="left"
    )
    if len(feat_df_ex) > 5000:
        feat_df_ex = feat_df_ex.sample(5000, random_state=42)

    fig12 = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=["Shape Feature Space (Slope vs. Volatility) — hover for shape",
                        "Cluster Centroids"],
    )

    # Left: scatter (slope vs. volatility)
    for i, cname in enumerate(cluster_names):
        sub = feat_df_ex[feat_df_ex["shape_cluster_name"] == cname]
        if len(sub) == 0:
            continue
        fig12.add_trace(
            go.Scatter(
                x=sub["slope"], y=sub["volatility"],
                mode="markers",
                marker=dict(color=CLUSTER_COLORS_HEX.get(cname, "#999999"),
                            size=6, opacity=0.6),
                name=cname.replace("_", " ").title(),
                customdata=sub[["conversation_id", "turns", "msg_count"]].values,
                hovertemplate=(
                    f"Cluster: {cname}<br>"
                    "Slope: %{x:.3f}  Volatility: %{y:.3f}<br>"
                    "Turns: %{customdata[1]:.0f}  "
                    "Msgs: %{customdata[2]:.0f}<extra></extra>"
                ),
                legendgroup=cname,
            ),
            row=1, col=1,
        )

    # Right: centroid lines (all clusters on same axes, different colors)
    for i, cname in enumerate(cluster_names):
        cnt = cluster_dist.get(cname, 0)
        fig12.add_trace(
            go.Scatter(
                x=list(range(N_BINS)), y=cluster_centers[i].tolist(),
                mode="lines+markers",
                marker=dict(size=5, color=CLUSTER_COLORS_HEX.get(cname, "#999999")),
                line=dict(color=CLUSTER_COLORS_HEX.get(cname, "#999999"), width=2.5),
                name=f"{cname.replace('_', ' ').title()} (n={cnt})",
                legendgroup=cname,
                showlegend=False,
                hovertemplate=f"Cluster: {cname}<br>Bin: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig12.update_xaxes(title_text="Slope", row=1, col=1)
    fig12.update_yaxes(title_text="Volatility", row=1, col=1)
    fig12.update_xaxes(title_text="Position Bin", row=1, col=2,
                       tickvals=list(range(N_BINS)),
                       ticktext=[f"{i*10}%" for i in range(N_BINS)])
    fig12.update_yaxes(title_text="Normalized Length", row=1, col=2)
    fig12.update_layout(
        title="Conversation Shape Explorer",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=10)),
        height=520,
    )
    fig12.write_html(figpath("12_cluster_explorer.html"), include_plotlyjs=True)
    figures_generated.append("outputs/figures/conversation_shapes/12_cluster_explorer.html")
    print("  Saved: 12_cluster_explorer.html")
except Exception as e:
    report["warnings"].append(f"Fig 12 error: {e}")
    print(f"  Fig 12 ERROR: {e}")

# ===========================================================================
# COGNITIVE SIGNATURE FRAGMENT
# ===========================================================================
try:
    # Build the narrative summary
    norm_ent_val   = stability_report.get("normalized_entropy", 0.5)
    dom_shape_name = stability_report.get("dominant_shape", dominant_shape)
    dom_shape_pct  = stability_report.get("dominant_shape_pct", 0.0)
    flex_word      = ("rigid" if norm_ent_val < 0.33
                      else "flexible" if norm_ent_val > 0.66 else "moderately diverse")
    era_v = stat_tests.get("shape_vs_model_era", {}).get("cramers_v")
    era_word = ("stable across model eras" if era_v is None
                else ("sensitive to model era changes" if era_v > 0.15
                      else "stable across model eras"))
    era_cramer = f"Cramer's V = {era_v:.3f}" if era_v is not None else "not computable (single era)"

    type_v = stat_tests.get("shape_vs_conversation_type", {}).get("cramers_v")
    type_word = ("independent of conversation type" if type_v is None or type_v < 0.1
                 else "associated with conversation type")

    u_cos = uva_report.get("overall_cosine_similarity")
    align_word = ("highly synchronized" if u_cos is not None and u_cos > 0.8
                  else "moderately synchronized" if u_cos is not None and u_cos > 0.5
                  else "weakly aligned" if u_cos is not None else "unknown")

    summary = (
        f"The user's most common conversation shape is '{dom_shape_name}' "
        f"({dom_shape_pct:.1f}% of shape-eligible conversations). "
        f"Shape usage is {flex_word} (normalized entropy {norm_ent_val:.2f}), "
        f"suggesting {'a consistent structural style' if norm_ent_val < 0.33 else 'variety in conversational structure'}. "
        f"Conversation shapes are {era_word} ({era_cramer}) and {type_word}. "
        f"User and assistant shape trajectories are {align_word} on average "
        f"(cosine similarity {u_cos:.3f})." if u_cos is not None else "."
    )

    cognitive_sig_fragment = {
        "dominant_shape": dom_shape_name,
        "shape_flexibility": flex_word,
        "shape_era_sensitive": (era_v is not None and era_v > 0.15) if era_v is not None else None,
        "user_assistant_alignment": align_word,
        "summary": summary,
    }
except Exception as e:
    cognitive_sig_fragment = {
        "dominant_shape": dominant_shape,
        "shape_flexibility": "unknown",
        "shape_era_sensitive": None,
        "user_assistant_alignment": "unknown",
        "summary": "Summary generation failed.",
    }
    report["warnings"].append(f"Cognitive signature fragment error: {e}")

# ===========================================================================
# BUILD & WRITE REPORT
# ===========================================================================
print("\n-- Writing report ------------------------------------------------------")

report.update({
    "input_data": report.get("input_data", {}),
    "methodology": {
        "n_bins": N_BINS,
        "normalization": "min_max",
        "clustering_method": clustering_method,
        "k_selected": best_k,
        "silhouette_score": clean(best_silhouette) if "best_silhouette" in dir() else None,
        "silhouette_scores_tested": silhouette_scores_tested,
    },
    "clusters": {
        "names": cluster_names,
        "distribution": {k: int(v) for k, v in cluster_dist.items()},
        "distribution_pct": {k: clean(v * 100) for k, v in cluster_pcts.items()},
        "profiles": cluster_analysis_report.get("profiles", {}),
        "exemplars": clean_dict(exemplars),
        "outliers": clean_dict(outliers),
    },
    "cross_tabulations": cluster_analysis_report.get("cross_tabulations", {}),
    "statistical_tests": stat_tests,
    "user_vs_assistant_shapes": uva_report,
    "stability": stability_report,
    "shape_features_summary": clean_dict({
        feat: {
            "mean":   clean(shape_features[feat].mean()),
            "std":    clean(shape_features[feat].std()),
            "median": clean(shape_features[feat].median()),
        }
        for feat in FEATURE_COLS if feat in shape_features.columns
    }),
    "cognitive_signature_fragment": cognitive_sig_fragment,
    "figures_generated": figures_generated,
    "data_outputs": ["data/processed/shape_clusters.parquet"],
})

clean_report = clean_dict(report)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_report, f, indent=2, ensure_ascii=False)
print(f"  Report written to: {REPORT_PATH}")

# ===========================================================================
# VALIDATION CHECKLIST
# ===========================================================================
print("\n== VALIDATION CHECKLIST =============================================")

checks = {}

# 1. Report exists with all keys
required_top_keys = {"input_data", "methodology", "clusters", "cross_tabulations",
                     "statistical_tests", "user_vs_assistant_shapes", "stability",
                     "shape_features_summary", "cognitive_signature_fragment",
                     "figures_generated", "data_outputs"}
checks["report_exists_all_keys"] = (
    os.path.isfile(REPORT_PATH) and required_top_keys.issubset(clean_report.keys())
)

# 2. shape_clusters.parquet exists with correct columns
required_pcols = {"shape_cluster_id", "shape_cluster_name"} | set(FEATURE_COLS)
try:
    pcdf = pd.read_parquet(CLUSTER_PATH)
    checks["shape_clusters_parquet_exists_correct_cols"] = required_pcols.issubset(set(pcdf.columns) | {pcdf.index.name})
    checks["shape_clusters_row_count_matches"] = (len(pcdf) == clean_report.get("input_data", {}).get("shape_eligible", -1))
    # All conv_ids in shape_clusters exist in conversations_clean
    known_ids = set(conversations["conversation_id"])
    checks["all_cluster_ids_in_conversations"] = set(pcdf.index).issubset(known_ids)
except Exception as pe:
    checks["shape_clusters_parquet_exists_correct_cols"] = False
    checks["shape_clusters_row_count_matches"] = False
    checks["all_cluster_ids_in_conversations"] = False
    print(f"  Parquet check error: {pe}")

# 3. Cluster distribution sums to shape-eligible
dist_sum = sum(clean_report.get("clusters", {}).get("distribution", {}).values())
checks["cluster_dist_sums_to_eligible"] = (dist_sum == clean_report.get("input_data", {}).get("shape_eligible", -1))

# 4. All 12 figures exist
expected_figs = [
    "01_shape_vector_method.png", "02_cluster_gallery.png", "03_cluster_distribution.png",
    "04_cluster_by_type.png", "05_cluster_by_era.png", "06_cluster_trend.html",
    "07_user_vs_assistant_shapes.png", "08_shape_features_scatter.html",
    "09_shape_stability.png", "10_extreme_shapes.png", "11_shape_signature.png",
    "12_cluster_explorer.html",
]
all_figs_exist = all(os.path.isfile(figpath(f)) for f in expected_figs)
checks["all_12_figures_exist"] = all_figs_exist
if not all_figs_exist:
    for f in expected_figs:
        if not os.path.isfile(figpath(f)):
            print(f"  MISSING: {f}")

# 5. PNGs >= 10KB
png_figs = [f for f in expected_figs if f.endswith(".png")]
all_pngs_ok = all(
    os.path.isfile(figpath(f)) and os.path.getsize(figpath(f)) >= 10_240
    for f in png_figs
)
checks["all_pngs_gte_10kb"] = all_pngs_ok
if not all_pngs_ok:
    for f in png_figs:
        p = figpath(f)
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        if sz < 10_240:
            print(f"  SMALL PNG ({sz}B): {f}")

# 6. HTML files have content
html_figs = [f for f in expected_figs if f.endswith(".html")]
all_htmls_ok = all(
    os.path.isfile(figpath(f)) and os.path.getsize(figpath(f)) > 1000
    for f in html_figs
)
checks["html_figures_have_content"] = all_htmls_ok

# 7. Silhouette scores in [-1, 1]
sil_ok = all(
    isinstance(v, (int, float)) and -1 <= v <= 1
    for v in silhouette_scores_tested.values()
)
checks["silhouette_scores_valid"] = sil_ok

# 8. No cluster has 0 members
checks["no_empty_clusters"] = all(
    v > 0 for v in clean_report.get("clusters", {}).get("distribution", {}).values()
)

# 9. No NaN/Inf in report
def check_no_nan(obj, path=""):
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return False, path
    if isinstance(d := obj, dict):
        for k, v in d.items():
            ok, p = check_no_nan(v, f"{path}.{k}")
            if not ok:
                return False, p
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            ok, p = check_no_nan(v, f"{path}[{i}]")
            if not ok:
                return False, p
    return True, ""

no_nan, nan_path = check_no_nan(clean_report)
checks["no_nan_inf_in_report"] = no_nan
if not no_nan:
    print(f"  NaN/Inf at: {nan_path}")

# 10. Cognitive signature fragment summary is non-empty
csf = clean_report.get("cognitive_signature_fragment", {})
checks["cognitive_signature_summary_nonempty"] = (
    isinstance(csf.get("summary"), str) and len(csf.get("summary", "")) > 10
)

print()
for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name}")

all_pass = all(checks.values())
print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

if report["warnings"]:
    print(f"\nWarnings ({len(report['warnings'])}):")
    for w in report["warnings"]:
        print(f"  * {str(w)[:150]}")
