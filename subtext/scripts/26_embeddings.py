"""
Module 26: Conversation Embeddings & Semantic Search
Generates sentence-transformer embeddings for all analysable conversations,
builds a FAISS index for semantic retrieval, and runs clustering/similarity
analyses.

Zero API cost -- local computation only.

Usage:
    python scripts/26_embeddings.py

Dependencies:
    pip install sentence-transformers faiss-cpu umap-learn hdbscan
    (First run downloads ~420 MB model from HuggingFace.)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import warnings
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
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "outputs" / "reports"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "embeddings"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "embeddings"

CONVOS_PATH = PROCESSED / "conversations_clean.parquet"
SUMMARIES_PATH = PROCESSED / "conversation_summaries.parquet"

OUT_PARQUET = PROCESSED / "conversation_embeddings.parquet"
OUT_FAISS = MODEL_DIR / "conversation_embeddings.index"
OUT_REPORT = REPORTS / "embeddings_report.json"

for d in [REPORTS, FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER = 15
HDBSCAN_MIN_SAMPLES = 3
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = [
    "#2E75B6", "#C55A11", "#E91E90", "#E15759",
    "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2",
    "#FF9DA7", "#9C755F", "#BAB0AC", "#4E79A7",
]
DPI = 150
sns.set_theme(style="whitegrid", font_scale=0.9)


# ---------------------------------------------------------------------------
# Utilities
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


# ---------------------------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------------------------
def load_data():
    """Load conversations + summaries, filter to analysable."""
    print("\n=== Step 1: Loading Data ===")

    convos = pd.read_parquet(CONVOS_PATH)
    summaries = pd.read_parquet(SUMMARIES_PATH)

    # Inner join -- only conversations with summaries
    df = convos[convos["is_analysable"]].merge(
        summaries[["conversation_id", "summary"]],
        on="conversation_id",
        how="inner",
    )

    # Drop rows with missing or empty summaries
    df = df[df["summary"].notna() & (df["summary"].str.strip() != "")]
    df = df.sort_values("created_at").reset_index(drop=True)

    print(f"  Analysable conversations with summaries: {len(df)}")
    print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Generate Embeddings
# ---------------------------------------------------------------------------
def generate_embeddings(df):
    """Encode summaries and titles using sentence-transformers."""
    print("\n=== Step 2: Generating Embeddings ===")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  (First run will download ~420 MB model from HuggingFace)")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed summaries
    summaries = df["summary"].tolist()
    print(f"  Encoding {len(summaries)} summaries...")
    summary_embs = model.encode(
        summaries,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Embed titles
    titles = df["title"].fillna("").tolist()
    print(f"  Encoding {len(titles)} titles...")
    title_embs = model.encode(
        titles,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  Summary embeddings shape: {summary_embs.shape}")
    print(f"  Title embeddings shape: {title_embs.shape}")
    return summary_embs, title_embs


# ---------------------------------------------------------------------------
# Step 3: Build FAISS Index
# ---------------------------------------------------------------------------
def build_faiss_index(summary_embs):
    """Build and save a FAISS inner-product index."""
    print("\n=== Step 3: Building FAISS Index ===")
    import faiss

    dim = summary_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine
    index.add(summary_embs.astype(np.float32))

    faiss.write_index(index, str(OUT_FAISS))
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    print(f"  Saved: {OUT_FAISS}")
    return index


# ---------------------------------------------------------------------------
# Step 4: UMAP Projection
# ---------------------------------------------------------------------------
def compute_umap(summary_embs):
    """Reduce to 2D with UMAP for visualization."""
    print("\n=== Step 4: UMAP Projection ===")
    import umap

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        n_components=2,
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(summary_embs)
    print(f"  UMAP complete: {coords.shape}")
    return coords


# ---------------------------------------------------------------------------
# Step 5: Clustering
# ---------------------------------------------------------------------------
def compute_clusters(umap_coords):
    """Run HDBSCAN on UMAP coordinates."""
    print("\n=== Step 5: HDBSCAN Clustering ===")
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(umap_coords)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    return labels, n_clusters, n_noise


# ---------------------------------------------------------------------------
# Step 6: Similarity & Nearest Neighbors
# ---------------------------------------------------------------------------
def compute_similarity(summary_embs):
    """Compute pairwise cosine similarity and nearest neighbors."""
    print("\n=== Step 6: Pairwise Similarity ===")

    # Since embeddings are L2-normalized, dot product = cosine similarity
    sim_matrix = summary_embs @ summary_embs.T

    # Nearest neighbor (exclude self)
    np.fill_diagonal(sim_matrix, -1)
    nn_indices = sim_matrix.argmax(axis=1)
    nn_distances = sim_matrix[np.arange(len(sim_matrix)), nn_indices]

    # Restore diagonal
    np.fill_diagonal(sim_matrix, 1.0)

    # Mean similarity per conversation (centrality)
    mean_sim = sim_matrix.mean(axis=1)

    print(f"  Mean pairwise similarity: {mean_sim.mean():.3f}")
    print(f"  Mean nearest-neighbor similarity: {nn_distances.mean():.3f}")
    return sim_matrix, nn_indices, nn_distances, mean_sim


# ---------------------------------------------------------------------------
# Step 7: Semantic Drift
# ---------------------------------------------------------------------------
def compute_drift(summary_embs, df):
    """Compute semantic distance from corpus centroid over time."""
    print("\n=== Step 7: Semantic Drift ===")

    centroid = summary_embs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # re-normalize

    # Cosine distance from centroid for each conversation
    distances = 1.0 - (summary_embs @ centroid)

    # Weekly aggregation
    drift_df = df[["year_week"]].copy()
    drift_df["centroid_distance"] = distances
    weekly_drift = drift_df.groupby("year_week")["centroid_distance"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    weekly_drift.columns = ["year_week", "mean_distance", "std_distance", "n_convos"]

    print(f"  Weeks tracked: {len(weekly_drift)}")
    print(f"  Mean centroid distance: {distances.mean():.3f}")
    return distances, weekly_drift


# ---------------------------------------------------------------------------
# Step 8: Figures
# ---------------------------------------------------------------------------
def generate_figures(df, umap_coords, cluster_labels, sim_matrix,
                     nn_distances, mean_sim, weekly_drift):
    """Generate all analysis figures."""
    print("\n=== Step 8: Generating Figures ===")
    figures = []

    # ---- Figure 01: UMAP by conversation type ----
    fig, ax = plt.subplots(figsize=(12, 8))
    types = df["conversation_type"].fillna("unknown")
    unique_types = types.value_counts().index.tolist()
    colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(unique_types)}

    for t in unique_types:
        mask = types == t
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=colors[t], label=t, s=10, alpha=0.6,
        )
    ax.set_title("Semantic Landscape by Conversation Type", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    plt.tight_layout()
    fname = "01_umap_by_type.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 02: UMAP by model era ----
    fig, ax = plt.subplots(figsize=(12, 8))
    eras = df["model_era"].fillna("unknown")
    unique_eras = eras.value_counts().index.tolist()
    era_colors = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(unique_eras)}

    for e in unique_eras:
        mask = eras == e
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=era_colors[e], label=e, s=10, alpha=0.6,
        )
    ax.set_title("Semantic Landscape by Model Era", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    plt.tight_layout()
    fname = "02_umap_by_era.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 03: UMAP by HDBSCAN cluster ----
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_clusters = sorted(set(cluster_labels))
    for cl in unique_clusters:
        mask = cluster_labels == cl
        label = f"Cluster {cl}" if cl >= 0 else "Noise"
        color = "#CCCCCC" if cl == -1 else PALETTE[cl % len(PALETTE)]
        alpha = 0.2 if cl == -1 else 0.6
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=color, label=label, s=10, alpha=alpha,
        )
    ax.set_title("Semantic Clusters (HDBSCAN)", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # Only show legend if reasonable number of clusters
    if len(unique_clusters) <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    plt.tight_layout()
    fname = "03_umap_clusters.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 04: Cluster composition ----
    fig, ax = plt.subplots(figsize=(14, 6))
    cl_df = pd.DataFrame({
        "cluster": cluster_labels,
        "conversation_type": df["conversation_type"].fillna("unknown").values,
    })
    # Exclude noise for composition chart
    cl_valid = cl_df[cl_df["cluster"] >= 0]
    if len(cl_valid) > 0:
        comp = cl_valid.groupby(["cluster", "conversation_type"]).size().unstack(fill_value=0)
        comp_pct = comp.div(comp.sum(axis=1), axis=0)
        comp_pct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:len(comp_pct.columns)])
        ax.set_title("Cluster Composition by Conversation Type", fontsize=14)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Proportion")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, "No clusters found", ha="center", va="center", fontsize=14)
    plt.tight_layout()
    fname = "04_cluster_composition.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 05: Similarity heatmap (top 50 most central) ----
    fig, ax = plt.subplots(figsize=(14, 12))
    top50_idx = np.argsort(mean_sim)[-50:]
    sub_sim = sim_matrix[np.ix_(top50_idx, top50_idx)]
    top50_titles = df.iloc[top50_idx]["title"].fillna("").str[:40].tolist()

    sns.heatmap(
        sub_sim, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
        xticklabels=top50_titles, yticklabels=top50_titles,
        square=True,
    )
    ax.set_title("Pairwise Similarity — 50 Most Central Conversations", fontsize=13)
    ax.tick_params(axis="both", labelsize=5)
    plt.tight_layout()
    fname = "05_similarity_heatmap.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 06: Semantic drift over time ----
    fig, ax = plt.subplots(figsize=(14, 5))
    wd = weekly_drift.copy()
    wd["week_idx"] = range(len(wd))

    ax.scatter(wd["week_idx"], wd["mean_distance"], s=12, alpha=0.4, c=PALETTE[0])

    # Rolling smoothing
    if len(wd) >= 8:
        smoothed = wd["mean_distance"].rolling(8, center=True, min_periods=3).mean()
        ax.plot(wd["week_idx"], smoothed, color=PALETTE[1], linewidth=2, label="8-week rolling mean")
        ax.legend()

    # Label every ~10th week
    tick_step = max(1, len(wd) // 15)
    ax.set_xticks(wd["week_idx"][::tick_step])
    ax.set_xticklabels(wd["year_week"][::tick_step], rotation=45, ha="right", fontsize=7)
    ax.set_title("Semantic Drift from Corpus Centroid Over Time", fontsize=14)
    ax.set_ylabel("Mean Cosine Distance from Centroid")
    ax.set_xlabel("Week")
    plt.tight_layout()
    fname = "06_semantic_drift.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    # ---- Figure 07: Nearest-neighbor distance distribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nn_distances, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(nn_distances.mean(), color=PALETTE[1], linestyle="--", linewidth=2,
               label=f"Mean = {nn_distances.mean():.3f}")
    ax.set_title("Nearest-Neighbor Cosine Similarity Distribution", fontsize=14)
    ax.set_xlabel("Cosine Similarity to Nearest Neighbor")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fname = "07_nn_distance_distribution.png"
    fig.savefig(figpath(fname), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    print(f"  {fname}")

    return figures


# ---------------------------------------------------------------------------
# Step 9: Build Output
# ---------------------------------------------------------------------------
def build_output(df, summary_embs, title_embs, umap_coords, cluster_labels,
                 centroid_distances, nn_indices, nn_distances):
    """Build and save the embeddings parquet."""
    print("\n=== Step 9: Saving Output ===")

    out = pd.DataFrame({
        "conversation_id": df["conversation_id"].values,
        "umap_x": umap_coords[:, 0].astype(np.float32),
        "umap_y": umap_coords[:, 1].astype(np.float32),
        "semantic_cluster": cluster_labels.astype(np.int32),
        "centroid_distance": centroid_distances.astype(np.float32),
        "nn_distance": nn_distances.astype(np.float32),
        "nn_conversation_id": df["conversation_id"].values[nn_indices],
    })

    # Store embeddings as lists of floats
    out["summary_embedding"] = [row.tolist() for row in summary_embs]
    out["title_embedding"] = [row.tolist() for row in title_embs]

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved: {OUT_PARQUET} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Step 10: Report
# ---------------------------------------------------------------------------
def build_report(df, summary_embs, n_clusters, n_noise, cluster_labels,
                 nn_distances, mean_sim, centroid_distances, weekly_drift,
                 figures):
    """Build and save JSON report."""
    print("\n=== Step 10: Building Report ===")

    # Cluster summaries
    cluster_info = {}
    for cl in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == cl
        cl_convos = df[mask]
        types = cl_convos["conversation_type"].value_counts()
        cluster_info[int(cl)] = {
            "size": int(mask.sum()),
            "top_types": types.head(3).to_dict(),
            "sample_titles": cl_convos["title"].fillna("").head(5).tolist(),
        }

    # Key findings
    findings = []
    findings.append(f"Embedded {len(df)} conversations into {EMBEDDING_DIM}-dimensional space using {EMBEDDING_MODEL}.")
    findings.append(f"HDBSCAN found {n_clusters} semantic clusters ({n_noise} noise points, {n_noise/len(df)*100:.1f}%).")
    findings.append(f"Mean nearest-neighbor similarity: {nn_distances.mean():.3f} (higher = more semantic redundancy).")
    findings.append(f"Mean centroid distance: {centroid_distances.mean():.3f} (semantic spread of corpus).")

    # Most and least typical conversations
    most_central_idx = mean_sim.argmax()
    least_central_idx = mean_sim.argmin()
    findings.append(f"Most central conversation: \"{df.iloc[most_central_idx]['title']}\"")
    findings.append(f"Most unique conversation: \"{df.iloc[least_central_idx]['title']}\"")

    # Drift trend
    if len(weekly_drift) >= 10:
        first_half = weekly_drift.iloc[:len(weekly_drift)//2]["mean_distance"].mean()
        second_half = weekly_drift.iloc[len(weekly_drift)//2:]["mean_distance"].mean()
        direction = "increasing" if second_half > first_half else "decreasing"
        findings.append(f"Semantic drift is {direction} over time ({first_half:.3f} first half vs {second_half:.3f} second half).")

    report = {
        "module": "embeddings",
        "module_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_data": {
            "conversations_embedded": len(df),
            "summary_mean_words": round(df["summary"].str.split().str.len().mean(), 1),
        },
        "embedding_model": {
            "name": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIM,
            "normalized": True,
        },
        "faiss_index": {
            "path": str(OUT_FAISS),
            "type": "IndexFlatIP",
            "vectors": len(df),
        },
        "clustering": {
            "method": "HDBSCAN",
            "min_cluster_size": HDBSCAN_MIN_CLUSTER,
            "min_samples": HDBSCAN_MIN_SAMPLES,
            "clusters_found": n_clusters,
            "noise_points": n_noise,
            "noise_pct": round(n_noise / len(df) * 100, 1),
            "cluster_details": clean_dict(cluster_info),
        },
        "similarity": {
            "mean_pairwise": round(float(mean_sim.mean()), 3),
            "mean_nn_similarity": round(float(nn_distances.mean()), 3),
            "median_nn_similarity": round(float(np.median(nn_distances)), 3),
        },
        "semantic_drift": {
            "mean_centroid_distance": round(float(centroid_distances.mean()), 3),
            "weeks_tracked": len(weekly_drift),
        },
        "key_findings": findings,
        "figures_generated": figures,
    }

    report = clean_dict(report)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {OUT_REPORT}")
    return report


# ---------------------------------------------------------------------------
# Step 11: Validation
# ---------------------------------------------------------------------------
def run_validation(out_df, summary_embs, cluster_labels, figures):
    """Run validation checks and print results."""
    print("\n=== Validation ===")
    checks = []

    # 1. Parquet exists
    checks.append(("Embeddings parquet exists", OUT_PARQUET.exists()))

    # 2. Row count matches
    n_expected = len(summary_embs)
    n_actual = len(out_df)
    checks.append((f"Row count matches ({n_actual} = {n_expected})", n_actual == n_expected))

    # 3. Embedding dimensionality
    emb_len = len(out_df["summary_embedding"].iloc[0]) if len(out_df) > 0 else 0
    checks.append((f"Summary embedding dim = {EMBEDDING_DIM} (got {emb_len})", emb_len == EMBEDDING_DIM))

    # 4. L2 normalization check (norms should be ~1.0)
    norms = np.linalg.norm(summary_embs, axis=1)
    norm_ok = np.allclose(norms, 1.0, atol=0.01)
    checks.append((f"Embeddings L2-normalized (mean norm = {norms.mean():.4f})", norm_ok))

    # 5. FAISS index exists and loadable
    faiss_ok = OUT_FAISS.exists()
    if faiss_ok:
        try:
            import faiss
            idx = faiss.read_index(str(OUT_FAISS))
            faiss_ok = idx.ntotal == n_expected
            checks.append((f"FAISS index loads ({idx.ntotal} vectors)", faiss_ok))
        except Exception as e:
            checks.append((f"FAISS index loads (error: {e})", False))
    else:
        checks.append(("FAISS index exists", False))

    # 6. Clusters found
    n_clusters = len(set(cluster_labels) - {-1})
    checks.append((f"HDBSCAN found >= 10 clusters (got {n_clusters})", n_clusters >= 10))

    # 7. All figures exist
    fig_ok = all((FIG_DIR / f).exists() for f in figures) if figures else False
    checks.append((f"All {len(figures)} figures exist", fig_ok))

    # 8. Report exists
    checks.append(("Report JSON exists", OUT_REPORT.exists()))

    # 9. No NaN in key columns
    nan_check = (
        out_df["umap_x"].isna().sum() == 0
        and out_df["umap_y"].isna().sum() == 0
        and out_df["centroid_distance"].isna().sum() == 0
    )
    checks.append(("No NaN in UMAP/distance columns", nan_check))

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
    print("Module 26: Conversation Embeddings & Semantic Search")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Embed
    summary_embs, title_embs = generate_embeddings(df)

    # Step 3: FAISS index
    index = build_faiss_index(summary_embs)

    # Step 4: UMAP
    umap_coords = compute_umap(summary_embs)

    # Step 5: Cluster
    cluster_labels, n_clusters, n_noise = compute_clusters(umap_coords)

    # Step 6: Similarity
    sim_matrix, nn_indices, nn_distances, mean_sim = compute_similarity(summary_embs)

    # Step 7: Drift
    centroid_distances, weekly_drift = compute_drift(summary_embs, df)

    # Step 8: Figures
    figures = generate_figures(
        df, umap_coords, cluster_labels, sim_matrix,
        nn_distances, mean_sim, weekly_drift,
    )

    # Step 9: Output parquet
    out_df = build_output(
        df, summary_embs, title_embs, umap_coords, cluster_labels,
        centroid_distances, nn_indices, nn_distances,
    )

    # Step 10: Report
    report = build_report(
        df, summary_embs, n_clusters, n_noise, cluster_labels,
        nn_distances, mean_sim, centroid_distances, weekly_drift,
        figures,
    )

    # Step 11: Validation
    run_validation(out_df, summary_embs, cluster_labels, figures)

    # Summary
    print("\n" + "=" * 60)
    print("Key Findings:")
    for f in report["key_findings"]:
        print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
