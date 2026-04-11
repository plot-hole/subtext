"""
Module 3.2: Thematic and Content Analysis — Topic Modeling
Script: 11_topics.py

Builds an unsupervised BERTopic model over Claude-generated conversation
summaries to discover recurring thematic clusters. Uses sentence-transformer
embeddings computed from the summaries (not raw messages).

Input:  outputs/reports/all_summaries.csv  (summary column)
Output: topic_info.csv, topic_report.json, topic_assignments.parquet,
        topic_model.html, topic_hierarchy.html, topic_over_time.html,
        topic_heatmap.png, serialized BERTopic model
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
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMMARIES_PATH  = os.path.join(BASE, "outputs", "reports", "all_summaries.csv")
EMBED_DIR       = os.path.join(BASE, "data", "embeddings")
EMBED_PATH      = os.path.join(EMBED_DIR, "summary_embeddings.npy")
CONV_IDS_PATH   = os.path.join(EMBED_DIR, "summary_conversation_ids.json")

OUT_MODEL_DIR   = os.path.join(BASE, "outputs", "models", "topic_model")
OUT_FIG_DIR     = os.path.join(BASE, "outputs", "figures")
OUT_REPORT_DIR  = os.path.join(BASE, "outputs", "reports")
OUT_ASSIGNMENTS = os.path.join(BASE, "data", "processed", "topic_assignments.parquet")
OUT_TOT_PARQUET = os.path.join(BASE, "data", "processed", "topics_over_time.parquet")

# Custom stopwords: English defaults + conversational/summary filler
CUSTOM_STOPS = [
    "like", "just", "im", "dont", "didnt", "doesnt", "ive", "thats", "its",
    "hes", "shes", "theyre", "wont", "cant", "gonna", "wanna", "also",
    "would", "could", "really", "thing", "stuff", "basically", "actually",
    "pretty", "kinda", "lol", "ok", "okay", "yeah", "yea", "got", "get",
    "one", "want", "said", "says", "asked", "told", "went", "going",
    "seems", "something", "well", "much", "around", "back", "still",
    "made", "make", "including", "mentioned", "appears", "involves",
    "various", "several", "specific", "related", "based", "using", "used",
    "user", "assistant", "conversation", "discusses", "discussed",
    "discussing", "requests", "requested", "asking", "looking",
]


def ensure_dirs():
    for d in [EMBED_DIR, OUT_MODEL_DIR, OUT_FIG_DIR, OUT_REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Step 1: Load Summaries ───────────────────────────────────────────────────
def load_summaries(args):
    path = args.summaries if args.summaries else SUMMARIES_PATH
    if not os.path.exists(path):
        print(f"ERROR: Summaries file not found at {path}")
        sys.exit(1)

    print(f"Loading summaries from {path}...")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Filter out empty/failed summaries
    df = df[df["summary"].notna() & (df["summary"].str.strip() != "")].copy()
    df = df[df["summary"] != "[SUMMARIZATION FAILED]"].copy()
    df = df.reset_index(drop=True)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Valid summaries: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── Step 2: Compute or Load Embeddings ───────────────────────────────────────
def load_or_compute_embeddings(docs_df, args):
    embed_path = args.embeddings if args.embeddings else EMBED_PATH
    ids_path = args.conversation_ids if args.conversation_ids else CONV_IDS_PATH

    # Always recompute when using summaries (they may have changed)
    print("Computing embeddings from summaries...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = docs_df["summary"].tolist()
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True
    )
    embeddings = np.array(embeddings)

    # Save for reuse
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    np.save(embed_path, embeddings)
    conversation_ids = docs_df["conversation_id"].tolist()
    with open(ids_path, "w") as f:
        json.dump(conversation_ids, f)
    print(f"  Computed and saved {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]}d)")

    assert len(documents) == embeddings.shape[0], \
        f"Row mismatch: {len(documents)} docs vs {embeddings.shape[0]} embeddings"

    return documents, embeddings


# ── Step 3: Build Topic Model ────────────────────────────────────────────────
def build_topic_model(documents, embeddings, args):
    print("Building BERTopic model...")
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer

    # Explicit embedding model so KeyBERTInspired can embed vocabulary words
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # Vectorizer: English stopwords + custom conversational stopwords
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stops = list(ENGLISH_STOP_WORDS) + CUSTOM_STOPS

    vectorizer_model = CountVectorizer(
        stop_words=all_stops,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95
    )

    # KeyBERT-style reranking for better topic labels
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=True
    )

    topics, probabilities = topic_model.fit_transform(documents, embeddings=embeddings)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"  Initial fit: {n_topics} topics, {n_outliers} outliers ({n_outliers/len(topics)*100:.1f}%)")

    return topic_model, topics, probabilities


# ── Step 4: Reduce Outliers ──────────────────────────────────────────────────
def reduce_outliers(topic_model, documents, topics, embeddings, args):
    if not args.reduce_outliers:
        print("Skipping outlier reduction (disabled)")
        return topics

    n_before = sum(1 for t in topics if t == -1)
    if n_before == 0:
        print("No outliers to reduce")
        return topics

    print(f"Reducing outliers ({n_before} outlier documents)...")
    new_topics = topic_model.reduce_outliers(
        documents,
        topics,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=0.0
    )
    topic_model.update_topics(documents, topics=new_topics)

    n_after = sum(1 for t in new_topics if t == -1)
    print(f"  Reassigned {n_before - n_after} outliers, {n_after} remain as -1 ({n_after/len(new_topics)*100:.1f}%)")
    return new_topics


# ── Step 5: Extract Topic Assignments ────────────────────────────────────────
def extract_topic_assignments(topic_model, docs_df, topics, probabilities):
    print("Extracting topic assignments...")
    topic_info = topic_model.get_topic_info()
    topic_names = topic_info.set_index("Topic")["Name"].to_dict()

    assignments = pd.DataFrame({
        "conversation_id": docs_df["conversation_id"].values,
        "topic_id": topics,
        "topic_probability": [
            float(probabilities[i][t]) if t >= 0 and probabilities is not None else 0.0
            for i, t in enumerate(topics)
        ]
    })

    assignments["topic_name"] = assignments["topic_id"].map(topic_names).fillna("Outlier")

    # Merge dates from summaries
    assignments = assignments.merge(
        docs_df[["conversation_id", "date"]],
        on="conversation_id",
        how="left"
    )
    assignments.rename(columns={"date": "created_at"}, inplace=True)

    print(f"  Assignments: {len(assignments):,} rows")
    return assignments, topic_info, topic_names


# ── Step 6: Topics Over Time ─────────────────────────────────────────────────
def compute_topics_over_time(topic_model, docs_df, documents):
    print("Computing topics over time...")
    timestamps = docs_df["date"].tolist()

    topics_over_time = topic_model.topics_over_time(
        documents,
        timestamps,
        nr_bins=20,
        datetime_format=None,
        evolution_tuning=True,
        global_tuning=True
    )
    print(f"  Topics-over-time rows: {len(topics_over_time):,}")
    return topics_over_time


# ── Step 7: Generate Visualizations ──────────────────────────────────────────
def generate_visualizations(topic_model, topics_over_time, assignments,
                            topic_names, args):
    print("Generating visualizations...")

    # 1. Intertopic distance map
    try:
        fig_distance = topic_model.visualize_topics()
        fig_distance.write_html(os.path.join(OUT_FIG_DIR, "topic_model.html"))
        print("  Saved: topic_model.html")
    except Exception as e:
        print(f"  WARNING: Failed to generate intertopic distance map: {e}")

    # 2. Topic hierarchy
    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(os.path.join(OUT_FIG_DIR, "topic_hierarchy.html"))
        print("  Saved: topic_hierarchy.html")
    except Exception as e:
        print(f"  WARNING: Failed to generate topic hierarchy: {e}")

    # 3. Topics over time (interactive)
    try:
        fig_time = topic_model.visualize_topics_over_time(
            topics_over_time, top_n_topics=args.top_n_topics
        )
        fig_time.write_html(os.path.join(OUT_FIG_DIR, "topic_over_time.html"))
        print("  Saved: topic_over_time.html")
    except Exception as e:
        print(f"  WARNING: Failed to generate topics over time: {e}")

    # 4. Topic heatmap (matplotlib static)
    try:
        _generate_heatmap(assignments, topic_names, args)
        print("  Saved: topic_heatmap.png")
    except Exception as e:
        print(f"  WARNING: Failed to generate heatmap: {e}")


def _generate_heatmap(assignments, topic_names, args):
    df = assignments.copy()
    df["year_month"] = pd.to_datetime(df["created_at"]).dt.to_period("M")

    top_n = args.top_n_topics
    top_topics = (
        df[df["topic_id"] >= 0]
        .groupby("topic_id")
        .size()
        .nlargest(top_n)
        .index.tolist()
    )

    heatmap_data = (
        df[df["topic_id"].isin(top_topics)]
        .groupby(["topic_id", "year_month"])
        .size()
        .unstack(fill_value=0)
    )

    heatmap_data.index = heatmap_data.index.map(
        lambda tid: topic_names.get(tid, f"Topic {tid}")
    )
    heatmap_data.columns = heatmap_data.columns.astype(str)

    fig, ax = plt.subplots(
        figsize=(max(12, len(heatmap_data.columns) * 0.8),
                 max(8, top_n * 0.4))
    )
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Topic")
    ax.set_title(f"Topic Prevalence Over Time (Top {top_n} Topics)")
    plt.colorbar(im, ax=ax, label="Conversation Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "topic_heatmap.png"), dpi=150)
    plt.close()


# ── Step 8: Generate Report ──────────────────────────────────────────────────
def generate_report(topic_model, topics, topic_names, args):
    print("Generating topic report...")
    from scipy.stats import entropy

    topic_counts = Counter(topics)
    total_convos = len(topics)
    n_topics = len([t for t in topic_counts if t >= 0])
    n_outliers = topic_counts.get(-1, 0)

    # Shannon entropy of topic distribution
    topic_dist = [
        count / total_convos
        for t, count in topic_counts.items()
        if t >= 0
    ]
    topic_diversity = float(entropy(topic_dist)) if topic_dist else 0.0

    top_topics_report = []
    for tid, count in topic_counts.most_common(30):
        if tid < 0:
            continue
        words = topic_model.get_topic(tid)
        top_topics_report.append({
            "topic_id": int(tid),
            "name": topic_names.get(tid, ""),
            "count": int(count),
            "percentage": round(count / total_convos * 100, 2),
            "top_terms": [w for w, _ in words[:10]]
        })

    report = {
        "total_conversations_modeled": int(total_convos),
        "num_topics_discovered": int(n_topics),
        "num_outlier_conversations": int(n_outliers),
        "outlier_percentage": round(n_outliers / total_convos * 100, 2),
        "topic_diversity_entropy": round(topic_diversity, 4),
        "top_topics": top_topics_report,
        "model_params": {
            "input_source": "all_summaries.csv (Claude-generated summaries)",
            "embedding_model": "all-MiniLM-L6-v2",
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "hdbscan_min_cluster_size": args.min_cluster_size,
            "hdbscan_min_samples": args.min_samples,
            "vectorizer_ngram_range": [1, 2],
            "vectorizer_min_df": 3,
            "custom_stopwords": len(CUSTOM_STOPS),
            "outlier_reduction": "embeddings" if args.reduce_outliers else "none",
            "random_seed": args.seed
        }
    }

    report_path = os.path.join(OUT_REPORT_DIR, "topic_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {report_path}")
    return report


# ── Step 9: Save All Outputs ────────────────────────────────────────────────
def save_outputs(topic_model, topic_info, assignments, topics_over_time):
    print("Saving outputs...")

    # Save BERTopic model
    topic_model.save(
        OUT_MODEL_DIR,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False
    )
    print(f"  Saved model: {OUT_MODEL_DIR}")

    # Save topic info CSV
    info_path = os.path.join(OUT_REPORT_DIR, "topic_info.csv")
    topic_info.to_csv(info_path, index=False)
    print(f"  Saved: {info_path}")

    # Save per-conversation topic assignments
    assignments.to_parquet(OUT_ASSIGNMENTS, index=False)
    print(f"  Saved: {OUT_ASSIGNMENTS}")

    # Save topics-over-time
    topics_over_time.to_parquet(OUT_TOT_PARQUET, index=False)
    print(f"  Saved: {OUT_TOT_PARQUET}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Module 3.2: Topic Modeling (Summary-based)")
    parser.add_argument("--summaries", default=None,
                        help="Path to all_summaries.csv")
    parser.add_argument("--embeddings", default=None,
                        help="Path to save/load embeddings (.npy)")
    parser.add_argument("--conversation-ids", default=None,
                        help="Path to conversation ID index (.json)")
    parser.add_argument("--min-cluster-size", type=int, default=15,
                        help="HDBSCAN min_cluster_size (lower = more topics)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples")
    parser.add_argument("--reduce-outliers", action="store_true", default=True,
                        help="Reassign outlier documents to nearest topic")
    parser.add_argument("--no-reduce-outliers", dest="reduce_outliers",
                        action="store_false")
    parser.add_argument("--top-n-topics", type=int, default=20,
                        help="Number of topics to show in visualizations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    ensure_dirs()

    # 1. Load summaries
    docs_df = load_summaries(args)

    # 2. Compute embeddings from summaries
    documents, embeddings = load_or_compute_embeddings(docs_df, args)

    # 3. Build topic model
    topic_model, topics, probabilities = build_topic_model(documents, embeddings, args)

    # Check for all-outliers edge case
    n_valid = len([t for t in topics if t >= 0])
    if n_valid == 0:
        print("WARNING: HDBSCAN produced only outliers. Try lowering --min-cluster-size.")
        sys.exit(1)

    # 4. Reduce outliers
    topics = reduce_outliers(topic_model, documents, topics, embeddings, args)

    # 5. Extract assignments
    assignments, topic_info, topic_names = extract_topic_assignments(
        topic_model, docs_df, topics, probabilities
    )

    # 6. Topics over time
    topics_over_time = compute_topics_over_time(topic_model, docs_df, documents)

    # 7. Visualizations
    generate_visualizations(
        topic_model, topics_over_time, assignments, topic_names, args
    )

    # 8. Report
    report = generate_report(topic_model, topics, topic_names, args)

    # 9. Save
    save_outputs(topic_model, topic_info, assignments, topics_over_time)

    # Summary
    print("\n" + "=" * 60)
    print("TOPIC MODELING COMPLETE")
    print("=" * 60)
    print(f"  Conversations modeled: {report['total_conversations_modeled']:,}")
    print(f"  Topics discovered:     {report['num_topics_discovered']}")
    print(f"  Outliers remaining:    {report['num_outlier_conversations']} ({report['outlier_percentage']}%)")
    print(f"  Topic diversity (H):   {report['topic_diversity_entropy']:.4f}")
    print(f"\n  Top 10 topics:")
    for t in report["top_topics"][:10]:
        print(f"    [{t['topic_id']:>3}] {t['name'][:60]:60s} ({t['count']} convos, {t['percentage']}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
