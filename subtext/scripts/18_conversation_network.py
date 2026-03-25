"""
Module 4.0: Conversation Network Graph
Script: 18_conversation_network.py

Builds an interactive network graph where each node is a conversation and edges
connect conversations that share functional purpose, emotional texture, topic
cluster, or temporal proximity. The result reveals hidden bridges between
otherwise separate thematic clusters — e.g., a "career_strategy" conversation
emotionally linked to "emotional_processing" through shared grief.

Node attributes:
    - Color  → functional category (12 categories)
    - Size   → conversation length (msg_count, log-scaled)
    - Border → emotional state (12 states)

Edge types (weighted):
    - Same functional category      (weight 0.4)
    - Same emotional state           (weight 0.3)
    - Same topic cluster             (weight 0.5)
    - Temporal proximity < 2 hours   (weight 0.2)

Outputs:
    - Interactive Plotly HTML network graph
    - Community detection (Louvain) with cross-category bridges highlighted
    - Network metrics JSON report (degree distribution, centrality, modularity)
    - Static matplotlib summary figure

Usage:
    python scripts/18_conversation_network.py

    # Limit edges for large corpora:
    python scripts/18_conversation_network.py --max-edges 20000

    # Adjust temporal window (hours):
    python scripts/18_conversation_network.py --temporal-window 4

    # Skip community detection:
    python scripts/18_conversation_network.py --skip-communities
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
import networkx as nx
import plotly.graph_objects as go
from collections import Counter, defaultdict
from datetime import datetime, timezone

# -- Paths -------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONV_PATH       = os.path.join(BASE, "data", "processed", "conversations_clean.parquet")
FUNC_CLASS_PATH = os.path.join(BASE, "data", "processed", "functional_classifications.parquet")
EMOT_PATH       = os.path.join(BASE, "data", "processed", "emotional_states.parquet")
TOPIC_PATH      = os.path.join(BASE, "data", "processed", "topic_assignments.parquet")
SHAPE_PATH      = os.path.join(BASE, "data", "processed", "shape_clusters.parquet")
CONFIG_PATH     = os.path.join(BASE, "config", "quality_config.json")

OUT_REPORT  = os.path.join(BASE, "outputs", "reports", "conversation_network_report.json")
FIG_DIR     = os.path.join(BASE, "outputs", "figures", "conversation_network")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "outputs", "reports"), exist_ok=True)

# -- Color palettes ----------------------------------------------------------
FUNC_COLORS = {
    "interpersonal_analysis": "#e6194b",
    "emotional_processing":   "#f58231",
    "creative_expression":    "#ffe119",
    "career_strategy":        "#3cb44b",
    "self_modeling":           "#42d4f4",
    "practical":               "#4363d8",
    "learning":                "#911eb4",
    "problem_solving":         "#f032e6",
    "coding":                  "#a9a9a9",
    "social_rehearsal":        "#fabebe",
    "planning":                "#469990",
    "work_professional":       "#dcbeff",
}

EMOT_COLORS = {
    "analytical":  "#4363d8",
    "anxious":     "#e6194b",
    "curious":     "#3cb44b",
    "frustrated":  "#f58231",
    "grieving":    "#800000",
    "playful":     "#ffe119",
    "reflective":  "#42d4f4",
    "strategic":   "#469990",
    "vulnerable":  "#f032e6",
    "energized":   "#bfef45",
    "numb":        "#a9a9a9",
    "determined":  "#911eb4",
}

DEFAULT_COLOR = "#cccccc"


# ── Load Data ───────────────────────────────────────────────────────────────
def load_data():
    """Load all available upstream data. Gracefully skip missing files."""
    print("Loading data...")

    if not os.path.exists(CONV_PATH):
        print(f"  ERROR: Conversations file not found: {CONV_PATH}")
        sys.exit(1)

    conv = pd.read_parquet(CONV_PATH)
    print(f"  Conversations: {len(conv):,}")

    func_class = None
    if os.path.exists(FUNC_CLASS_PATH):
        func_class = pd.read_parquet(FUNC_CLASS_PATH)
        print(f"  Functional classifications: {len(func_class):,}")
    else:
        print("  Functional classifications: not found (skipping)")

    emot = None
    if os.path.exists(EMOT_PATH):
        emot = pd.read_parquet(EMOT_PATH)
        print(f"  Emotional states: {len(emot):,}")
    else:
        print("  Emotional states: not found (skipping)")

    topics = None
    if os.path.exists(TOPIC_PATH):
        topics = pd.read_parquet(TOPIC_PATH)
        print(f"  Topic assignments: {len(topics):,}")
    else:
        print("  Topic assignments: not found (skipping)")

    shapes = None
    if os.path.exists(SHAPE_PATH):
        shapes = pd.read_parquet(SHAPE_PATH)
        print(f"  Shape clusters: {len(shapes):,}")
    else:
        print("  Shape clusters: not found (skipping)")

    return conv, func_class, emot, topics, shapes


# ── Merge Attributes ────────────────────────────────────────────────────────
def merge_attributes(conv, func_class, emot, topics, shapes):
    """Merge all classification data onto conversations."""
    df = conv[["conversation_id"]].copy()

    # Conversation metadata
    for col in ["msg_count", "user_msg_count", "created_at", "title",
                 "user_token_total", "assistant_token_total"]:
        if col in conv.columns:
            df[col] = conv[col].values

    # Functional classification
    if func_class is not None and "function" in func_class.columns:
        fmap = func_class.set_index("conversation_id")["function"]
        df["function"] = df["conversation_id"].map(fmap).fillna("unknown")
    else:
        df["function"] = "unknown"

    # Emotional state
    if emot is not None and "emotion" in emot.columns:
        emap = emot.set_index("conversation_id")["emotion"]
        df["emotion"] = df["conversation_id"].map(emap).fillna("unknown")
    else:
        df["emotion"] = "unknown"

    # Topic cluster
    if topics is not None and "topic" in topics.columns:
        tmap = topics.set_index("conversation_id")["topic"]
        df["topic"] = df["conversation_id"].map(tmap).fillna(-1).astype(int)
    else:
        df["topic"] = -1

    # Shape cluster
    if shapes is not None:
        shape_col = "cluster_label" if "cluster_label" in shapes.columns else "shape_archetype"
        if shape_col in shapes.columns:
            smap = shapes.set_index("conversation_id")[shape_col]
            df["shape"] = df["conversation_id"].map(smap).fillna("unknown")
        else:
            df["shape"] = "unknown"
    else:
        df["shape"] = "unknown"

    print(f"\n  Merged dataset: {len(df):,} conversations")
    print(f"  Functions: {df['function'].nunique()} unique")
    print(f"  Emotions:  {df['emotion'].nunique()} unique")
    print(f"  Topics:    {df['topic'].nunique()} unique")
    return df


# ── Build Network ───────────────────────────────────────────────────────────
def build_graph(df, temporal_window_hours=2, max_edges=15000):
    """
    Build a weighted undirected graph.

    Nodes = conversations.
    Edges connect conversations sharing attributes, with weight proportional
    to the number of shared dimensions.
    """
    print(f"\nBuilding network graph...")

    G = nx.Graph()

    # Add nodes with attributes
    for _, row in df.iterrows():
        cid = row["conversation_id"]
        G.add_node(cid,
                   function=row["function"],
                   emotion=row["emotion"],
                   topic=int(row["topic"]),
                   shape=row.get("shape", "unknown"),
                   msg_count=int(row.get("msg_count", 1)),
                   title=str(row.get("title", "")),
                   created_at=str(row.get("created_at", "")))

    # Build edge candidates by shared attributes (index-based for speed)
    print("  Indexing shared attributes...")
    edge_weights = defaultdict(float)

    # Group by function
    if df["function"].nunique() > 1:
        for func, group in df.groupby("function"):
            if func == "unknown" or len(group) < 2:
                continue
            ids = group["conversation_id"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_weights[(ids[i], ids[j])] += 0.4
        print(f"    Function edges indexed")

    # Group by emotion
    if df["emotion"].nunique() > 1:
        for emo, group in df.groupby("emotion"):
            if emo == "unknown" or len(group) < 2:
                continue
            ids = group["conversation_id"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_weights[(ids[i], ids[j])] += 0.3
        print(f"    Emotion edges indexed")

    # Group by topic (skip outlier topic -1)
    if df["topic"].nunique() > 1:
        for topic, group in df.groupby("topic"):
            if topic == -1 or len(group) < 2:
                continue
            ids = group["conversation_id"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_weights[(ids[i], ids[j])] += 0.5
        print(f"    Topic edges indexed")

    # Temporal proximity
    if "created_at" in df.columns:
        df_sorted = df.sort_values("created_at").reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(df_sorted["created_at"]):
            window_ns = temporal_window_hours * 3600 * 1e9
            times = df_sorted["created_at"].values.astype(np.int64)
            cids = df_sorted["conversation_id"].values
            temporal_count = 0
            for i in range(len(df_sorted)):
                j = i + 1
                while j < len(df_sorted) and (times[j] - times[i]) < window_ns:
                    edge_weights[(cids[i], cids[j])] += 0.2
                    temporal_count += 1
                    j += 1
            print(f"    Temporal edges: {temporal_count:,} (within {temporal_window_hours}h)")

    # Filter to top edges by weight if too many
    print(f"  Total edge candidates: {len(edge_weights):,}")
    if len(edge_weights) > max_edges:
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        edge_weights = dict(sorted_edges[:max_edges])
        print(f"  Trimmed to top {max_edges:,} edges by weight")

    # Add edges
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    # Remove isolates
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"  Removed {len(isolates)} isolated nodes")

    print(f"  Final graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── Community Detection ─────────────────────────────────────────────────────
def detect_communities(G):
    """Louvain community detection."""
    print("\nDetecting communities (Louvain)...")
    communities = nx.community.louvain_communities(G, seed=42, weight="weight")
    print(f"  Found {len(communities)} communities")

    # Assign community labels to nodes
    for i, community in enumerate(communities):
        for node in community:
            G.nodes[node]["community"] = i

    # Identify cross-category bridges (nodes whose neighbors span multiple functions)
    bridges = []
    for node in G.nodes():
        neighbor_funcs = set()
        for neighbor in G.neighbors(node):
            neighbor_funcs.add(G.nodes[neighbor].get("function", "unknown"))
        if len(neighbor_funcs) >= 3:
            bridges.append({
                "conversation_id": node,
                "function": G.nodes[node].get("function", "unknown"),
                "emotion": G.nodes[node].get("emotion", "unknown"),
                "title": G.nodes[node].get("title", ""),
                "neighbor_functions": sorted(neighbor_funcs),
                "degree": G.degree(node),
            })

    bridges.sort(key=lambda x: x["degree"], reverse=True)
    print(f"  Cross-category bridges (>=3 functional neighbors): {len(bridges)}")

    return communities, bridges[:50]


# ── Layout ──────────────────────────────────────────────────────────────────
def compute_layout(G):
    """Spring layout with weight-aware positioning."""
    print("\nComputing layout (spring)...")
    pos = nx.spring_layout(G, k=1.5 / np.sqrt(max(G.number_of_nodes(), 1)),
                           iterations=80, seed=42, weight="weight")
    return pos


# ── Interactive Plotly Graph ────────────────────────────────────────────────
def make_interactive_graph(G, pos, communities):
    """Build an interactive Plotly network visualization."""
    print("\nGenerating interactive graph...")

    # Edge traces (light gray, opacity by weight)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color="rgba(150,150,150,0.15)"),
        hoverinfo="none",
        mode="lines",
    )

    # Node traces — one per functional category for legend
    node_traces = []
    func_groups = defaultdict(list)
    for node in G.nodes():
        func = G.nodes[node].get("function", "unknown")
        func_groups[func].append(node)

    for func, nodes in sorted(func_groups.items()):
        x_vals = [pos[n][0] for n in nodes]
        y_vals = [pos[n][1] for n in nodes]
        sizes = [max(5, min(30, np.log2(G.nodes[n].get("msg_count", 1) + 1) * 4)) for n in nodes]
        color = FUNC_COLORS.get(func, DEFAULT_COLOR)

        hover_texts = []
        for n in nodes:
            nd = G.nodes[n]
            hover_texts.append(
                f"<b>{nd.get('title', 'Untitled')[:60]}</b><br>"
                f"Function: {nd.get('function', '?')}<br>"
                f"Emotion: {nd.get('emotion', '?')}<br>"
                f"Topic: {nd.get('topic', '?')}<br>"
                f"Messages: {nd.get('msg_count', '?')}<br>"
                f"Community: {nd.get('community', '?')}<br>"
                f"Connections: {G.degree(n)}"
            )

        node_traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            name=func.replace("_", " ").title(),
            marker=dict(
                size=sizes,
                color=color,
                line=dict(
                    width=1.5,
                    color=[EMOT_COLORS.get(G.nodes[n].get("emotion", ""), DEFAULT_COLOR) for n in nodes],
                ),
                opacity=0.85,
            ),
            text=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text="Conversation Network — Nodes by Function, Borders by Emotion",
                font=dict(size=16),
            ),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1400,
            height=900,
            legend=dict(
                title="Functional Category",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            ),
            margin=dict(l=20, r=20, t=60, b=20),
        ),
    )

    out_path = os.path.join(FIG_DIR, "conversation_network.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved: {out_path}")
    return fig


# ── Static Summary Figure ──────────────────────────────────────────────────
def make_summary_figure(G, communities, bridges, df):
    """4-panel static summary: degree dist, community sizes, bridge highlights, function × emotion heatmap."""
    print("\nGenerating summary figure...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Conversation Network — Summary", fontsize=16, fontweight="bold", y=0.98)

    # 1. Degree distribution
    ax = axes[0, 0]
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=min(50, max(10, len(set(degrees)))),
            color="#4363d8", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Degree (connections)")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution")
    ax.axvline(np.median(degrees), color="red", linestyle="--", alpha=0.7,
               label=f"Median: {np.median(degrees):.0f}")
    ax.legend()

    # 2. Community sizes
    ax = axes[0, 1]
    comm_sizes = sorted([len(c) for c in communities], reverse=True)
    colors_list = plt.cm.Set3(np.linspace(0, 1, len(comm_sizes)))
    ax.bar(range(len(comm_sizes)), comm_sizes, color=colors_list, edgecolor="white")
    ax.set_xlabel("Community")
    ax.set_ylabel("Size (conversations)")
    ax.set_title(f"Community Sizes ({len(communities)} communities)")

    # 3. Top bridges
    ax = axes[1, 0]
    if bridges:
        top_bridges = bridges[:15]
        titles = [b["title"][:35] + "..." if len(b["title"]) > 35 else b["title"] for b in top_bridges]
        degrees_b = [b["degree"] for b in top_bridges]
        bar_colors = [FUNC_COLORS.get(b["function"], DEFAULT_COLOR) for b in top_bridges]
        y_pos = range(len(titles))
        ax.barh(y_pos, degrees_b, color=bar_colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=7)
        ax.set_xlabel("Degree")
        ax.set_title("Top Cross-Category Bridges")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No bridges detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Cross-Category Bridges")

    # 4. Function × Emotion heatmap
    ax = axes[1, 1]
    cross = pd.crosstab(df["function"], df["emotion"])
    if cross.shape[0] > 1 and cross.shape[1] > 1:
        im = ax.imshow(cross.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(cross.columns)))
        ax.set_xticklabels(cross.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cross.index)))
        ax.set_yticklabels(cross.index, fontsize=7)
        ax.set_title("Function × Emotion Co-occurrence")
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "Insufficient data for heatmap", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Function × Emotion Co-occurrence")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "network_summary.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Network Metrics ─────────────────────────────────────────────────────────
def compute_metrics(G, communities, bridges):
    """Compute graph-level and node-level metrics."""
    print("\nComputing network metrics...")

    degrees = dict(G.degree())
    degree_vals = list(degrees.values())

    # Betweenness centrality (sample for large graphs)
    if G.number_of_nodes() > 500:
        betweenness = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()), seed=42)
    else:
        betweenness = nx.betweenness_centrality(G)

    # Top central nodes
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]

    # Modularity
    comm_sets = [set(c) for c in communities]
    modularity = nx.community.modularity(G, comm_sets, weight="weight")

    # Community composition
    community_profiles = []
    for i, comm in enumerate(communities):
        funcs = Counter(G.nodes[n].get("function", "unknown") for n in comm)
        emots = Counter(G.nodes[n].get("emotion", "unknown") for n in comm)
        community_profiles.append({
            "community": i,
            "size": len(comm),
            "top_functions": funcs.most_common(3),
            "top_emotions": emots.most_common(3),
            "dominant_function": funcs.most_common(1)[0][0] if funcs else "unknown",
        })

    metrics = {
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 6),
            "avg_degree": round(np.mean(degree_vals), 2),
            "median_degree": round(np.median(degree_vals), 2),
            "max_degree": max(degree_vals),
            "modularity": round(modularity, 4),
            "n_communities": len(communities),
            "n_components": nx.number_connected_components(G),
        },
        "top_degree": [
            {"id": n, "degree": d, "title": G.nodes[n].get("title", ""),
             "function": G.nodes[n].get("function", "")}
            for n, d in top_degree
        ],
        "top_betweenness": [
            {"id": n, "centrality": round(b, 6), "title": G.nodes[n].get("title", ""),
             "function": G.nodes[n].get("function", "")}
            for n, b in top_between
        ],
        "communities": community_profiles,
        "bridges": bridges,
    }

    print(f"  Density: {metrics['graph']['density']}")
    print(f"  Modularity: {metrics['graph']['modularity']}")
    print(f"  Avg degree: {metrics['graph']['avg_degree']}")
    print(f"  Components: {metrics['graph']['n_components']}")

    return metrics


# ── Report ──────────────────────────────────────────────────────────────────
def generate_report(metrics, df):
    """Save JSON report."""
    report = {
        "module": "18_conversation_network",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Network graph connecting conversations by shared attributes",
        "input_conversations": len(df),
        "metrics": metrics,
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report saved: {OUT_REPORT}")
    return report


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build conversation network graph")
    parser.add_argument("--max-edges", type=int, default=15000,
                        help="Maximum edges to keep (default: 15000)")
    parser.add_argument("--temporal-window", type=float, default=2.0,
                        help="Temporal proximity window in hours (default: 2)")
    parser.add_argument("--skip-communities", action="store_true",
                        help="Skip Louvain community detection")
    args = parser.parse_args()

    print("=" * 70)
    print("MODULE 18: CONVERSATION NETWORK GRAPH")
    print("=" * 70)

    conv, func_class, emot, topics, shapes = load_data()
    df = merge_attributes(conv, func_class, emot, topics, shapes)

    G = build_graph(df, temporal_window_hours=args.temporal_window,
                    max_edges=args.max_edges)

    if G.number_of_nodes() < 2:
        print("\nERROR: Graph has fewer than 2 nodes. Check upstream data.")
        sys.exit(1)

    # Community detection
    communities = [{n} for n in G.nodes()]  # fallback: each node is own community
    bridges = []
    if not args.skip_communities:
        communities, bridges = detect_communities(G)

    pos = compute_layout(G)

    make_interactive_graph(G, pos, communities)
    make_summary_figure(G, communities, bridges, df)

    metrics = compute_metrics(G, communities, bridges)
    report = generate_report(metrics, df)

    # Print highlights
    print("\n" + "=" * 70)
    print("HIGHLIGHTS")
    print("=" * 70)
    print(f"  Nodes: {metrics['graph']['nodes']:,}")
    print(f"  Edges: {metrics['graph']['edges']:,}")
    print(f"  Communities: {metrics['graph']['n_communities']}")
    print(f"  Modularity: {metrics['graph']['modularity']}")

    if bridges:
        print(f"\n  Top bridge conversation:")
        b = bridges[0]
        print(f"    \"{b['title'][:60]}\"")
        print(f"    Function: {b['function']}, Emotion: {b['emotion']}")
        print(f"    Connects {len(b['neighbor_functions'])} functional categories")

    print("\nDone!")


if __name__ == "__main__":
    main()
