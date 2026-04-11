"""
Shared data loaders for SUBTEXT Corpus Explorer.
All heavy loading is cached via Streamlit decorators.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "outputs" / "models"

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_conversations():
    return pd.read_parquet(PROCESSED / "conversations_clean.parquet")


@st.cache_data
def load_messages():
    return pd.read_parquet(PROCESSED / "messages_clean.parquet")


@st.cache_data
def load_summaries():
    return pd.read_parquet(PROCESSED / "conversation_summaries.parquet")


@st.cache_data
def load_embeddings_meta():
    df = pd.read_parquet(PROCESSED / "conversation_embeddings.parquet")
    return df.drop(columns=["summary_embedding", "title_embedding"], errors="ignore")


@st.cache_data
def load_entity_attention():
    return pd.read_parquet(PROCESSED / "entity_attention.parquet")


@st.cache_data
def load_entity_weekly():
    return pd.read_parquet(PROCESSED / "entity_attention_weekly.parquet")


@st.cache_data
def load_emotional_states():
    return pd.read_parquet(PROCESSED / "emotional_states.parquet")


@st.cache_data
def load_functional():
    return pd.read_parquet(PROCESSED / "functional_classifications.parquet")


@st.cache_data
def load_prediction_outcomes():
    return pd.read_parquet(PROCESSED / "prediction_outcomes.parquet")


@st.cache_data
def load_hypotheses():
    return pd.read_parquet(PROCESSED / "hypotheses.parquet")


# ---------------------------------------------------------------------------
# Heavy resources (non-serializable, cached in memory)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_faiss_index():
    import faiss
    return faiss.read_index(str(MODELS / "embeddings" / "conversation_embeddings.index"))


@st.cache_resource
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


@st.cache_data
def load_embedding_vectors():
    """Load raw embedding vectors as a (N, 768) float32 array."""
    df = pd.read_parquet(PROCESSED / "conversation_embeddings.parquet")
    vectors = np.vstack(df["summary_embedding"].values).astype(np.float32)
    conv_ids = df["conversation_id"].tolist()
    return vectors, conv_ids


# ---------------------------------------------------------------------------
# Precomputed joins
# ---------------------------------------------------------------------------

@st.cache_data
def load_map_data():
    """Merge core tables into a single plot-ready DataFrame."""
    convos = load_conversations()
    summaries = load_summaries()
    emb_meta = load_embeddings_meta()
    emotions = load_emotional_states()
    funcs = load_functional()
    entities = load_entity_attention()

    # Get top entity per conversation
    top_entity = (
        entities.groupby("conversation_id")
        .apply(lambda g: g.loc[g["mention_count"].idxmax(), "entity"], include_groups=False)
        .reset_index()
    )
    top_entity.columns = ["conversation_id", "top_entity"]

    df = convos[convos["is_analysable"]].merge(
        summaries[["conversation_id", "summary"]], on="conversation_id", how="left"
    ).merge(
        emb_meta, on="conversation_id", how="left"
    ).merge(
        emotions[["conversation_id", "emotion_primary", "emotion_secondary"]],
        on="conversation_id", how="left"
    ).merge(
        funcs[["conversation_id", "function_primary", "function_secondary"]],
        on="conversation_id", how="left"
    ).merge(
        top_entity, on="conversation_id", how="left"
    )

    return df


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------

def semantic_search(query: str, k: int = 10):
    """Encode a query and search the FAISS index. Returns (scores, conversation_ids)."""
    model = load_sentence_model()
    index = load_faiss_index()
    _, conv_ids = load_embedding_vectors()

    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_emb, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < len(conv_ids):
            results.append({"conversation_id": conv_ids[idx], "score": float(scores[0][i])})

    return results


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PALETTE = [
    "#2E75B6", "#C55A11", "#E91E90", "#E15759",
    "#59A14F", "#B07AA1", "#F28E2B", "#76B7B2",
    "#FF9DA7", "#9C755F", "#BAB0AC", "#4E79A7",
]
