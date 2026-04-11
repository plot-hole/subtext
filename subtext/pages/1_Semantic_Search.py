"""Semantic Search -- query the corpus by meaning."""

import streamlit as st
import plotly.graph_objects as go
from app_utils import (
    semantic_search, load_map_data, PALETTE,
)

st.set_page_config(page_title="Semantic Search", layout="wide", page_icon="\u2609")
st.title("Semantic Search")

query = st.text_input("Search the corpus", placeholder="moments where I felt stuck about my direction")
k = st.slider("Results", 5, 50, 10)

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query, k=k)

    map_data = load_map_data()

    result_ids = [r["conversation_id"] for r in results]
    result_scores = {r["conversation_id"]: r["score"] for r in results}

    matched = map_data[map_data["conversation_id"].isin(result_ids)].copy()
    matched["score"] = matched["conversation_id"].map(result_scores)
    matched = matched.sort_values("score", ascending=False)

    # Mini UMAP showing where results land
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=map_data["umap_x"], y=map_data["umap_y"],
        mode="markers",
        marker=dict(size=4, color="#DDDDDD", opacity=0.3),
        hoverinfo="skip", name="Corpus",
    ))
    fig.add_trace(go.Scattergl(
        x=matched["umap_x"], y=matched["umap_y"],
        mode="markers",
        marker=dict(size=10, color=PALETTE[0], opacity=0.8),
        text=matched["title"],
        hovertemplate="%{text}<extra></extra>",
        name="Results",
    ))
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Results
    for _, row in matched.iterrows():
        score_pct = f"{row['score'] * 100:.1f}%"
        with st.expander(f"**{row['title']}** — {score_pct} match | {str(row['created_at'])[:10]}"):
            cols = st.columns([3, 1])
            with cols[0]:
                st.write(row.get("summary", ""))
            with cols[1]:
                st.caption(f"Era: {row.get('model_era', '')}")
                st.caption(f"Type: {row.get('conversation_type', '')}")
                st.caption(f"Emotion: {row.get('emotion_primary', '')}")
                st.caption(f"Function: {row.get('function_primary', '')}")
                if row.get("top_entity"):
                    st.caption(f"Top entity: {row['top_entity']}")
            if st.button("Read conversation", key=row["conversation_id"]):
                st.session_state["selected_conversation_id"] = row["conversation_id"]
                st.switch_page("pages/5_Conversation_Reader.py")
