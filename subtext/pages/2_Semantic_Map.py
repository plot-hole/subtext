"""Semantic Map -- interactive UMAP scatter plot."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app_utils import load_map_data, load_entity_attention, PALETTE

st.set_page_config(page_title="Semantic Map", layout="wide", page_icon="\u2609")
st.title("Semantic Map")

map_data = load_map_data()
entities_df = load_entity_attention()

# --- Controls ---
col1, col2, col3 = st.columns(3)

with col1:
    color_by = st.selectbox("Color by", [
        "Semantic Cluster", "Model Era", "Conversation Type",
        "Emotion", "Function", "Entity Mention",
    ])

with col2:
    if color_by == "Entity Mention":
        top_entities = (
            entities_df.groupby("entity")["mention_count"].sum()
            .sort_values(ascending=False).head(30).index.tolist()
        )
        selected_entity = st.selectbox("Entity", top_entities)
    else:
        selected_entity = None

with col3:
    show_noise = st.checkbox("Show noise points (cluster -1)", value=True)

# --- Filter ---
df = map_data.copy()
if not show_noise:
    df = df[df["semantic_cluster"] != -1]

# --- Build figure ---
if color_by == "Entity Mention" and selected_entity:
    entity_convos = set(
        entities_df[entities_df["entity"] == selected_entity]["conversation_id"]
    )
    df["has_entity"] = df["conversation_id"].isin(entity_convos)

    fig = go.Figure()
    no_entity = df[~df["has_entity"]]
    yes_entity = df[df["has_entity"]]

    fig.add_trace(go.Scattergl(
        x=no_entity["umap_x"], y=no_entity["umap_y"],
        mode="markers",
        marker=dict(size=5, color="#DDDDDD", opacity=0.3),
        text=no_entity["title"],
        hovertemplate="%{text}<extra></extra>",
        name="Other",
    ))
    fig.add_trace(go.Scattergl(
        x=yes_entity["umap_x"], y=yes_entity["umap_y"],
        mode="markers",
        marker=dict(size=8, color=PALETTE[0], opacity=0.7),
        text=yes_entity["title"],
        customdata=yes_entity["conversation_id"],
        hovertemplate="%{text}<extra></extra>",
        name=selected_entity,
    ))
    fig.update_layout(title=f"Conversations mentioning {selected_entity} ({len(yes_entity)})")

else:
    col_map = {
        "Semantic Cluster": "semantic_cluster",
        "Model Era": "model_era",
        "Conversation Type": "conversation_type",
        "Emotion": "emotion_primary",
        "Function": "function_primary",
    }
    color_col = col_map[color_by]
    df[color_col] = df[color_col].fillna("unknown").astype(str)

    fig = px.scatter(
        df, x="umap_x", y="umap_y",
        color=color_col,
        hover_data=["title", "model_era", "conversation_type"],
        custom_data=["conversation_id"],
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6))

fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    legend=dict(font=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

# --- Click to inspect ---
st.subheader("Inspect a Conversation")
search_title = st.text_input("Search by title")
if search_title:
    matches = df[df["title"].str.contains(search_title, case=False, na=False)]
    if len(matches) > 0:
        selected = st.selectbox(
            "Select", matches["conversation_id"].tolist(),
            format_func=lambda x: matches[matches["conversation_id"] == x]["title"].iloc[0],
        )
        row = matches[matches["conversation_id"] == selected].iloc[0]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{row['title']}**")
            st.write(row.get("summary", ""))
        with col2:
            st.caption(f"Date: {str(row['created_at'])[:10]}")
            st.caption(f"Era: {row.get('model_era', '')}")
            st.caption(f"Cluster: {row.get('semantic_cluster', '')}")
            st.caption(f"Centroid dist: {row.get('centroid_distance', 0):.3f}")
            st.caption(f"Emotion: {row.get('emotion_primary', '')}")
            st.caption(f"Function: {row.get('function_primary', '')}")
        if st.button("Read full conversation"):
            st.session_state["selected_conversation_id"] = selected
            st.switch_page("pages/5_Conversation_Reader.py")
    else:
        st.info("No matches found.")
