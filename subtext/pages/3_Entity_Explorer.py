"""Entity Explorer -- semantic territory, co-occurrences, attention over time."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app_utils import (
    load_map_data, load_entity_attention, load_entity_weekly,
    load_emotional_states, PALETTE,
)

st.set_page_config(page_title="Entity Explorer", layout="wide", page_icon="\u2609")
st.title("Entity Explorer")

entities_df = load_entity_attention()
weekly_df = load_entity_weekly()
map_data = load_map_data()
emotions = load_emotional_states()

# Top entities by total mentions
entity_totals = entities_df.groupby("entity")["mention_count"].sum().sort_values(ascending=False)
top_entities = entity_totals.head(50).index.tolist()

selected = st.selectbox("Select entity", top_entities)

if selected:
    entity_convos_df = entities_df[entities_df["entity"] == selected]
    entity_conv_ids = set(entity_convos_df["conversation_id"])
    entity_map = map_data[map_data["conversation_id"].isin(entity_conv_ids)]

    # --- Overview metrics ---
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mentions", f"{int(entity_totals[selected]):,}")
    c2.metric("Conversations", f"{len(entity_conv_ids)}")
    c3.metric("Mean Mentions/Convo", f"{entity_convos_df['mention_count'].mean():.1f}")

    if len(entity_map) > 0 and "emotion_primary" in entity_map.columns:
        top_emotion = entity_map["emotion_primary"].mode()
        c4.metric("Top Emotion", top_emotion.iloc[0] if len(top_emotion) > 0 else "—")
    else:
        c4.metric("Top Emotion", "—")

    st.divider()

    # --- Semantic territory ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Semantic Territory")
        fig = go.Figure()
        other = map_data[~map_data["conversation_id"].isin(entity_conv_ids)]
        fig.add_trace(go.Scattergl(
            x=other["umap_x"], y=other["umap_y"],
            mode="markers",
            marker=dict(size=4, color="#DDDDDD", opacity=0.3),
            hoverinfo="skip", name="Other",
        ))
        fig.add_trace(go.Scattergl(
            x=entity_map["umap_x"], y=entity_map["umap_y"],
            mode="markers",
            marker=dict(size=8, color=PALETTE[0], opacity=0.7),
            text=entity_map["title"],
            hovertemplate="%{text}<extra></extra>",
            name=selected,
        ))
        fig.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Co-occurring Entities")
        other_entities = (
            entities_df[entities_df["conversation_id"].isin(entity_conv_ids)]
            .groupby("entity")["mention_count"].sum()
            .sort_values(ascending=False)
        )
        # Remove self
        other_entities = other_entities.drop(selected, errors="ignore").head(15)

        if len(other_entities) > 0:
            fig_co = px.bar(
                x=other_entities.values,
                y=other_entities.index,
                orientation="h",
                labels={"x": "Mentions", "y": "Entity"},
            )
            fig_co.update_layout(
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange="reversed"),
            )
            fig_co.update_traces(marker_color=PALETTE[1])
            st.plotly_chart(fig_co, use_container_width=True)

    st.divider()

    # --- Attention over time ---
    st.subheader("Attention Over Time")
    entity_weekly = weekly_df[weekly_df["entity"] == selected].copy()
    if len(entity_weekly) > 0:
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=entity_weekly["year_week"],
            y=entity_weekly["mention_count"],
            name="Mentions",
            marker_color=PALETTE[0],
            opacity=0.6,
        ))
        if "conversation_count" in entity_weekly.columns:
            fig_time.add_trace(go.Scatter(
                x=entity_weekly["year_week"],
                y=entity_weekly["conversation_count"],
                name="Conversations",
                yaxis="y2",
                line=dict(color=PALETTE[1], width=2),
            ))
            fig_time.update_layout(
                yaxis2=dict(title="Conversations", overlaying="y", side="right"),
            )
        fig_time.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickangle=-45, dtick=4),
            yaxis=dict(title="Mentions"),
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No weekly data for this entity.")

    st.divider()

    # --- Conversations table ---
    st.subheader("Conversations")
    display = entity_map[["title", "created_at", "model_era", "emotion_primary", "function_primary", "summary"]].copy()
    display["created_at"] = display["created_at"].dt.strftime("%Y-%m-%d")
    display = display.rename(columns={
        "created_at": "Date", "model_era": "Era",
        "emotion_primary": "Emotion", "function_primary": "Function",
        "title": "Title", "summary": "Summary",
    })
    st.dataframe(display, use_container_width=True, height=400)
