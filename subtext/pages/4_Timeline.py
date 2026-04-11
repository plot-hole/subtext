"""Timeline -- volume, semantic drift, and composition over time."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app_utils import load_map_data, load_entity_attention, PALETTE

st.set_page_config(page_title="Timeline", layout="wide", page_icon="\u2609")
st.title("Timeline")

map_data = load_map_data()
entities_df = load_entity_attention()

# --- Sidebar filters ---
with st.sidebar:
    st.subheader("Filters")

    era_options = sorted(map_data["model_era"].dropna().unique().tolist())
    selected_eras = st.multiselect("Model Era", era_options, default=era_options)

    type_options = sorted(map_data["conversation_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Conversation Type", type_options, default=type_options)

    top_entities = (
        entities_df.groupby("entity")["mention_count"].sum()
        .sort_values(ascending=False).head(20).index.tolist()
    )
    selected_entities = st.multiselect("Entity filter (optional)", top_entities)

# --- Apply filters ---
df = map_data.copy()
df = df[df["model_era"].isin(selected_eras)]
df = df[df["conversation_type"].isin(selected_types)]

if selected_entities:
    entity_conv_ids = set(
        entities_df[entities_df["entity"].isin(selected_entities)]["conversation_id"]
    )
    df = df[df["conversation_id"].isin(entity_conv_ids)]

st.caption(f"Showing {len(df)} conversations")

# --- Chart A: Volume ---
st.subheader("Conversation Volume")
stack_by = st.radio("Stack by", ["Conversation Type", "Model Era"], horizontal=True)
stack_col = "conversation_type" if stack_by == "Conversation Type" else "model_era"

vol = df.groupby(["year_week", stack_col]).size().reset_index(name="count")
fig_vol = px.bar(
    vol, x="year_week", y="count", color=stack_col,
    color_discrete_sequence=PALETTE,
)
fig_vol.update_layout(
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(tickangle=-45, dtick=4, title=""),
    yaxis=dict(title="Conversations"),
    legend=dict(title="", orientation="h", y=-0.3),
)
st.plotly_chart(fig_vol, use_container_width=True)

# --- Chart B: Semantic Drift ---
st.subheader("Semantic Drift")
if "centroid_distance" in df.columns:
    drift = df.groupby("year_week")["centroid_distance"].agg(["mean", "count"]).reset_index()
    drift.columns = ["year_week", "mean_distance", "n"]

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=drift["year_week"], y=drift["mean_distance"],
        mode="markers", marker=dict(size=6, color=PALETTE[0], opacity=0.4),
        name="Weekly mean",
    ))
    if len(drift) >= 8:
        smoothed = drift["mean_distance"].rolling(8, center=True, min_periods=3).mean()
        fig_drift.add_trace(go.Scatter(
            x=drift["year_week"], y=smoothed,
            mode="lines", line=dict(color=PALETTE[1], width=2),
            name="8-week rolling avg",
        ))
    fig_drift.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-45, dtick=4, title=""),
        yaxis=dict(title="Mean Centroid Distance"),
    )
    st.plotly_chart(fig_drift, use_container_width=True)

# --- Chart C: Emotional Composition ---
st.subheader("Emotional Composition Over Time")
if "emotion_primary" in df.columns:
    emo = df.groupby(["year_week", "emotion_primary"]).size().reset_index(name="count")
    week_totals = emo.groupby("year_week")["count"].transform("sum")
    emo["proportion"] = emo["count"] / week_totals

    fig_emo = px.area(
        emo, x="year_week", y="proportion", color="emotion_primary",
        color_discrete_sequence=PALETTE,
    )
    fig_emo.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-45, dtick=4, title=""),
        yaxis=dict(title="Proportion"),
        legend=dict(title="", orientation="h", y=-0.3),
    )
    st.plotly_chart(fig_emo, use_container_width=True)

# --- Chart D: Functional Composition ---
st.subheader("Functional Composition Over Time")
if "function_primary" in df.columns:
    func = df.groupby(["year_week", "function_primary"]).size().reset_index(name="count")
    week_totals = func.groupby("year_week")["count"].transform("sum")
    func["proportion"] = func["count"] / week_totals

    fig_func = px.area(
        func, x="year_week", y="proportion", color="function_primary",
        color_discrete_sequence=PALETTE,
    )
    fig_func.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-45, dtick=4, title=""),
        yaxis=dict(title="Proportion"),
        legend=dict(title="", orientation="h", y=-0.3),
    )
    st.plotly_chart(fig_func, use_container_width=True)
