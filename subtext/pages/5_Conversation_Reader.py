"""Conversation Reader -- read full message threads with metadata."""

import streamlit as st
import pandas as pd
from app_utils import (
    load_conversations, load_messages, load_summaries,
    load_embeddings_meta, load_emotional_states, load_functional,
    load_entity_attention, load_hypotheses, load_prediction_outcomes,
)

st.set_page_config(page_title="Conversation Reader", layout="wide", page_icon="\u2609")
st.title("Conversation Reader")

convos = load_conversations()
messages = load_messages()
summaries = load_summaries()
emb_meta = load_embeddings_meta()
emotions = load_emotional_states()
funcs = load_functional()
entities = load_entity_attention()

analysable = convos[convos["is_analysable"]].sort_values("created_at", ascending=False)

# --- Conversation selector ---
# Check if navigated from another page
preselected = st.session_state.get("selected_conversation_id", None)

search = st.text_input("Search by title")
if search:
    matches = analysable[analysable["title"].str.contains(search, case=False, na=False)]
else:
    matches = analysable

if len(matches) == 0:
    st.info("No conversations match your search.")
    st.stop()

title_map = dict(zip(matches["conversation_id"], matches["title"] + " (" + matches["created_at"].dt.strftime("%Y-%m-%d") + ")"))
conv_ids = matches["conversation_id"].tolist()

# Set default index for preselected
default_idx = 0
if preselected and preselected in conv_ids:
    default_idx = conv_ids.index(preselected)

selected_id = st.selectbox(
    "Select conversation",
    conv_ids,
    index=default_idx,
    format_func=lambda x: title_map.get(x, x),
)

# Clear preselection after use
if "selected_conversation_id" in st.session_state:
    del st.session_state["selected_conversation_id"]

if not selected_id:
    st.stop()

# --- Load conversation data ---
conv_row = convos[convos["conversation_id"] == selected_id].iloc[0]
conv_msgs = messages[messages["conversation_id"] == selected_id].sort_values("msg_index")

# --- Layout ---
msg_col, meta_col = st.columns([3, 1])

with meta_col:
    st.subheader("Metadata")
    st.caption(f"**Date:** {str(conv_row['created_at'])[:10]}")
    st.caption(f"**Messages:** {conv_row['msg_count']}")
    st.caption(f"**Duration:** {conv_row['duration_minutes']:.0f} min")
    st.caption(f"**Era:** {conv_row['model_era']}")
    st.caption(f"**Type:** {conv_row['conversation_type']}")

    # Enrichments
    st.divider()
    emo_row = emotions[emotions["conversation_id"] == selected_id]
    if len(emo_row) > 0:
        st.caption(f"**Emotion:** {emo_row.iloc[0].get('emotion_primary', '')}")
        sec = emo_row.iloc[0].get("emotion_secondary", "")
        if sec:
            st.caption(f"**Secondary:** {sec}")

    func_row = funcs[funcs["conversation_id"] == selected_id]
    if len(func_row) > 0:
        st.caption(f"**Function:** {func_row.iloc[0].get('function_primary', '')}")

    # Entities
    conv_entities = entities[entities["conversation_id"] == selected_id].sort_values("mention_count", ascending=False)
    if len(conv_entities) > 0:
        st.divider()
        st.caption("**Entities:**")
        for _, e in conv_entities.head(10).iterrows():
            st.caption(f"  {e['entity']}: {e['mention_count']}")

    # Embedding info
    emb_row = emb_meta[emb_meta["conversation_id"] == selected_id]
    if len(emb_row) > 0:
        er = emb_row.iloc[0]
        st.divider()
        st.caption(f"**Cluster:** {er.get('semantic_cluster', '')}")
        st.caption(f"**Centroid dist:** {er.get('centroid_distance', 0):.3f}")
        nn_id = er.get("nn_conversation_id", "")
        if nn_id:
            nn_title = convos[convos["conversation_id"] == nn_id]["title"]
            nn_name = nn_title.iloc[0] if len(nn_title) > 0 else nn_id
            st.caption(f"**Nearest neighbor:** {nn_name} ({er.get('nn_distance', 0):.3f})")

    # Summary
    summ = summaries[summaries["conversation_id"] == selected_id]
    if len(summ) > 0:
        st.divider()
        st.caption("**Summary:**")
        st.caption(summ.iloc[0]["summary"])

with msg_col:
    st.subheader(conv_row["title"])

    for _, msg in conv_msgs.iterrows():
        role = msg["role"]
        text = msg["text"] if pd.notna(msg["text"]) else ""
        if not text.strip():
            continue

        with st.chat_message(role):
            if len(text) > 3000:
                st.write(text[:3000])
                with st.expander("Show full message"):
                    st.write(text)
            else:
                st.write(text)
