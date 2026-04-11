"""
SUBTEXT Corpus Explorer
Interactive exploration of 1,573 conversations across 11 months.
"""

import streamlit as st
from app_utils import load_conversations, load_summaries, load_embeddings_meta

st.set_page_config(
    page_title="SUBTEXT Corpus Explorer",
    page_icon="\u2609",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SUBTEXT Corpus Explorer")

# Load basic stats
convos = load_conversations()
analysable = convos[convos["is_analysable"]]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversations", f"{len(analysable):,}")
col2.metric("Messages", f"{convos['msg_count'].sum():,}")
col3.metric("Date Range", f"{str(analysable['created_at'].min())[:10]} to {str(analysable['created_at'].max())[:10]}")
col4.metric("Model Eras", f"{analysable['model_era'].nunique()}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Era Distribution")
    era_counts = analysable["model_era"].value_counts()
    for era, count in era_counts.items():
        st.text(f"  {era}: {count}")

with col2:
    st.subheader("Conversation Types")
    type_counts = analysable["conversation_type"].value_counts()
    for t, count in type_counts.items():
        st.text(f"  {t}: {count}")

st.divider()

st.markdown("""
### Pages

- **Semantic Search** -- Ask a question, find conversations by meaning
- **Semantic Map** -- Interactive UMAP scatter plot of the entire corpus
- **Entity Explorer** -- Who you talk about, where they live semantically
- **Timeline** -- Volume, drift, and emotional composition over time
- **Conversation Reader** -- Read any conversation with full metadata
""")

st.caption("Local only. Contains personal data. Not for deployment.")
