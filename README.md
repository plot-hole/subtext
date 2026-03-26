# SUBTEXT

**Longitudinal NLP research on a personal corpus of 1,573 AI and human conversations.**

research log: https://www.notion.so/32c77cb7d15d81e88606ecec5566e59f

This project applies computational linguistics to a personal corpus of conversational data collected over 11 months (March 2025 – February 2026). The corpus includes AI dialogue, human conversation transcripts, technical problem-solving, creative writing, philosophical inquiry, and ordinary searching. The analysis spans structural, thematic, and longitudinal dimensions.

---

## The Corpus

1,573 conversations. ~11 months.

The raw data is a ChatGPT export — a ZIP archive containing `conversations.json`, where each conversation is a DAG (directed acyclic graph) of message nodes, not a flat array. Conversations with edits or regenerations have branching paths. The extraction pipeline linearizes these DAGs using a last-child-at-branch heuristic and flags branching conversations for separate analysis.

---

## Pipeline

The project runs as a sequential pipeline. Each step reads from the output of the previous step and writes structured output (Parquet, JSON, HTML) to `outputs/`.

| # | Script | Question |
|---|--------|----------|
| 01 | `extract` | Can I reliably decompress, parse, and linearize a DAG-structured ChatGPT export? |
| 02 | `parse` | What does the corpus look like as structured data? (→ `messages.parquet` + `conversations.parquet`) |
| 03 | `clean` | What's broken? Nulls, duplicates, truncations, encoding, empty conversations, plugin artifacts. |
| 04 | `edge_cases` | What doesn't fit? Massive paste-ins, multi-modal content, custom GPT sessions, tool-role messages. |
| 04 | `report` | Data inventory dashboard — temporal heatmaps, volume time series, length distributions, role ratios. |
| 05 | `enrich` | Token counts (tiktoken cl100k_base), code detection, quality scores, timestamp normalization. |
| 06 | `pii_scan` | PII detection and anonymization — spaCy NER + regex patterns for names, emails, phone numbers, workplaces. |
| 07 | `quality_report` | Automated quality audit — % null per column, completeness scores, coverage validation. |
| 08 | `turn_dynamics` | How do conversations flow? Token ratios, message length per turn position, engagement intensity. |
| 09 | `conversation_shapes` | What shapes do conversations take? Cluster by length vector using DTW + k-means (deepening, plateau, spiral). |
| 10 | `opening_taxonomy` | How do conversations begin? Classify first user message — question, command, statement, fragment, code — via spaCy POS tagging. |
| 11 | `topics` | What latent thematic structure exists, and how do themes shift over time? BERTopic + sentence-transformers + UMAP + HDBSCAN. |
| 12 | `summarize` | Compression layer — conversation-level summaries for downstream analysis. |
| 13 | `functional_classify` | What is the user doing in each conversation? 12-category functional taxonomy via Claude Haiku Batch API. |
| 14 | `emotional_state` | What emotional states appear across conversations? 8-category classification + GoEmotions baseline validation. |
| 15 | `goemotions_baseline` | Local GoEmotions transformer baseline — zero-API-cost emotion detection for comparison with LLM classifications. |
| 16 | `length_weight_analysis` | How do count-based vs message-weighted distributions differ? Reveals categories inflated by short conversations. |
| 17 | `frame_adoption` | How does the user respond to AI framing? Message-level classification of adopt/extend/redirect/reject/ignore/steer. |
| 18 | `vocab_transfer` | Which AI words and phrases permanently entered the user's vocabulary? Corpus-level NLP via spaCy + wordfreq — no API calls. |
| 19 | `hypothesis_extraction` | What predictions, intuitions, assessments, and interpretations did the user express? Two-pass Batch API extraction with referent resolution. |
| 20 | `belief_trajectories` | How do beliefs form, evolve, persist, and dissolve over time? Thread clustering + provenance classification + frame adoption overlay. |

### Planned

| # | Module | Question |
|---|--------|----------|
| — | `embeddings` | Conversation-level embeddings (sentence-transformers) → vector store (FAISS) for semantic search and retrieval. |
| — | `knowledge_graph` | What does the relational structure of a personal corpus look like extracted purely from language? spaCy + NetworkX + Louvain. |
| — | `longitudinal` | How does cognition change over time? Vocabulary evolution, complexity metrics, inflection detection (PELT via ruptures). |

---

## Data Architecture

```
chatgpt_export.zip
│
├── 01_extract    → raw_conversations.json
├── 02_parse      → messages.parquet + conversations.parquet
├── 03_clean      → messages_clean.parquet + quality_report.json
├── 05_enrich     → messages_enriched.parquet
├── 06_pii_scan   → messages_anonymized.parquet
│
├── 08-10         → structural_report.json + figures/
├── 11_topics     → topic_model/ + topic_viz.html
├── 12_summarize  → summaries.parquet
│
├── 13_functional → functional_classifications.parquet
├── 14_emotional  → emotional_states.parquet
├── 15_goemotions → goemotions_baseline (local transformer)
├── 16_length_wt  → weighted distribution figures
├── 17_frame      → frame_adoption.parquet
├── 18_vocab      → vocab_transfer.parquet
├── 19_hypothesis → hypotheses.parquet + hypothesis_catalog.csv
└── 20_belief     → belief_threads.parquet + belief_provenance.parquet + belief_thread_summaries.parquet
```

### Core Schemas

**messages.parquet**

| Column | Type | Description |
|--------|------|-------------|
| conversation_id | string | FK to conversations table |
| msg_index | int32 | Position in linearized thread |
| role | category | system \| user \| assistant \| tool |
| text | string | Concatenated text from content.parts |
| timestamp | datetime64 | UTC |
| token_count | int32 | tiktoken cl100k_base estimate |
| has_code | bool | Contains markdown code fences |
| has_attachment | bool | Non-text content in parts |
| is_branched | bool | Conversation had edits/regenerations |

**conversations.parquet**

| Column | Type | Description |
|--------|------|-------------|
| conversation_id | string | Primary key |
| title | string | Auto-generated or null |
| created_at | datetime64 | First message timestamp (UTC) |
| duration_minutes | float32 | Last - first message |
| msg_count | int32 | Total messages |
| user_msg_count | int32 | User messages only |
| user_token_total | int32 | Sum of user token counts |
| assistant_token_total | int32 | Sum of assistant token counts |
| hour_of_day | int8 | Hour of creation (local TZ) |
| has_code | bool | Any message contains code |

---

## Methodology

I'm including decisions and tradeoffs for posterity.

Example: the first topic modeling run produced 27 topics whose top terms were all stopwords (`the`, `and`, `it`, `to`). Switching from raw conversation text to pre-generated summaries, adding stopword removal, and using KeyBERTInspired representations fixed it completely. That debugging process is documented in the notebook because it's more instructive than the clean output.

### Validity Controls

The researcher-subject overlap is the primary methodological risk. These controls are built into the pipeline:

| Risk | Mitigation |
|------|------------|
| **Confirmation bias** | Automated analyses run before subjective interpretation. Pre-registration (`hypotheses.json`) written before Phase 2. |
| **Cherry-picking** | Full distributions reported, not selected examples. Random stratified sampling (seed=42) for qualitative deep-dives. |
| **Avoidance bias** | Automated coverage report flags topics with zero manual annotations. |
| **Third-party privacy** | NER-based PII scan before any data leaves local environment. |
| **Temporal confounds** | Conversations tagged by model era (GPT-3.5 → GPT-4 → GPT-4o); model_era included as covariate in longitudinal analysis. |

### Evaluation

There's no answer key for "the correct interpretation of a conversation." So evaluation checks three things:
Stability — run the same analysis with different random seeds. If the topic model produces roughly the same clusters each time, the structure is real. If it changes dramatically between runs, it's fitting noise.
Alignment with known events — the corpus spans a period that includes GPT model transitions (GPT-3.5 → GPT-4 → GPT-4o) and other independently documented events. These are known structural changes that should show up as detectable shifts. A manual record of partial information is also used as reference.
Manual plausibility — pull random stratified samples from each cluster (seed=42 for reproducibility), read them, and ask: does this grouping make sense? Qualitative spot-checking catches obvious failures that quantitative metrics miss.

---

## Technical Stack

| Layer | Package | Purpose |
|-------|---------|---------|
| Core | Python 3.11+ | Runtime |
| Data | pandas, pyarrow | DataFrame ops + Parquet I/O |
| NLP | spaCy (en_core_web_trf) | Tokenization, POS, dependency parsing, NER |
| Tokenization | tiktoken | GPT token estimates (cl100k_base) |
| Embeddings | sentence-transformers | Conversation-level embeddings |
| Topic Modeling | BERTopic + HDBSCAN + UMAP | Unsupervised thematic clustering |
| Sentiment | vaderSentiment | Coarse sentiment baseline |
| Word Frequency | wordfreq | Baseline lexical frequency for vocab transfer detection |
| Change Detection | ruptures | PELT algorithm for inflection detection |
| Visualization | plotly, matplotlib | Interactive timelines + static figures |
| LLM | anthropic SDK | Claude Batch API for qualitative classification |


## Where This Is Headed

This project is the foundation for a broader research direction: **recursive meaning systems** — modeling how meaning forms, propagates, and revises itself across conversations, people, and time.

"Memory is not a recording, its a rewriting that never stops" -Frederic Bartlett

---

The technical specification for this project, including detailed schemas, edge case taxonomy, bias mitigation framework, LLM classification protocol with calibration procedures, and resource estimates, is in [`docs/technical_specification.md`](docs/technical_specification.md).
