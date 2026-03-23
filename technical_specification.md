**TECHNICAL SPECIFICATION**

**Self-Organized Cognition**

Longitudinal Analysis of Cognitive Patterns

600MB ChatGPT Conversation Corpus

*From Research Brief to Execution Plan*

March 2026 \| v1.0

1\. Data Ingestion Pipeline

This section translates **Phase 1: Data Extraction & Inventory** into
executable technical steps.

1.1 Source Format & Extraction

ChatGPT exports arrive as a ZIP archive containing a top-level
conversations.json file plus media attachments. The JSON is an array of
conversation objects.

  ------------------------ ------------------ ------------------------------------
  **Field**                **Type**           **Description**

  **id**                   string (UUID)      Unique conversation identifier

  **title**                string \| null     Auto-generated conversation title

  **create_time**          float (epoch)      Unix timestamp of conversation
                                              creation

  **update_time**          float (epoch)      Unix timestamp of last update

  **mapping**              dict\<id, node\>   DAG of message nodes (not a flat
                                              list)

  **moderation_results**   list               Safety filter triggers (usually
                                              empty)
  ------------------------ ------------------ ------------------------------------

1.2 Message Node Structure

Each node in the mapping dict contains:

> node.id → str \# unique node ID
>
> node.message → Message\|null \# null for root/system nodes
>
> node.parent → str\|null \# parent node ID
>
> node.children → list\[str\] \# child node IDs
>
> message.author.role → \'system\'\|\'user\'\|\'assistant\'\|\'tool\'
>
> message.content.parts → list\[str\|dict\] \# text or multimodal
>
> message.create_time → float\|null

**Critical note:** The mapping is a DAG (directed acyclic graph), not a
flat array. Conversations with edits or regenerations will have
branching paths. The linearization strategy must be defined (e.g.,
follow the last child at each branch point, or extract all branches as
separate threads).

1.3 Extraction Script Specification

  --------------------- -------------------------------------------------
  **Step**              **Implementation**

  **Decompress**        zipfile.extractall() → identify
                        conversations.json path

  **Parse JSON**        json.load() with error handling for malformed
                        UTF-8; expect 500MB+ in memory

  **Linearize DAG**     BFS/DFS walk of mapping; follow
                        last-child-at-branch heuristic; flag branching
                        convos

  **Flatten to rows**   One row per message: conversation_id, msg_index,
                        role, text, timestamp, token_est, has_attachment

  **Write Parquet**     pandas DataFrame → .parquet (pyarrow) for fast
                        columnar queries; also export .csv for inspection
  --------------------- -------------------------------------------------

1.4 Target Schema: messages.parquet

  --------------------- ------------ -------------- ---------------------------
  **Column**            **dtype**    **Nullable**   **Notes**

  **conversation_id**   string       No             FK to conversations table

  **msg_index**         int32        No             Position in linearized
                                                    thread

  **role**              category     No             system \| user \| assistant
                                                    \| tool

  **text**              string       Yes            Concatenated text from
                                                    content.parts

  **timestamp**         datetime64   Yes            UTC; converted from epoch
                                                    float

  **token_count**       int32        Yes            tiktoken cl100k_base
                                                    estimate

  **char_count**        int32        Yes            len(text)

  **word_count**        int32        Yes            Whitespace split count

  **has_code**          bool         No             Contains markdown code
                                                    fences

  **has_attachment**    bool         No             content.parts contains
                                                    non-text dict

  **is_branched**       bool         No             Conversation had
                                                    regeneration/edits
  --------------------- ------------ -------------- ---------------------------

1.5 Target Schema: conversations.parquet

  --------------------------- ------------ -------------------------------------
  **Column**                  **dtype**    **Notes**

  **conversation_id**         string       Primary key

  **title**                   string       May be null or auto-generated

  **created_at**              datetime64   First message timestamp (UTC)

  **updated_at**              datetime64   Last message timestamp (UTC)

  **duration_minutes**        float32      updated_at - created_at

  **msg_count**               int32        Total messages (all roles)

  **user_msg_count**          int32        User messages only

  **user_token_total**        int32        Sum of user token_count

  **assistant_token_total**   int32        Sum of assistant token_count

  **hour_of_day**             int8         Hour of created_at (local TZ, must
                                           configure)

  **day_of_week**             int8         0=Mon through 6=Sun

  **has_code**                bool         Any message contains code

  **is_branched**             bool         DAG has branching nodes
  --------------------------- ------------ -------------------------------------

2\. Data Quality & Cleaning Protocol

2.1 Known Edge Cases

  ------------------- ------------------------ ---------------------------
  **Issue**           **Detection**            **Resolution**

  **Empty             msg_count == 0 or only   Flag and exclude from
  conversations**     system msgs              analysis; count separately

  **Plugin/tool       role == \'tool\' or      Tag with plugin_used;
  conversations**     content_type != \'text\' analyze separately

  **Custom GPT        System prompt differs    Tag with gpt_config;
  sessions**          from default             segment for comparison

  **Multi-modal       content.parts contains   Extract text portions; flag
  content**           dict objects             as multimodal

  **Truncated         Missing timestamps,      Include with quality_flag =
  conversations**     partial mapping          \'partial\'

  **Null timestamps** create_time is None      Impute from neighboring
                                               messages or
                                               conversation-level time

  **Massive single    token_count \> 10k       Tag as bulk_input; handle
  messages**          (paste-ins)              separately in NLP
  ------------------- ------------------------ ---------------------------

2.2 Cleaning Pipeline

-   **Step 1 --- Dedup:** Remove exact-duplicate conversation IDs
    (export artifacts)

-   **Step 2 --- Null audit:** Report % null per column; define
    imputation or exclusion rules

-   **Step 3 --- Timezone normalization:** Convert all timestamps to
    user's local timezone (configure once)

-   **Step 4 --- Text normalization:** Strip system prompt boilerplate;
    normalize whitespace; handle unicode edge cases

-   **Step 5 --- Quality flags:** Add quality_score column (0--1) based
    on completeness, timestamp presence, text length

3\. Analysis Module Specifications

Each research phase maps to one or more independent Python modules. All
modules read from the cleaned Parquet files and write structured output
(JSON + visualizations) to an outputs/ directory.

3.1 Module: Structural Analysis (Phase 2)

*Maps to research question:* **Cognitive Signature**

  ------------------- ---------------------------------------------------
  **Analysis**        **Technical Implementation**

  **Turn dynamics**   Compute user_token_ratio = user_tokens /
                      total_tokens per conversation; distribution plot +
                      trend over time

  **Conversation      Measure message length per turn position; cluster
  shapes**            conversations by shape vector (deepening, plateau,
                      spiral) using DTW + k-means

  **Opening           Classify first user message: question (contains ?),
  taxonomy**          command (imperative verb), statement, fragment,
                      code. Use spaCy POS tagging + heuristics

  **Rhetorical        Claude API few-shot classifier: tag each user turn
  moves**             as clarify \| expand \| reframe \| challenge \|
                      validate \| redirect \| meta. Batch via Claude
                      Batch API for cost efficiency

  **Conversation      Messages per minute; inter-message delay
  velocity**          distribution; detect engagement intensity patterns
  ------------------- ---------------------------------------------------

**Output:** structural_report.json + 8--12 matplotlib/plotly figures

3.2 Module: Thematic & Content Analysis (Phase 3)

*Maps to research question:* **Functional Taxonomy + Secondary
Questions**

  ------------------- ---------------------------------------------------
  **Analysis**        **Technical Implementation**

  **Topic modeling**  BERTopic with sentence-transformers
                      (all-MiniLM-L6-v2 for speed, or all-mpnet-base-v2
                      for quality); run on concatenated user messages per
                      conversation; UMAP dimensionality reduction +
                      HDBSCAN clustering

  **Functional        Claude API batch classification: each conversation
  classification**    tagged with primary function: learning \|
                      problem-solving \| emotional-processing \| creative
                      \| planning \| coding \| research \|
                      social-rehearsal \| meta-cognition \| other

  **Sentiment         Two-track approach: (a) VADER for coarse valence
  analysis**          per message; (b) Claude API for nuanced emotional
                      state per conversation (granular labels: anxious,
                      curious, frustrated, playful, reflective, etc.)

  **Abstraction       Claude API classification of each user message into
  tracking**          abstraction tier: concrete (specific facts/tasks)
                      \| general (principles/patterns) \| meta (thinking
                      about thinking). Plot distribution over time

  **Recurring         Semantic similarity (cosine) between conversation
  threads**           embeddings; find clusters that span \>30 days;
                      identify topic recurrence vs. resolution patterns
  ------------------- ---------------------------------------------------

**Output:** topic_model.html (interactive BERTopic viz),
thematic_report.json, function_distribution.png, sentiment_timeline.html

3.3 Module: Longitudinal & Developmental Analysis (Phase 4)

*Maps to research question:* **Cognitive Evolution + Force Functions**

  ------------------- ---------------------------------------------------
  **Analysis**        **Technical Implementation**

  **Vocabulary        Rolling window (30-day) type-token ratio, unique
  evolution**         word growth rate, new domain-specific vocabulary
                      introduction rate

  **Complexity        Flesch-Kincaid, avg sentence length, subordinate
  metrics**           clause frequency (spaCy dependency parse) --- all
                      tracked as monthly time series

  **Question          Classify questions by Bloom's taxonomy level
  evolution**         (knowledge → synthesis) using Claude API; track
                      distribution shift over time

  **Inflection        Change point detection (ruptures library, PELT
  detection**         algorithm) on: usage frequency, topic distribution,
                      sentiment, complexity. Cross-reference with known
                      life events if available

  **Force function    Identify sequences where user self-imposes
  test**              constraints, escalates difficulty, or restructures
                      interaction patterns. Operationalize as: (a)
                      increasing question complexity within topic
                      clusters, (b) deliberate reframing moves, (c)
                      meta-cognitive turns that redirect conversation
                      purpose
  ------------------- ---------------------------------------------------

**Output:** longitudinal_timeline.html (interactive Plotly timeline),
inflection_points.json, development_report.json

4\. Computational Architecture

4.1 Processing Pipeline

> chatgpt_export.zip
>
> │
>
> ├── 01_extract.py → raw_conversations.json
>
> ├── 02_parse.py → messages.parquet + conversations.parquet
>
> ├── 03_clean.py → messages_clean.parquet + quality_report.json
>
> ├── 04_embed.py → embeddings.npy (sentence-transformers)
>
> ├── 05_structural.py → structural_report.json + figures/
>
> ├── 06_topics.py → topic_model/ + thematic_report.json
>
> ├── 07_sentiment.py → sentiment_scores.parquet
>
> ├── 08_longitudinal.py → timeline.json + inflection_points.json
>
> ├── 09_llm_classify.py → classifications.parquet (Claude Batch API)
>
> └── 10_synthesis.py → cognitive_profile.json + final_report/

4.2 Dependency Stack

  ------------------- ----------------------- -------------------------------------
  **Layer**           **Package**             **Purpose & Version**

  **Core**            Python 3.11+            Runtime; 3.11 for performance
                                              improvements on string processing

  **Data**            pandas, pyarrow         DataFrame ops + Parquet I/O; pyarrow
                                              for columnar speed

  **NLP**             spaCy (en_core_web_lg)  Tokenization, POS tagging, dependency
                                              parsing, NER

  **Tokenization**    tiktoken                GPT token count estimates
                                              (cl100k_base encoding)

  **Embeddings**      sentence-transformers   Conversation-level embeddings for
                                              clustering/similarity

  **Topic Model**     BERTopic + HDBSCAN      Unsupervised thematic clustering with
                                              UMAP reduction

  **Sentiment**       vaderSentiment          Coarse sentiment baseline (compound
                                              score per message)

  **Change            ruptures                PELT algorithm for inflection point
  Detection**                                 detection in time series

  **Visualization**   plotly, matplotlib      Interactive timelines (plotly);
                                              static figures (matplotlib)

  **LLM**             anthropic (SDK)         Claude API for qualitative
                                              classification; use Batch API for
                                              cost
  ------------------- ----------------------- -------------------------------------

4.3 Resource Estimates

  ------------------------ ----------------------- -----------------------
  **Resource**             **Estimate**            **Assumption**

  **Raw JSON in memory**   1.5--2 GB               \~3x compressed size

  **Parquet on disk**      200--400 MB             Columnar compression

  **Embedding matrix**     50--200 MB              384-dim × \~10k
                                                   conversations

  **Claude API (Batch)**   \$15--50 total          \~5M input tokens,
                                                   Haiku for
                                                   classification

  **Processing time**      2--4 hours              Full pipeline end to
                                                   end on M-series Mac
  ------------------------ ----------------------- -----------------------

5\. LLM-Assisted Analysis Protocol

Several analysis steps use the Claude API for qualitative
classification. This section defines the prompting strategy, quality
controls, and cost management approach.

5.1 Classification Tasks

  --------------- ---------------- ------------------- -------------------
  **Task**        **Input Unit**   **Label Set**       **Model**

  **Functional    Full             10 categories       claude-haiku-4-5
  class**         conversation                         

  **Rhetorical    Single user turn 7 categories        claude-haiku-4-5
  moves**                                              

  **Abstraction   Single user turn 3 tiers             claude-haiku-4-5
  tier**                                               

  **Emotional     Full             Open label +        claude-sonnet-4-5
  state**         conversation     valence             

  **Bloom's       User questions   6 levels            claude-haiku-4-5
  taxonomy**                                           

  **Force         Conversation     Binary +            claude-sonnet-4-5
  function ID**   sequence         description         
  --------------- ---------------- ------------------- -------------------

5.2 Prompting Strategy

-   **Structured output:** All classification prompts request JSON
    output with a defined schema; parse with json.loads() and validate
    against Pydantic models

-   **Few-shot examples:** Each classifier includes 5--10 hand-labeled
    examples spanning edge cases; examples are stored in
    prompts/examples/ as JSON

-   **Calibration set:** Manually label 50--100 items per task; run
    Claude against same set; compute agreement (Cohen's kappa); iterate
    prompt until kappa \> 0.75

-   **Batch API:** Use Claude Batch API for all classification runs (50%
    cost reduction); batch sizes of 1000--5000 messages; poll for
    completion

5.3 Cost Management

-   **Tier by model:** Use Haiku for simple categorical classification;
    Sonnet only for tasks requiring nuance (emotional state, force
    function identification)

-   **Input truncation:** For conversation-level classification, pass
    first 2000 tokens + last 500 tokens rather than full conversations

-   **Cache embeddings:** Compute sentence-transformer embeddings once;
    reuse across topic modeling, similarity, and clustering

-   **Incremental runs:** Track which conversations have been
    classified; only re-run on new/modified data

6\. Visualization & Output Specifications

6.1 Phase 1 --- Data Inventory Dashboard

-   **Corpus overview card:** Total conversations, messages, date range,
    total tokens

-   **Temporal heatmap:** Day-of-week × hour-of-day grid; color =
    message count (plotly heatmap)

-   **Volume time series:** Daily message count with 7-day rolling
    average (dual-axis: count + cumulative)

-   **Length distributions:** Histogram of conversation length
    (messages) + user message length (tokens); log scale if skewed

-   **Role ratio:** Stacked bar or area chart of user vs. assistant
    token volume over time

6.2 Phase 2 --- Structural Figures

-   **Opening pattern pie chart:** Distribution of first-message types

-   **Conversation shape gallery:** Representative examples of each
    shape cluster (line plots of message length by turn)

-   **Velocity distribution:** Histogram of inter-message delays;
    annotate median and 90th percentile

6.3 Phase 3 --- Thematic Figures

-   **BERTopic intertopic distance map:** Interactive HTML visualization
    (built-in to BERTopic)

-   **Function distribution:** Horizontal bar chart of conversation
    function categories

-   **Sentiment timeline:** Scatter + smoothed line of sentiment over
    time; color by topic cluster

-   **Abstraction level stacked area:** Proportion of concrete / general
    / meta over monthly windows

6.4 Phase 4 --- Longitudinal Figures

-   **Cognitive timeline:** Multi-track Plotly timeline: usage
    intensity, complexity score, topic diversity, sentiment --- all
    aligned on same x-axis with inflection points marked

-   **Vocabulary growth curve:** Cumulative unique terms over time with
    domain-specific vocabulary highlighted

-   **Bloom's distribution shift:** Stacked area of question taxonomy
    levels over quarterly windows

7\. Bias Mitigation & Validity Controls

The researcher-subject overlap is the primary methodological risk. These
controls are built into the pipeline, not afterthoughts.

  -------------------- -------------------------- --------------------------
  **Risk**             **Mitigation**             **Implementation**

  **Confirmation       Run all automated analyses Pre-registration file:
  bias**               before subjective          hypotheses.json written
                       interpretation; document   before Phase 2 begins
                       hypotheses before seeing   
                       results                    

  **Cherry-picking**   Report full distributions, Random stratified sampling
                       not selected examples;     for qualitative
                       automated selection of     deep-dives; seed = 42 for
                       representative samples     reproducibility

  **Avoidance bias**   LLM analysis is            Automated coverage report:
                       comprehensive by default;  topics/functions with 0
                       flag topics the researcher manual annotations flagged
                       skips in manual review     

  **Third-party        NER-based PII scan before  spaCy NER + regex patterns
  privacy**            any data leaves local      for names, emails, phone
                       environment                numbers; auto-redact in
                                                  shared outputs

  **Temporal           Control for ChatGPT model  Tag conversations by model
  confounds**          changes (GPT-3.5 → GPT-4 → era; include model_era as
                       GPT-4o) which affect       covariate in longitudinal
                       conversation dynamics      analysis
  -------------------- -------------------------- --------------------------

8\. Execution Plan & Milestones

  -------- ---------------- ------------------------- ------------------ ------------
  **\#**   **Phase**        **Deliverable**           **Dependencies**   **Est.
                                                                         Time**

  1a       Extract & Parse  messages.parquet +        Raw ZIP upload     2--4 hrs
                            conversations.parquet                        

  1b       Clean & Validate Cleaned Parquet +         1a                 1--2 hrs
                            quality_report.json                          

  1c       Inventory Report Phase 1 Report (stats +   1b                 2--3 hrs
                            figures)                                     

  2        Structural       Structural analysis       1b                 4--6 hrs
                            report + figures                             

  3a       Embeddings       embeddings.npy            1b                 1--2 hrs

  3b       Topic Modeling   BERTopic model + viz      3a                 2--4 hrs

  3c       LLM              classifications.parquet   1b + API key       3--5 hrs
           Classification                                                

  3d       Thematic Report  Phase 3 Report            3b + 3c            3---4 hrs

  4        Longitudinal     Timeline + inflection     3c + 3b            4--6 hrs
                            points + Phase 4 Report                      

  5        Synthesis        Cognitive Self-Model      All prior          6--10 hrs
                            (collaborative)                              
  -------- ---------------- ------------------------- ------------------ ------------

**Total estimated effort:** 30--50 hours across all phases (not
including researcher reflection time in Phase 5)

**Critical path:** 1a → 1b → 3a → 3b → 4 → 5 (embedding and topic
modeling gate the longitudinal analysis)

9\. Immediate Next Steps

To begin execution, the following are required:

  -------- -------------------------------- -------------------- -------------
  **\#**   **Action**                       **Owner**            **Status**

  1        Upload ChatGPT ZIP export        Researcher           **PENDING**

  2        Configure local timezone for     Researcher           **PENDING**
           timestamp normalization                               

  3        Set up Python environment        Claude + Researcher  **PENDING**
           (requirements.txt provided                            
           below)                                                

  4        Run 01_extract.py + 02_parse.py  Claude               **BLOCKED**
           on export                                             

  5        Generate Phase 1 Data Inventory  Claude               **BLOCKED**
           Report                                                

  6        Write hypotheses.json            Researcher           **PENDING**
           (pre-registration) before Phase                       
           2                                                     

  7        Provision Anthropic API key for  Researcher           **PENDING**
           Claude Batch API access                               
  -------- -------------------------------- -------------------- -------------

Appendix A: requirements.txt

> \# Core
>
> pandas\>=2.1
>
> pyarrow\>=14.0
>
> tiktoken\>=0.5
>
> \# NLP
>
> spacy\>=3.7
>
> vaderSentiment\>=3.3
>
> \# Embeddings & Topic Modeling
>
> sentence-transformers\>=2.2
>
> bertopic\>=0.16
>
> hdbscan\>=0.8
>
> umap-learn\>=0.5
>
> \# Change Detection
>
> ruptures\>=1.1
>
> \# Visualization
>
> plotly\>=5.18
>
> matplotlib\>=3.8
>
> seaborn\>=0.13
>
> \# LLM Integration
>
> anthropic\>=0.40
>
> \# Utilities
>
> tqdm\>=4.66
>
> pydantic\>=2.5

Appendix B: Project Directory Structure

> self-organized-cognition/
>
> ├── data/
>
> │ ├── raw/ \# Original ZIP + extracted JSON
>
> │ ├── processed/ \# Parquet files
>
> │ └── embeddings/ \# .npy embedding matrices
>
> ├── scripts/
>
> │ ├── 01_extract.py
>
> │ ├── 02_parse.py
>
> │ ├── 03_clean.py
>
> │ ├── \...
>
> │ └── 10_synthesis.py
>
> ├── prompts/
>
> │ ├── examples/ \# Few-shot examples per classifier
>
> │ └── templates/ \# Prompt templates (Jinja2)
>
> ├── outputs/
>
> │ ├── figures/ \# All visualizations
>
> │ ├── reports/ \# Phase reports (JSON + rendered)
>
> │ └── models/ \# BERTopic model artifacts
>
> ├── hypotheses.json \# Pre-registration
>
> ├── requirements.txt
>
> └── README.md
