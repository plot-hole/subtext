# Module 3.2a: Functional Classification

## What This Module Does

Classifies every analysable conversation into one of 10 functional categories using the Claude Batch API. This tells us *what the user was trying to accomplish* in each conversation — learning, problem-solving, coding, emotional processing, etc.

This is the first of several LLM-assisted classification tasks in Module 3.2 (Thematic & Content Analysis).

---

## Context: Where This Fits in the Pipeline

**Depends on:**
- `data/processed/conversations_clean.parquet` (from scripts 01-07)
- `data/processed/messages_clean.parquet` (from scripts 01-07)
- `outputs/reports/all_summaries.csv` (from script 12_summarize.py — Claude-generated conversation summaries)

**Produces:**
- `data/processed/functional_classifications.parquet` — one row per conversation with classification
- `outputs/reports/functional_classification_report.json` — distribution stats, quality metrics, cost
- `outputs/figures/functional_classification/` — 6-8 figures

**Will be consumed by:**
- Topic modeling cross-analysis (how do topics distribute across functions?)
- Longitudinal analysis (how does function usage change over time?)
- Cognitive signature synthesis

---

## Classification Taxonomy

Each conversation gets exactly ONE primary function label. This taxonomy was refined after reviewing the actual corpus summaries.

| Label | Description | Examples |
|-------|-------------|----------|
| `interpersonal_analysis` | Analyzing relationships, decoding social dynamics, mapping power structures, processing interactions with specific people | Analyzing a colleague's communication patterns, decoding workplace dynamics, mapping relational history |
| `emotional_processing` | Processing feelings, grief, anxiety, burnout, trauma — therapeutic rather than analytical | Caregiver burnout, processing a breakup, working through fear or shame |
| `creative_expression` | Writing poetry, analyzing art, personal mythology building, expressive journaling | Original poems, art interpretation, stream-of-consciousness writing, creative self-portraiture |
| `career_strategy` | Job preparation, role analysis, pitch development, conference planning, professional positioning | Director pitch prep, resume strategy, RSA conference planning, interview preparation |
| `self_modeling` | Asking the AI to analyze, profile, or mirror the user's own patterns, identity, or psychology | "What challenges would you face as my manager?", identity pillars, psychological threadmaps |
| `practical` | Quick utilitarian questions — health, logistics, products, how-to, factual lookups | Dog vet questions, lease decisions, stain removal, gaming PC specs, packing advice |
| `learning` | Seeking to understand a concept, domain, or skill | "Explain chaos magic", "What is information theory?", book analysis |
| `problem_solving` | Working through a specific technical or logical problem | Debugging, structuring a query, troubleshooting AC problems |
| `coding` | Writing, debugging, reviewing, or generating code as the primary activity | "Write a Python script that...", code review, refactoring |
| `social_rehearsal` | Drafting messages, emails, preparing for specific conversations | Follow-up emails to recruiters, LinkedIn messages, difficult conversation prep |
| `work_professional` | Work-specific analysis — security policy, governance documents, vendor review — that isn't career strategy | Data governance drafts, security questionnaire review, TPRM process, tool evaluation |
| `planning` | Organizing actions, trips, projects that aren't career-specific | Trip itineraries, retreat planning, moving logistics |

**Notes on classification:**
- `interpersonal_analysis` vs `emotional_processing`: If the conversation is primarily *analytical* (decoding signals, mapping dynamics, tracking patterns), use `interpersonal_analysis`. If it's primarily *feeling* (grief, anxiety, venting, therapeutic processing), use `emotional_processing`. Many conversations will have elements of both — choose the dominant mode.
- `career_strategy` vs `work_professional`: Career strategy is about *the user's trajectory* (getting a role, building a portfolio, conference networking). Work professional is about *doing the current job* (writing policies, reviewing vendors, building governance documents).
- `self_modeling` vs `interpersonal_analysis`: Self-modeling is when the AI is asked to analyze *the user themselves*. Interpersonal analysis is when the AI helps analyze *other people or dynamics between people*.
- `creative_expression` includes both original creative work AND deep analysis of creative works (poems, art, mythology) when the analysis itself is exploratory/expressive rather than academic.
- If a summary is clearly a persona capture (the summarizer returned the GPT's voice instead of an actual summary), classify based on whatever contextual clues are available, and set confidence low.

---

## Technical Implementation

### Input

Use the **conversation summaries** from `all_summaries.csv` as input, not the raw messages. The summaries are 2-4 sentences each and capture the essence of each conversation. This is much cheaper and more reliable than sending full conversations.

For conversations where the summary is `[SUMMARIZATION FAILED]` or missing, use the first 2000 tokens of concatenated user messages as fallback input.

### Model

Use `claude-haiku-4-5` for this task. It's a straightforward categorical classification — Haiku is fast, cheap, and accurate enough. Reserve Sonnet for tasks requiring nuance (emotional state, force functions).

### Batch API

Use the same Batch API infrastructure established in `12_summarize.py`:
1. Build JSONL request file with one request per conversation
2. Submit batch via `client.messages.batches.create()`
3. Poll for completion
4. Retrieve results
5. Retry failures via standard API
6. Fill placeholders for permanent failures

### System Prompt

```
You are a research assistant classifying conversations by their primary function.

Given a summary of a conversation between a user and an AI assistant, classify the conversation into exactly ONE of these categories:

- interpersonal_analysis: The user is analyzing relationships, decoding social dynamics, mapping power structures, or processing interactions with specific people. The mode is analytical rather than emotional.
- emotional_processing: The user is processing feelings, grief, anxiety, burnout, or trauma. The mode is therapeutic rather than analytical.
- creative_expression: The user is writing poetry, analyzing art, building personal mythology, or doing expressive journaling.
- career_strategy: The user is preparing for jobs, analyzing roles, building pitches, planning for conferences, or positioning themselves professionally.
- self_modeling: The user is asking the AI to analyze, profile, or mirror their own patterns, identity, or psychology.
- practical: Quick utilitarian questions — health, logistics, products, how-to, factual lookups.
- learning: The user is seeking to understand a concept, domain, or skill.
- problem_solving: The user is working through a specific technical or logical problem.
- coding: Writing, debugging, reviewing, or generating code is the primary activity.
- social_rehearsal: The user is drafting messages, emails, or preparing for specific conversations.
- work_professional: Work-specific tasks — security policy, governance documents, vendor review, tool evaluation — that aren't about career strategy.
- planning: Organizing actions, trips, or projects that aren't career-specific.

Choose the BEST single category. If a conversation spans multiple functions, choose the one that best describes the PRIMARY purpose.

Key distinctions:
- interpersonal_analysis vs emotional_processing: analytical (decoding, mapping, tracking) vs feeling (grief, venting, therapeutic)
- career_strategy vs work_professional: user's trajectory vs doing the current job
- self_modeling vs interpersonal_analysis: AI analyzing the user vs AI helping analyze other people

Respond with ONLY a JSON object in this exact format:
{"function": "<category>", "confidence": <0.0-1.0>, "secondary": "<category_or_null>"}

- "function": The primary function label (one of the 12 categories above)
- "confidence": Your confidence in the classification (0.0 to 1.0)
- "secondary": If the conversation clearly spans two functions, provide the secondary one. Otherwise null.

Do not include any other text, explanation, or markdown formatting. Only the JSON object.
```

### User Message Format

```
Classify this conversation:

Title: {title}
Summary: {summary}
```

If no summary is available (failed summarization), use:

```
Classify this conversation based on the user's messages:

Title: {title}
User messages:
{first 2000 tokens of concatenated user messages}
```

### Output Schema

**functional_classifications.parquet:**

| Column | dtype | Description |
|--------|-------|-------------|
| `conversation_id` | string | FK to conversations table |
| `function_primary` | category | One of the 10 function labels |
| `function_confidence` | float32 | Model confidence (0.0-1.0) |
| `function_secondary` | category (nullable) | Secondary function if applicable |
| `classification_input` | category | "summary" or "fallback_messages" |
| `input_tokens` | int32 | Tokens used for this classification |
| `output_tokens` | int32 | Tokens used for this classification |

---

## Calibration Protocol

Before running the full batch:

1. **Sample 50 conversations** — stratified by conversation_type from the conversations_clean parquet (standard, code_heavy, single_turn, tool_assisted, custom_gpt, multimodal). Use seed=42.

2. **Manually label the 50 conversations** — read each summary (or the conversation if needed) and assign a function label. Save as `prompts/examples/functional_calibration_set.json`:
   ```json
   [
     {"conversation_id": "...", "function": "coding", "notes": "Debugging a Python script"},
     ...
   ]
   ```

3. **Run Claude against the same 50** using the system prompt above.

4. **Compute Cohen's kappa** between manual labels and Claude labels. Target: kappa > 0.75.

5. **If kappa < 0.75:** Review disagreements, adjust the system prompt or category descriptions, and re-run. Document iterations.

6. **Save calibration results** to `outputs/reports/functional_calibration.json`.

**NOTE:** If the calibration set doesn't exist yet, the script should run without it and log a warning. The calibration is important for methodology but shouldn't block the full classification run.

---

## Figures to Generate

Save all figures to `outputs/figures/functional_classification/`.

### Figure 1: Function Distribution
Horizontal bar chart of all 12 categories sorted by count. Include count and percentage labels. Color each bar by function category. Use a consistent color palette defined at the top of the script.

### Figure 2: Function Distribution by Conversation Type
Stacked bar chart. X-axis = conversation_type (from conversations_clean). Y-axis = proportion. Each stack segment = function category. Normalized to 100% per conversation type.

### Figure 3: Function Distribution by Model Era
Same as Figure 2 but x-axis = model_era. Include era boundary lines if applicable.

### Figure 4: Function Distribution Over Time (HTML)
Interactive Plotly stacked area chart. X-axis = year_month. Y-axis = proportion. Each area = function category. Hover shows month, function, percentage. Include model era boundary lines.

### Figure 5: Confidence Distribution
Histogram of confidence scores. Separate colors for high-confidence (>0.8), medium (0.5-0.8), and low (<0.5). Annotate with mean, median, and % in each confidence band.

### Figure 6: Function by Time of Day
Grouped or stacked bar chart. X-axis = time_of_day (morning, afternoon, evening, night). Y-axis = proportion. Each segment = function category.

### Figure 7: Function Predicts Conversation Depth (2x2 grid)
Four subplots showing mean turns, mean duration, mean token ratio, mean message count by function category. Include error bars (SEM). Mark significance with * if Kruskal-Wallis p < 0.05.

### Figure 8: Cognitive Signature Summary Dashboard
Summary panel showing:
- Dominant function and percentage
- Top 3 functions as a pie chart
- Function diversity (Shannon entropy)
- Key insight (e.g., "coding and problem_solving account for 60% of conversations")

---

## Report Structure

Save to `outputs/reports/functional_classification_report.json`:

```json
{
  "module": "functional_classification",
  "module_version": "1.0",
  "generated_at": "ISO timestamp",
  "model": "claude-haiku-4-5",
  "batch_id": "msgbatch_xxxxx",
  "input_data": {
    "conversations_classified": 1500,
    "input_source_summary": 1480,
    "input_source_fallback": 20,
    "classification_errors": 5
  },
  "distribution": {
    "interpersonal_analysis": {"count": 350, "pct": 23.3},
    "emotional_processing": {"count": 180, "pct": 12.0},
    "creative_expression": {"count": 150, "pct": 10.0},
    "career_strategy": {"count": 140, "pct": 9.3},
    ...
  },
  "confidence_stats": {
    "mean": 0.85,
    "median": 0.88,
    "std": 0.12,
    "pct_high": 72.0,
    "pct_medium": 23.0,
    "pct_low": 5.0
  },
  "cross_tabulations": {
    "by_conversation_type": { ... },
    "by_model_era": { ... },
    "by_time_of_day": { ... }
  },
  "statistical_tests": {
    "function_vs_turns": {"H": 45.2, "p": 0.0001, "significant": true},
    "function_vs_duration": { ... },
    ...
  },
  "calibration": {
    "cohens_kappa": 0.82,
    "n_calibration_items": 50,
    "disagreements": 7,
    "note": "or 'calibration set not found — skipped'"
  },
  "cost": {
    "input_tokens": 250000,
    "output_tokens": 15000,
    "total_cost_usd": 0.15
  },
  "cognitive_signature_fragment": {
    "dominant_function": "coding",
    "dominant_pct": 23.3,
    "function_diversity_entropy": 2.1,
    "top_3": ["coding", "learning", "problem_solving"],
    "summary": "..."
  },
  "figures_generated": [ ... ],
  "data_outputs": ["data/processed/functional_classifications.parquet"],
  "warnings": [ ... ]
}
```

---

## Validation Checklist

The script should print a validation checklist at the end (same pattern as scripts 08-10):

1. `functional_classifications.parquet` exists with correct columns
2. Row count matches analysable conversations (within 1%)
3. All conversation_ids exist in conversations_clean
4. No duplicate conversation_ids
5. All function_primary values are from the valid 12-label set
6. Confidence scores are in [0.0, 1.0]
7. Classification errors < 1%
8. Report JSON exists with all required keys
9. All 8 figures exist
10. All PNGs >= 10KB
11. HTML figures are self-contained
12. No NaN/Infinity in report JSON
13. Cognitive signature fragment summary is non-empty

---

## Script Conventions (Match Existing Codebase)

Follow the patterns established in scripts 08-12:

- **File path:** `subtext/scripts/13_functional_classify.py`
- **Path setup:** Use `PROJECT_ROOT = Path(__file__).resolve().parent.parent` pattern
- **Config:** Load from `config/quality_config.json` for timezone, random seed
- **Data loading:** Read from `data/processed/conversations_clean.parquet` and `messages_clean.parquet`
- **Filter:** Only classify `is_analysable == True` conversations
- **Style constants:** Same color palette as other scripts (`COLOR_PRIMARY = "#2E75B6"`, `COLOR_ACCENT = "#C55A11"`, etc.)
- **DPI:** 150 for all static figures
- **Plotly:** Include plotlyjs in HTML files
- **JSON serialization:** Use the `clean()` and `clean_dict()` helper functions to handle numpy types and NaN/Inf
- **UTF-8 stdout:** Include the `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` pattern
- **Progress:** Use tqdm for loops
- **Error handling:** Wrap major sections in try/except, log errors to report["warnings"]
- **Print progress:** Print step-by-step progress with `===` headers matching the style of scripts 08-10
- **API key:** Read from `ANTHROPIC_API_KEY` environment variable, fail with clear message if missing
- **Batch resume:** Support `--resume-batch-id` argument (same as 12_summarize.py)
- **Cost tracking:** Track input/output tokens per request, compute total cost in report

---

## Command Line Interface

```
python scripts/13_functional_classify.py

# Resume a previously submitted batch:
python scripts/13_functional_classify.py --resume-batch-id msgbatch_xxxxx

# Skip calibration check (run full batch without calibration set):
python scripts/13_functional_classify.py --skip-calibration

# Dry run (build requests, don't submit):
python scripts/13_functional_classify.py --dry-run
```

---

## Cost Estimate

- ~1,500 conversations × ~100 input tokens per summary = ~150K input tokens
- ~1,500 conversations × ~20 output tokens per response = ~30K output tokens
- Haiku Batch API pricing: $0.40/MTok input, $2.00/MTok output (with 50% batch discount)
- Estimated cost: **< $0.15**
