"""
Centralized path configuration for SUBTEXT pipeline scripts.

Usage:
    from paths import PROCESSED, INTERIM, PROMPTS, FIGURES, REPORTS, MODELS, CONFIG
    # or for specific files:
    from paths import MSGS_PATH, CONV_PATH, MSGS_CLEAN, CONV_CLEAN
"""

import os
from pathlib import Path

# Project root: subtext/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Also expose as string for scripts that use os.path
BASE = str(PROJECT_ROOT)

# ── Core directories ────────────────────────────────────────────────────────
DATA       = PROJECT_ROOT / "data"
PROCESSED  = DATA / "processed"
INTERIM    = DATA / "interim"
RAW        = DATA / "raw"

PROMPTS    = PROJECT_ROOT / "prompts"
CONFIG_DIR = PROJECT_ROOT / "config"
TAXONOMIES = PROJECT_ROOT / "taxonomies"

OUTPUTS    = PROJECT_ROOT / "outputs"
FIGURES    = OUTPUTS / "figures"
REPORTS    = OUTPUTS / "reports"
MODELS     = OUTPUTS / "models"

# ── Commonly used data files ────────────────────────────────────────────────
MSGS_PATH       = PROCESSED / "messages.parquet"
CONV_PATH       = PROCESSED / "conversations.parquet"
MSGS_CLEAN      = PROCESSED / "messages_clean.parquet"
CONV_CLEAN      = PROCESSED / "conversations_clean.parquet"
MSGS_ENRICHED   = PROCESSED / "messages_enriched.parquet"
CONV_ENRICHED   = PROCESSED / "conversations_enriched.parquet"
SUMMARIES_PATH  = PROCESSED / "conversation_summaries.parquet"
EMBEDDINGS_PATH = PROCESSED / "conversation_embeddings.parquet"

# ── Config files ────────────────────────────────────────────────────────────
QUALITY_CONFIG  = CONFIG_DIR / "quality_config.json"
KNOWN_ENTITIES  = CONFIG_DIR / "known_entities.json"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [PROCESSED, INTERIM, RAW, FIGURES, REPORTS, MODELS]:
        d.mkdir(parents=True, exist_ok=True)
