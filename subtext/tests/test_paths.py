"""Tests for scripts/paths.py — centralized path configuration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from paths import (
    PROJECT_ROOT, BASE, PROCESSED, INTERIM, PROMPTS,
    CONFIG_DIR, TAXONOMIES, MSGS_CLEAN, CONV_CLEAN,
    QUALITY_CONFIG, ensure_dirs,
)


class TestPaths:
    def test_project_root_is_subtext_dir(self):
        assert PROJECT_ROOT.name == "subtext"
        assert PROJECT_ROOT.exists()

    def test_base_matches_project_root(self):
        assert BASE == str(PROJECT_ROOT)

    def test_core_dirs_under_project_root(self):
        assert str(PROCESSED).startswith(str(PROJECT_ROOT))
        assert str(INTERIM).startswith(str(PROJECT_ROOT))
        assert str(PROMPTS).startswith(str(PROJECT_ROOT))

    def test_config_files_exist(self):
        assert QUALITY_CONFIG.exists(), f"Missing {QUALITY_CONFIG}"

    def test_taxonomy_dir_exists(self):
        assert TAXONOMIES.exists()

    def test_ensure_dirs_is_idempotent(self):
        ensure_dirs()
        assert PROCESSED.exists()
        assert INTERIM.exists()
