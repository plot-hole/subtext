"""Tests for taxonomy.py — loading, validation, and prompt generation."""

import json
import os
import tempfile

import pytest

# Adjust path so taxonomy.py is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taxonomy import (
    load_taxonomy,
    get_labels,
    get_colors,
    get_descriptions,
    build_system_prompt,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "taxonomies")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_taxonomy(tmp, data):
    path = os.path.join(tmp, "tax.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _minimal_taxonomy(tax_type="functional", n_categories=3):
    """Build a minimal valid taxonomy dict."""
    cats = [
        {"label": f"cat_{i}", "description": f"Description {i}", "color": f"#00000{i}"}
        for i in range(n_categories)
    ]
    return {
        "schema_version": "1.0",
        "taxonomy_type": tax_type,
        "taxonomy_id": f"{tax_type}_test",
        "categories": cats,
    }


# ---------------------------------------------------------------------------
# load_taxonomy
# ---------------------------------------------------------------------------

class TestLoadTaxonomy:
    def test_loads_default_functional(self):
        tax = load_taxonomy(os.path.join(FIXTURES_DIR, "default_functional.json"))
        assert tax["taxonomy_type"] == "functional"
        assert len(tax["categories"]) >= 2

    def test_loads_default_emotional(self):
        tax = load_taxonomy(os.path.join(FIXTURES_DIR, "default_emotional.json"))
        assert tax["taxonomy_type"] == "emotional"
        assert len(tax["categories"]) >= 2

    def test_rejects_missing_required_fields(self, tmp_path):
        data = {"schema_version": "1.0", "categories": []}
        path = _write_taxonomy(str(tmp_path), data)
        with pytest.raises(ValueError, match="missing required fields"):
            load_taxonomy(path)

    def test_rejects_too_few_categories(self, tmp_path):
        data = _minimal_taxonomy()
        data["categories"] = [data["categories"][0]]
        path = _write_taxonomy(str(tmp_path), data)
        with pytest.raises(ValueError, match="at least 2 categories"):
            load_taxonomy(path)

    def test_rejects_missing_category_fields(self, tmp_path):
        data = _minimal_taxonomy()
        del data["categories"][0]["description"]
        path = _write_taxonomy(str(tmp_path), data)
        with pytest.raises(ValueError, match="missing fields"):
            load_taxonomy(path)

    def test_rejects_invalid_label(self, tmp_path):
        data = _minimal_taxonomy()
        data["categories"][0]["label"] = "Invalid Label!!"
        path = _write_taxonomy(str(tmp_path), data)
        with pytest.raises(ValueError, match="lowercase_underscore"):
            load_taxonomy(path)


# ---------------------------------------------------------------------------
# get_labels / get_colors / get_descriptions
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_get_labels(self):
        tax = _minimal_taxonomy()
        assert get_labels(tax) == ["cat_0", "cat_1", "cat_2"]

    def test_get_colors(self):
        tax = _minimal_taxonomy()
        colors = get_colors(tax)
        assert colors["cat_0"] == "#000000"
        assert len(colors) == 3

    def test_get_colors_default(self):
        tax = _minimal_taxonomy()
        del tax["categories"][0]["color"]
        colors = get_colors(tax)
        assert colors["cat_0"] == "#999999"

    def test_get_descriptions(self):
        tax = _minimal_taxonomy()
        descs = get_descriptions(tax)
        assert descs["cat_1"] == "Description 1"


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_functional_prompt_contains_categories(self):
        tax = _minimal_taxonomy("functional")
        prompt = build_system_prompt(tax)
        assert "cat_0" in prompt
        assert "cat_1" in prompt
        assert "function" in prompt.lower()
        assert "JSON" in prompt

    def test_emotional_prompt_contains_categories(self):
        tax = _minimal_taxonomy("emotional")
        prompt = build_system_prompt(tax)
        assert "cat_0" in prompt
        assert "emotion" in prompt.lower()
        assert "JSON" in prompt

    def test_unknown_type_raises(self):
        tax = _minimal_taxonomy("unknown_type")
        with pytest.raises(ValueError, match="Unknown taxonomy_type"):
            build_system_prompt(tax)

    def test_key_distinctions_included(self):
        tax = _minimal_taxonomy("functional")
        tax["key_distinctions"] = ["A is not B", "C overlaps with D"]
        prompt = build_system_prompt(tax)
        assert "A is not B" in prompt
        assert "C overlaps with D" in prompt

    def test_prompt_from_real_functional(self):
        tax = load_taxonomy(os.path.join(FIXTURES_DIR, "default_functional.json"))
        prompt = build_system_prompt(tax)
        assert "interpersonal_analysis" in prompt
        assert len(prompt) > 200

    def test_prompt_from_real_emotional(self):
        tax = load_taxonomy(os.path.join(FIXTURES_DIR, "default_emotional.json"))
        prompt = build_system_prompt(tax)
        assert len(prompt) > 200
