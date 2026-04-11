"""Tests for scripts/batch_api.py — shared batch API utilities."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from batch_api import check_api_key


class TestCheckApiKey:
    def test_exits_if_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            check_api_key()

    def test_returns_key_if_set(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234")
        key = check_api_key()
        assert key == "sk-ant-test-key-1234"
        captured = capsys.readouterr()
        assert "****" in captured.out
        assert "sk-ant" not in captured.out  # key must NOT be printed
