"""Shared fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_embedx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate every test that builds Settings from the developer's real
    # environment (EMBEDX_CONFIG, EMBEDX_* overrides).
    for key in list(os.environ):
        if key.upper().startswith("EMBEDX_"):
            monkeypatch.delenv(key, raising=False)
