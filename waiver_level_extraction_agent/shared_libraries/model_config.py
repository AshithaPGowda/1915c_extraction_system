"""
Single place resolving which Gemini model + Vertex location a call should
use, for the model-comparison experiment (gemini-2.5-flash vs
gemini-3.1-pro-preview vs gemini-3.7-flash).

Every hardcoded `model="gemini-2.5-flash"` literal in the codebase should
be replaced with `get_model_name()` (or an explicit `model=` param that
defaults to it), and every `_get_client()` should resolve its location via
`get_location(model)` rather than reading LOCATION directly.

Confirmed directly against the live API (2026-09-14, project waiver-496913,
not just docs): gemini-3.1-pro-preview AND gemini-3.7-flash both 404 on
every regional endpoint tried (us-central1, us-east5, us-east1, us-west1,
europe-west4) and both succeed on "global" - this project has full access
to both models, the earlier "not found or your project does not have
access to it" message was Vertex's (misleading) 404 text for "wrong
endpoint for this model," not an entitlement problem.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.5-flash"

# Model IDs confirmed live against the API - these 404 on every regional
# endpoint and only work on "global". Extend this set (and re-verify
# empirically, not just from docs - see module docstring) if another
# global-only model gets added to the comparison.
_GLOBAL_ONLY_MODELS = {"gemini-3.1-pro-preview", "gemini-3.7-flash"}


def get_model_name() -> str:
    """The default model for this process, from GEMINI_MODEL - unchanged
    behavior (gemini-2.5-flash) unless explicitly overridden."""
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def get_location(model: str | None = None) -> str:
    """Resolve the Vertex location for a given model.

    A global-only model ALWAYS resolves to "global" - this is a hard API
    requirement (confirmed empirically), not a preference, so it overrides
    even an explicit LOCATION env var; there is no location other than
    "global" that ever works for these models, so respecting a generic
    ambient LOCATION=us-central1 (set for the unrelated gemini-2.5-flash
    workflow) would just guarantee a 404. For every other model, LOCATION
    wins if set, else the historical "us-central1" default."""
    model = model or get_model_name()
    if model in _GLOBAL_ONLY_MODELS:
        return "global"
    return os.getenv("LOCATION", "us-central1")
