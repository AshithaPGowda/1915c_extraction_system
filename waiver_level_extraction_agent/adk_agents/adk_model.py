"""
Model selection for the waiver-level ADK app.

The ADK app runs on Gemini 3.7 Flash by default. Setting GEMINI_MODEL in this
process (before the core sub_agents read it via shared_libraries.model_config)
also moves the critic / re-extraction / normalisation calls onto the same
model; CLI runs (run_pipeline.py) are separate processes and keep their own
default. Override with GEMINI_MODEL=<model> when launching `adk web`.

3.7 is global-endpoint-only on Vertex, so GOOGLE_CLOUD_LOCATION (used by ADK's
own LlmAgent client) is set to match get_location(model).
"""

import os

from ..shared_libraries.model_config import get_model_name, get_location

os.environ.setdefault("GEMINI_MODEL", "gemini-3.7-flash")
ADK_MODEL = get_model_name()
os.environ["GOOGLE_CLOUD_LOCATION"] = get_location(ADK_MODEL)
