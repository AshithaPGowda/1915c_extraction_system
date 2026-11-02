"""
Normalisation ADK Agent.

Normalises the final extraction, builds the evidence chain, and saves the
result (overview.json, SQLite, Chroma). Everything is read from ADK session
state.
"""

from google.adk.agents import LlmAgent
from google.genai import types

from .adk_model import ADK_MODEL

from .tools import finalize_waiver

NORMALISATION_INSTRUCTION = """
You are the Normalisation Agent for 1915(c) HCBS Medicaid waiver extractions.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **WAIT for the tool result** before reporting.
3. **Pass ONLY the waiver_key.** The final extraction and critiques are read
   from session state - never pass them yourself.

## Your Tool

`finalize_waiver(waiver_key)` - normalises values, builds the evidence chain,
saves overview.json plus SQLite/Chroma, and returns the final result
(status, accuracy, attempts_used, fields_extracted, output_path, human-queue
fields).

## Workflow

1. Call `finalize_waiver(waiver_key)`.
2. Report the returned result fields.
3. **ALWAYS call `transfer_to_agent("waiver_level_extraction_agent")`
   immediately after reporting, even on error.**
"""

normalisation_agent = LlmAgent(
    name="normalisation_agent",
    model=ADK_MODEL,
    description=(
        "Normalises the final extraction, builds the evidence chain, and "
        "saves the result to overview.json, SQLite and Chroma."
    ),
    instruction=NORMALISATION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[finalize_waiver],
)
