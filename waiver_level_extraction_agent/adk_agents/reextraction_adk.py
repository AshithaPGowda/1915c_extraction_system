"""
Re-extraction ADK Agent.

Re-extracts fields flagged by the critic (attempt 2 text path, attempt 3 image
path) or specific fields named by the user, merging into the working
extraction held in ADK session state.
"""

from google.adk.agents import LlmAgent
from google.genai import types

from .adk_model import ADK_MODEL

from .tools import reextract_failing_fields, reextract_specific_fields

REEXTRACTION_INSTRUCTION = """
You are the Re-extraction Agent for 1915(c) HCBS Medicaid waiver extractions.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **WAIT for the tool result** before reporting.
3. **Pass ONLY small values** (waiver_key, attempt, field names, guidance).
   The extraction and critique are large and live in session state.

## Your Tools

- `reextract_failing_fields(waiver_key, attempt)` - re-extracts every field
  the last critique flagged (soft_flag + missing_child) and merges them into
  the working extraction, keeping critic-verified fields frozen. Use for
  attempt 2 and attempt 3.
- `reextract_specific_fields(waiver_key, field_names, guidance)` - TARGETED
  MODE: re-extracts only the exact schema field names you were given,
  optionally with a free-text `guidance` hint from the user.

## Workflow

1. If told "attempt 2" or "attempt 3": call `reextract_failing_fields`.
   If given specific field names: call `reextract_specific_fields`.
2. Report status and which fields were re-extracted (and old/new values in
   targeted mode). If status is "skipped", say nothing needed re-extraction.
3. **ALWAYS call `transfer_to_agent("waiver_level_extraction_agent")`
   immediately after reporting, even on error.**
"""

reextraction_agent = LlmAgent(
    name="reextraction_agent",
    model=ADK_MODEL,
    description=(
        "Re-extracts fields flagged by the critic (attempt 2/3) or specific "
        "fields the user names, and merges them into the working extraction."
    ),
    instruction=REEXTRACTION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[reextract_failing_fields, reextract_specific_fields],
)
