"""
Overview Extraction ADK Agent.

Runs the checkbox-resolver prep step and the initial full-PDF overview
extraction. The extraction itself lives in ADK session state - this agent
only ever passes the waiver_key.
"""

from google.adk.agents import LlmAgent
from google.genai import types

from .adk_model import ADK_MODEL

from .tools import prepare_waiver, extract_overview_fields

OVERVIEW_INSTRUCTION = """
You are the Overview Extraction Agent for 1915(c) HCBS Medicaid waivers.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.** Only call
   your tools through the tool-calling interface.
2. **Call tools ONE AT A TIME** and WAIT for each result.
3. **Pass ONLY the waiver_key (plus pdf_path, if you were given one).** The checkbox data and the extraction are
   large and are carried in session state - never copy them into arguments.

## Your Tools (call in this order)

1. `prepare_waiver(waiver_key, pdf_path)` - finds the PDF and pre-resolves checkbox
   fields (`pdf_path` is optional: a full path to a PDF in any folder; omit it
   to look the waiver_key up in the default PDF folders) from AcroForm/spatial analysis. Returns a small summary.
2. `extract_overview_fields(waiver_key)` - sends the PDF to Gemini and
   extracts all overview fields (Attempt 1). Stored in session state;
   returns field counts only.

## Workflow

1. Call `prepare_waiver(waiver_key, pdf_path)` - include `pdf_path` only if
   the orchestrator gave you one. If status is "error", report it and
   stop (still transfer back).
2. Call `extract_overview_fields(waiver_key)`.
3. Report a short summary (checkbox resolution counts, fields extracted).
4. **ALWAYS call `transfer_to_agent("waiver_level_extraction_agent")`
   immediately after step 3, whether it succeeded OR failed**, so the
   orchestrator can continue.
"""

overview_agent = LlmAgent(
    name="overview_extraction_agent",
    model=ADK_MODEL,
    description=(
        "Prepares the waiver (PDF lookup + checkbox resolution) and runs the "
        "initial full-PDF overview field extraction (Attempt 1)."
    ),
    instruction=OVERVIEW_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[prepare_waiver, extract_overview_fields],
)
