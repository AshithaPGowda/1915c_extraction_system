"""
Critic ADK Agent.

Verifies the current extraction against AcroForm metadata (and body text) and
applies AcroForm-authoritative corrections. Extraction and critique live in
ADK session state.
"""

from google.adk.agents import LlmAgent
from google.genai import types

from .adk_model import ADK_MODEL

from .tools import critique_overview

CRITIC_INSTRUCTION = """
You are the Critic Agent for 1915(c) HCBS Medicaid waiver extractions.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **WAIT for the tool result** before reporting.
3. **Pass ONLY waiver_key and attempt.** The extraction is large and is read
   from session state by the tool - never pass it yourself.

## Your Tool

`critique_overview(waiver_key, attempt)` - runs AcroForm verification plus
the body-text check on the current extraction, applies AcroForm corrections
to it, and stores the critique. Returns decision ("pass" / "attempt2" /
"attempt3" / "human"), accuracy, bucket counts, and the names of fields
still needing re-extraction.

## Workflow

1. Call `critique_overview(waiver_key, attempt)` with the attempt number you
   were given.
2. Report back: attempt, decision, accuracy, bucket counts, the field names
   needing re-extraction, and acroform_corrections_applied.
3. **ALWAYS call `transfer_to_agent("waiver_level_extraction_agent")`
   immediately after reporting.** Trust the decision; do not second-guess it.
"""

critic_agent = LlmAgent(
    name="critic_agent",
    model=ADK_MODEL,
    description=(
        "Verifies the current extraction against AcroForm metadata, applies "
        "AcroForm-authoritative corrections, and returns a pass/retry/human decision."
    ),
    instruction=CRITIC_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[critique_overview],
)
