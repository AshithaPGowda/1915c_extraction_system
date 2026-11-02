"""
Services Reextraction ADK Agent.

Re-extracts the services the critic flagged and merges results back in.
Only relevant when the critic's decision is "attempt2" - there is no
attempt3 for services (MAX_ATTEMPTS = 2, unlike the overview pipeline's 3).
"""

from google.adk.agents import LlmAgent
from waiver_level_extraction_agent.adk_agents.adk_model import ADK_MODEL
from google.genai import types

from .tools import reextract_flagged_services

REEXTRACTION_INSTRUCTION = """
You are the Services Reextraction Agent for 1915(c) HCBS Medicaid waivers.

Your job is to re-extract services the critic flagged, and merge the
results back into the full services list.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **Only act when told the critic's decision was "attempt2".** There is
   no attempt3 for services extraction.
3. **WAIT for the result** before reporting back.

## Your Tool

- `reextract_flagged_services(pdf_path, waiver_key, state, waiver_title,
  file_name)` - re-extracts every service in critique['soft_flag'] and
  merges into services_data. **Do NOT pass services_data, critique, or
  correlation** - those are large (up to ~30 full per-service extractions)
  and are read/written automatically via session state. Passing your own
  copy is exactly what caused a silent empty-list critique on large
  waivers before this was fixed. `appendix_hints` is optional and small
  enough to pass through if the orchestrator gave it to you, but omitting
  it is fine too (defaults to {}).

## Workflow

1. Call `reextract_flagged_services(pdf_path, waiver_key, state,
   waiver_title, file_name)` with only these small identifying values.
2. Report the returned services_count back - the orchestrator will
   delegate to normalisation_agent's normalise_services_data next before
   critiquing again (attempt 2's critique needs normalised data, same as
   attempt 1). It reads the merged data from state automatically.
"""

reextraction_agent = LlmAgent(
    name="services_reextraction_agent",
    model=ADK_MODEL,
    description=(
        "Re-extracts services flagged by the critic and merges results "
        "back into the full services list. Only fires on attempt2 - "
        "services extraction has no attempt3."
    ),
    instruction=REEXTRACTION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[
        reextract_flagged_services,
    ],
)
