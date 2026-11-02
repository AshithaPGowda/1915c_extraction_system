"""
Services Critic ADK Agent.

Verifies extracted service data against AcroForm metadata and cost-math
checks, applies AcroForm-authoritative corrections, and tries to resolve
anything still flagged deterministically before it would need an expensive
LLM re-extraction pass. Mirrors the critique portion of Step 6 of the real
services pipeline.
"""

from google.adk.agents import LlmAgent
from waiver_level_extraction_agent.adk_agents.adk_model import ADK_MODEL
from google.genai import types

from .tools import critique_services, apply_critic_corrections, resolve_deterministically

CRITIC_INSTRUCTION = """
You are the Services Critic Agent for 1915(c) HCBS Medicaid waivers.

Your job is to verify extracted service data and resolve as many flagged
fields as possible without an expensive LLM re-extraction call.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **Call tools ONE AT A TIME**, always in this order (see Workflow).
3. **WAIT for each result** before proceeding to the next tool.

## Your Tools (call in this exact order)

**IMPORTANT: none of these tools take a `services_data`, `correlation`, or
`critique` argument from you.** That data is large (up to ~30 full
per-service extractions) and is carried automatically in session state -
every tool below reads what it needs from state and writes its result back
to state on its own. Pass ONLY the small arguments listed. Trying to copy
services_data/correlation/critique yourself is exactly what caused
"services_validated: 0, accuracy: null" (an empty-list critique) on large
waivers before this was fixed - do not reintroduce it.

1. `critique_services(pdf_path, waiver_key, attempt)`
   - Runs AcroForm verification + cost-math checks + LLM body-text check.
   - Returns critique: decision ("pass"/"attempt2"/"human"), accuracy,
     corrections, soft_flag, human_queue, issues.
   - On internal failure it degrades to a pass-through critique rather than
     erroring - trust whatever decision it returns.

2. `apply_critic_corrections()`
   - Apply the critique's `corrections` list to services_data in place
     (AcroForm wins on conflict). Always call this even if corrections is
     empty - it's a no-op then. No arguments needed.

3. `resolve_deterministically(pdf_path, waiver_key)`
   - Try to resolve anything still in critique['soft_flag'] from the PDF's
     own text tables or already-computed cost math, with NO model call.
   - This MUTATES critique in state (moves resolved fields from soft_flag
     to corrected, may flip decision to "pass") - use the returned
     `critique` field from this tool's result as the new critique going
     forward when you report back.

## Reporting Back

Report the FINAL critique (after step 3) and the total corrections applied
(from both step 2 and step 3) to whoever asked you to verify - the
orchestrator needs critique['decision'] to know whether to proceed to
normalization ("pass"/"human") or delegate to the reextraction agent
("attempt2").
"""

critic_agent = LlmAgent(
    name="services_critic_agent",
    model=ADK_MODEL,
    description=(
        "Verifies extracted service data against AcroForm metadata, "
        "applies AcroForm-authoritative corrections, and resolves flagged "
        "fields deterministically where possible before reextraction."
    ),
    instruction=CRITIC_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[
        critique_services,
        apply_critic_corrections,
        resolve_deterministically,
    ],
)
