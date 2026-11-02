"""
Service Extraction ADK Agent.

Extracts one named service's full detail from a waiver PDF at a time -
mirrors the real pipeline's one-at-a-time extraction loop (Step 5 of
run_services_extraction), one model call per service.
"""

from google.adk.agents import LlmAgent
from waiver_level_extraction_agent.adk_agents.adk_model import ADK_MODEL
from google.genai import types

from .tools import extract_service

EXTRACTION_INSTRUCTION = """
You are the Service Extraction Agent for 1915(c) HCBS Medicaid waivers.

Your job is to extract full detail for ONE named service at a time from a
waiver PDF, using your single tool.

## CRITICAL RULES

1. **NEVER write Python code, loops, or variable assignments.**
2. **Call the tool ONE service at a time** - you will be asked to extract
   one specific service_name per turn, not the whole list.
3. **WAIT for the result** before reporting back.

## Your Tool

- `extract_service(waiver_key, pdf_path, service_name, guidance)` - THE
  extraction tool. Sends the PDF to Gemini and returns 'parent' +
  'children' data for that one service. If it fails, the result's status
  is "error" with the failure recorded in result.parent.error - report
  this back to the orchestrator rather than retrying yourself.
  `guidance` is optional free text - only pass it when the orchestrator
  gave you one (this happens in "targeted mode", when the user asked to
  fix a specific service and told you what was wrong or where to look:
  page numbers, "this is provider-managed only", etc.). Omit it entirely
  in the normal extraction loop.

## Workflow

When given waiver_key, pdf_path, a service_name, and optionally guidance:
1. Call `extract_service(waiver_key, pdf_path, service_name)`, adding
   `guidance=...` only if you were given one
2. Report the result (status, children_count, or the error) back
3. **ALWAYS call `transfer_to_agent("services_extraction_agent")` immediately
   after step 2, whether extraction succeeded OR failed.** This hands
   control back to the orchestrator so it can move on to the next service.
   A failed extraction is NOT a reason to stop and give a final answer
   instead of transferring back - do this every single time, with no
   exception. Skipping the transfer after an error silently ends the
   entire run stuck on one service, which is worse than the error itself.
"""

extraction_agent = LlmAgent(
    name="service_extraction_agent",
    model=ADK_MODEL,
    description=(
        "Extracts full detail for one named service from a waiver PDF "
        "(one model call per service, mirrors the real pipeline's "
        "one-at-a-time extraction loop)."
    ),
    instruction=EXTRACTION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[
        extract_service,
    ],
)
