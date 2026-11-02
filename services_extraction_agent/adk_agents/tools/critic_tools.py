"""
Critic Tools - for the services_critic_agent sub-agent, wrapping the
critique portion of Step 6 of the real services pipeline: verify
extractions against AcroForm, apply AcroForm corrections, then try to
resolve anything still flagged deterministically (PDF text tables /
already-computed cost math) before it would go to the reextraction agent.
"""

from waiver_level_extraction_agent.core.services_pipeline import (
    _run_critic_safely,
    _apply_deterministic_resolution,
)
from waiver_level_extraction_agent.sub_agents.services_critic_agent import apply_corrections

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None


def critique_services(
    pdf_path: str,
    waiver_key: str,
    attempt: int = 1,
    services_data: list = None,
    correlation: dict = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Run the services critic (AcroForm verification + cost-math checks +
    LLM body-text check). On critic failure, degrades to a pass-through
    critique rather than raising - same as the real pipeline.

    services_data and correlation default to session state if not given -
    do not pass services_data explicitly. Passing the model's own copy of
    it is what silently produced an empty-list critique (services_validated
    0, accuracy null) on large waivers, since the model has no reliable way
    to carry ~30 full per-service extractions verbatim across this many
    tool calls.

    Returns the full critique dict: decision ("pass"/"attempt2"/"human"),
    accuracy, corrections, soft_flag, human_queue, issues.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    if not correlation and tool_context is not None:
        correlation = tool_context.state.get("correlation", {})

    critique = _run_critic_safely(
        pdf_path=pdf_path,
        services_data=services_data or [],
        waiver_key=waiver_key,
        correlation=correlation or {},
        attempt=attempt,
    )
    if tool_context is not None:
        tool_context.state["critique"] = critique
    return critique


def apply_critic_corrections(
    services_data: list = None,
    corrections: list = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Apply AcroForm-authoritative corrections to the extraction in place
    (AcroForm wins on conflict). Mutates services_data.

    services_data defaults to session state; corrections defaults to the
    last critique's state["critique"]["corrections"] if not given.

    Returns the number of fields corrected.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    if corrections is None and tool_context is not None:
        corrections = (tool_context.state.get("critique") or {}).get("corrections", [])
    services_data = services_data or []

    applied = apply_corrections(services_data, corrections or [])
    if tool_context is not None:
        tool_context.state["services_data"] = services_data
    return {"status": "success", "corrections_applied": applied}


def resolve_deterministically(
    pdf_path: str,
    waiver_key: str,
    services_data: list = None,
    critique: dict = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Before anything flagged goes to the (expensive) LLM re-extraction
    agent, try to resolve it deterministically - from the PDF text tables
    or a value the critic's own cost-math check already computed. Mutates
    `critique` in place: resolved fields move soft_flag -> corrected, and
    accuracy/decision are recomputed. Also mutates services_data via the
    corrections it applies.

    services_data and critique default to session state if not given.

    Returns the number of fields resolved this way, and the mutated
    critique for the caller to use going forward.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    if not critique and tool_context is not None:
        critique = tool_context.state.get("critique", {})
    services_data = services_data or []
    critique = critique or {}

    resolved_count = _apply_deterministic_resolution(
        pdf_path, waiver_key, services_data, critique
    )
    if tool_context is not None:
        tool_context.state["services_data"] = services_data
        tool_context.state["critique"] = critique
    return {
        "status": "success",
        "resolved_count": resolved_count,
        "critique": critique,
    }
