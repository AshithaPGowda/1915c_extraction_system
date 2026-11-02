"""
Reextraction Tools - for the services_reextraction_agent sub-agent,
wrapping Step 6's re-extraction path of the real services pipeline: only
fires when the critic's decision is "attempt2" (MAX_ATTEMPTS = 2 for
services - there is no attempt3 here, unlike the overview pipeline).
"""

from waiver_level_extraction_agent.sub_agents.services_reextraction_agent import (
    run_services_reextraction,
)

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None


def reextract_flagged_services(
    pdf_path: str,
    waiver_key: str,
    state: str,
    waiver_title: str,
    file_name: str,
    services_data: list = None,
    critique: dict = None,
    correlation: dict = None,
    appendix_hints: dict | None = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Re-extract the services the critic flagged (critique['soft_flag'])
    and merge the results back into services_data.

    Args:
        pdf_path: Path to the PDF
        waiver_key: The waiver identifier
        state, waiver_title, file_name: from get_waiver_metadata
        services_data, critique, correlation: default to session state if
            not given - do not pass these explicitly, they're already
            being carried automatically since discover/critique steps.
        appendix_hints: from discover_service_list (small, fine to pass or
            omit - defaults to {} if omitted)

    Returns:
        dict with status and services_count. The merged services_data is
        written to session state (not returned in full) - the orchestrator
        should delegate to services_normalisation_agent next, which reads
        it from state on its own.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    if not critique and tool_context is not None:
        critique = tool_context.state.get("critique", {})
    if not correlation and tool_context is not None:
        correlation = tool_context.state.get("correlation", {})
    services_data = services_data or []
    critique = critique or {}
    correlation = correlation or {}

    try:
        merged = run_services_reextraction(
            pdf_path=pdf_path,
            waiver_key=waiver_key,
            services_data=services_data,
            critique=critique,
            correlation=correlation,
            state=state,
            waiver_title=waiver_title,
            file_name=file_name,
            appendix_hints=appendix_hints or {},
        )
        if tool_context is not None:
            tool_context.state["services_data"] = merged
        return {"status": "success", "services_count": len(merged)}
    except Exception as e:
        return {"status": "error", "error": str(e), "services_count": len(services_data)}
