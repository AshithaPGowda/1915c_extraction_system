"""
Extraction Tools - for the service_extraction_agent sub-agent, wrapping
Step 5 of the real services pipeline: extracting one service at a time.

Uses core.services_pipeline.extract_one_service() directly - the same
function --only-service on the CLI uses to retry a single failing service.
One model call per invocation, appended to the waiver's
services_attempt1.jsonl checkpoint (no critique, no DB save - those are
separate sub-agents' jobs).
"""

from waiver_level_extraction_agent.core.services_pipeline import extract_one_service

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None


def extract_service(
    waiver_key: str,
    pdf_path: str,
    service_name: str,
    guidance: str | None = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Extract a single named service from the PDF (one model call).

    Args:
        waiver_key: The waiver identifier
        pdf_path: Path to the PDF
        service_name: The exact service name to extract (from
            discover_service_list's services_list)
        guidance: Optional free-text steer for THIS attempt only (e.g. "the
            cost table is on page 210-215, not the usual location" or "last
            attempt hallucinated self-direction fields - this service is
            provider-managed only, verify against C-1 before filling that
            field"). Use this for the ADK orchestrator's targeted
            single-service mode (re-extracting one named service on
            request, with user-supplied hints about what went wrong) - omit
            it in the normal full-waiver discovery loop.

    Returns:
        dict with 'parent' and 'children' data for this service, or a
        'parent.error' field if extraction failed for this service. Each
        result is also written into session state (state["services_data"])
        as it's produced - the orchestrator does NOT need to manually
        collect these into a list and pass it forward; every downstream
        tool (normalise_services_data, critique_services, etc.) reads the
        accumulated list from state on its own. Relying on the model to
        carry a growing list of full per-service extractions across many
        tool calls is what silently dropped services_data (empty list
        reaching the critic) on large waivers before this fix. If
        services_data already has an entry for this service_name (e.g.
        pre-loaded by discover_service_list's resume, or a prior attempt
        this same session), this result REPLACES it rather than adding a
        duplicate row - last extraction for a given service always wins.

    Transient failures (504 DEADLINE_EXCEEDED, other API errors) are
    caught here and turned into a 'parent.error' result rather than left
    to propagate - extract_one_service (unlike the CLI batch loop's
    extract_all_services) does not catch these itself, since it's shared
    with the CLI's single-service --only-service retry path where the
    caller wants to see the raw exception. Left uncaught here, ADK's
    function-calling layer turns it into a fatal ServerError event that
    kills the whole agent run instead of letting the orchestrator move on
    to the next service (the isolated-per-service-errors-are-self-
    recovering behavior the CLI pipeline relies on).
    """
    try:
        result = extract_one_service(
            waiver_key=waiver_key,
            pdf_path=pdf_path,
            service_name=service_name,
            guidance=guidance,
        )
    except Exception as e:
        result = {
            "parent": {
                "row_type": "parent",
                "service_name": service_name,
                "error": str(e),
            },
            "children": [],
        }
    parent = result.get("parent", {})

    if tool_context is not None:
        services_data = [
            s for s in tool_context.state.get("services_data", [])
            if s.get("parent", {}).get("service_name") != service_name
        ]
        services_data.append(result)
        tool_context.state["services_data"] = services_data

    return {
        "status": "error" if parent.get("error") else "success",
        "service_name": service_name,
        "children_count": len(result.get("children", [])),
        "result": result,
    }
