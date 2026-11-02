"""
Normalisation Tools - for the services_normalisation_agent sub-agent,
wrapping the rest of Step 6 (normalise + taxonomy fallback) and Steps 7-9
of the real services pipeline (flatten to rows, save to SQLite, write JSON
outputs + evidence, export the global CSV, and build the final result).
"""

import os
from datetime import datetime

from waiver_level_extraction_agent.sub_agents.services_normalisation import normalise_services
from waiver_level_extraction_agent.sub_agents.services_agent import (
    infer_taxonomy_fallback,
    flatten_services_to_rows,
)
from waiver_level_extraction_agent.storage.services_store import (
    save_services,
    save_services_run,
    export_all_services_to_csv,
)
from waiver_level_extraction_agent.tools.services_tools import save_services_output
from waiver_level_extraction_agent.shared_libraries.file_utils import write_json_output
from waiver_level_extraction_agent.core.services_pipeline import (
    ALL_SERVICES_CSV,
    _critique_summary,
    _extract_evidence,
)

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None


def normalise_services_data(
    waiver_key: str,
    services_data: list = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Normalise raw extraction output: types, computed component costs.

    services_data defaults to session state (state["services_data"]) if not
    given - do not pass it explicitly, the accumulated extraction results
    from extract_service are already there. The normalised list is written
    back to state; only a small summary is returned to the model, since
    copying the full per-service data back and forth as a call argument is
    what caused critique_services to see an empty list on large waivers.

    Returns dict with status and services_count.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    normalised = normalise_services(services_data or [], waiver_key)
    if tool_context is not None:
        tool_context.state["services_data"] = normalised
    return {"status": "success", "services_count": len(normalised)}


def infer_taxonomy(
    waiver_key: str,
    services_data: list = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Fill in HCBS taxonomy for services whose taxonomy table was empty in
    the PDF, using the official CMS crosswalk (marked as inferred in
    evidence, not read from the document). Mutates services_data in place.

    services_data defaults to session state if not given.

    Returns the number of services classified this way.
    """
    if not services_data and tool_context is not None:
        services_data = tool_context.state.get("services_data", [])
    services_data = services_data or []
    try:
        inferred = infer_taxonomy_fallback(services_data, waiver_key)
    except Exception:
        inferred = 0
    if tool_context is not None:
        tool_context.state["services_data"] = services_data
    return {"status": "success", "inferred_count": inferred}


def finalize_and_save(
    waiver_key: str,
    pdf_path: str,
    state: str,
    waiver_title: str,
    file_name: str,
    services_count: int,
    attempts_used: int,
    corrections_applied: int,
    services_data: list = None,
    critique: dict = None,
    correlation: dict = None,
    tool_context: "ToolContext" = None,
) -> dict:
    """Steps 7-9: flatten to rows, save to SQLite, write JSON output +
    evidence, export the global services CSV, and build the final result -
    exactly the same final_status decision logic as the CLI pipeline.

    Args:
        services_data: The final normalised services list
        critique: The final critique dict (post-reextraction if attempt2 ran)
        services_count: len(services_list) from discover_service_list
        attempts_used: 1 or 2, matching how many critique rounds ran
        corrections_applied: running total from apply_critic_corrections +
            resolve_deterministically calls

    Returns:
        dict matching run_services_extraction()'s real result shape:
        status, output_path, evidence_path, services_extracted, parent_rows,
        child_rows, total_rows, sqlite_rows_saved, accuracy, attempts,
        acroform_confidence, corrections_applied, critique_decision,
        soft_flags_remaining, human_queue
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

    flat_rows = flatten_services_to_rows(
        services_data=services_data,
        waiver_key=waiver_key,
        state=state,
        waiver_title=waiver_title,
        file_name=file_name,
    )

    try:
        rows_saved = save_services(waiver_key, flat_rows)
    except Exception:
        rows_saved = 0

    decision = critique.get("decision")
    accuracy = critique.get("accuracy")
    if decision == "pass":
        final_status = "success"
    elif decision == "human":
        final_status = "human_review_required"
    elif accuracy is not None and accuracy >= 0.85:
        final_status = "success"
    else:
        final_status = "validation_failed"

    try:
        save_services_run(
            waiver_key=waiver_key,
            accuracy=accuracy,
            attempts=attempts_used,
            final_status=final_status,
            services_count=services_count,
        )
    except Exception:
        pass

    output_data = {
        "waiver_key": waiver_key,
        "state": state,
        "waiver_title": waiver_title,
        "file_name": file_name,
        "services": services_data,
        "accuracy": accuracy,
        "attempts": attempts_used,
        "final_status": final_status,
        "extraction_timestamp": datetime.utcnow().isoformat(),
    }
    output_path = save_services_output(waiver_key, output_data)

    evidence_data = {
        "waiver_key": waiver_key,
        "generated_at": datetime.utcnow().isoformat(),
        "acroform_correlation": {
            "confidence": correlation.get("confidence"),
            "count_match": correlation.get("count_match"),
            "hints": correlation.get("hints"),
        },
        "services_evidence": _extract_evidence(services_data),
        "critique": _critique_summary(critique),
    }
    evidence_path = write_json_output(waiver_key, "services_evidence.json", evidence_data)

    # No automatic CSV write in ADK mode: the master all_services.csv is only
    # touched on request via write_services_to_csv (merges this waiver's rows).

    parent_count = sum(1 for r in flat_rows if r.get("row_type") == "parent")
    child_count = sum(1 for r in flat_rows if r.get("row_type") == "child")

    return {
        "status": final_status,
        "waiver_key": waiver_key,
        "pdf_path": pdf_path,
        "output_path": output_path,
        "evidence_path": evidence_path,
        "services_extracted": services_count,
        "parent_rows": parent_count,
        "child_rows": child_count,
        "total_rows": len(flat_rows),
        "sqlite_rows_saved": rows_saved,
        "accuracy": accuracy,
        "attempts": attempts_used,
        "acroform_confidence": correlation.get("confidence"),
        "corrections_applied": corrections_applied,
        "critique_decision": decision,
        "soft_flags_remaining": len(critique.get("soft_flag", [])),
        "human_queue": len(critique.get("human_queue", [])),
    }
