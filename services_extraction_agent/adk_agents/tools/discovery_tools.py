"""
Discovery/Prepare Tools - orchestrator-level, Steps 0-4 of the real
services pipeline (waiver_level_extraction_agent/core/services_pipeline.py::
run_services_extraction). Thin wrappers only - no logic duplicated.
"""

from waiver_level_extraction_agent.parsers.services_field_mapping import (
    extract_services_acroform,
    correlate_services,
    extract_service_taxonomies,
)
from waiver_level_extraction_agent.core.services_pipeline import (
    _resolve_service_list,
    _format_file_name,
)
from waiver_level_extraction_agent.tools.services_tools import (
    get_waiver_metadata_from_overview,
)
from waiver_level_extraction_agent.sub_agents.services_agent import (
    get_checkpointed_services,
)
from waiver_level_extraction_agent.shared_libraries.file_utils import (
    waiver_key_from_filename,
)
from waiver_level_extraction_agent.storage.services_store import (
    list_waiver_keys_with_services,
)

from .constants import PDF_DIR, PROJECT_ROOT

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None

from waiver_level_extraction_agent.shared_libraries.paths import KNOWN_BAD_SOURCES_PATH


def _load_known_bad_keys() -> set:
    """Parse data/known_bad_sources.txt into a set of excluded waiver keys.

    Format is one entry per line: `WAIVER_KEY  # reason`, with full-line
    `#`-comments and blank lines ignored. Same file the manual CLI batch
    workflow checks before picking waivers (see
    [[services-extraction-workflow]] memory) - this tool reads the exact
    same source of truth rather than duplicating the exclusion list.
    """
    if not KNOWN_BAD_SOURCES_PATH.exists():
        return set()

    bad_keys = set()
    for line in KNOWN_BAD_SOURCES_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split("#", 1)[0].strip()
        if key:
            bad_keys.add(key)
    return bad_keys


def list_pending_waivers(limit: int = 20) -> dict:
    """Step 0: Find the next waiver PDFs to process, for batch/queue mode.

    Mirrors the manual selection process the CLI workflow has used all
    along: every PDF basename in PDF_DIR (pdfs_267_latest), alphabetically,
    minus waiver keys that already have rows in the `services` table, minus
    keys listed in data/known_bad_sources.txt (misfiled/unusable source
    PDFs). Use this whenever the caller asks to process "the next waiver(s)"
    or "continue the batch" without naming a specific waiver_key - do not
    guess a waiver_key yourself, and do not ask the user to supply one when
    this tool can determine it.

    Args:
        limit: Max number of pending waiver keys to return (default 20,
            matching the batch size the manual workflow settled on).

    Returns:
        dict with status, total_pdfs, already_done_count,
        known_bad_count, pending_count, and next_waivers (the next `limit`
        keys, alphabetical, ready to feed one at a time into Step 1
        (get_pdf_path) of the normal pipeline).
    """
    if not PDF_DIR.exists():
        return {
            "status": "error",
            "error": f"PDF directory not found: {PDF_DIR}",
        }

    all_keys = sorted({
        waiver_key_from_filename(p.name)
        for p in PDF_DIR.glob("*.pdf")
        if waiver_key_from_filename(p.name)
    })

    already_done = set(list_waiver_keys_with_services())
    known_bad = _load_known_bad_keys()

    pending = [k for k in all_keys if k not in already_done and k not in known_bad]

    return {
        "status": "success",
        "total_pdfs": len(all_keys),
        "already_done_count": len(already_done & set(all_keys)),
        "known_bad_count": len(known_bad & set(all_keys)),
        "pending_count": len(pending),
        "next_waivers": pending[:limit],
    }


def discover_service_list(
    waiver_key: str,
    pdf_path: str,
    standalone: bool = False,
    resume: bool = True,
    tool_context: "ToolContext" = None,
) -> dict:
    """Step 1+2 of the real pipeline: AcroForm pre-extraction, then resolve
    the service list (explicit list > overview.json > LLM discovery,
    cross-checked against the AcroForm service-block count).

    Args:
        waiver_key: The waiver identifier
        pdf_path: Path to the PDF
        standalone: If True, discover services from the PDF via LLM instead
            of using overview.json's _discovered_services
        resume: Default True - the same skip-what's-already-checkpointed
            behavior as the CLI's `--resume` flag. If this waiver has a
            services_attempt1.jsonl checkpoint from an earlier run (ADK or
            CLI), successfully-extracted services are pre-loaded into
            session state (state["services_data"]) instead of being
            re-extracted, and only the genuinely remaining ones are
            returned in `services_remaining` for the extraction loop. Set
            False only to force a full fresh extraction of every service
            regardless of any existing checkpoint.

    Returns:
        dict with status, services_list (the FULL list - use this for
        len(services_list) / services_count in later steps),
        services_remaining (the subset to actually delegate extraction
        for - already-checkpointed ones are skipped),
        already_done_count, acroform_has_acroform, acroform_service_count,
        and appendix_hints. The full AcroForm payload is NOT returned here
        - it's stashed in session state (tool_context.state["acroform"])
        for correlate_services_and_taxonomy to pick up on its own, so the
        model never has to copy this large nested structure back as a
        function-call argument (doing so reliably triggers
        MALFORMED_FUNCTION_CALL for waivers with many services).
    """
    try:
        acroform = extract_services_acroform(pdf_path)
    except Exception as e:
        acroform = {"has_acroform": False, "service_count": 0, "services": {},
                    "self_direction": [], "info_assistance": [], "capitation": {}}

    services_list, appendix_hints = _resolve_service_list(
        pdf_path=pdf_path,
        waiver_key=waiver_key,
        discovered_services=None,
        standalone=standalone,
        acroform=acroform,
    )

    if not services_list:
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": "No services found in waiver",
        }

    already_done = get_checkpointed_services(waiver_key) if resume else {}
    services_remaining = [s for s in services_list if s not in already_done]

    if tool_context is not None:
        tool_context.state["acroform"] = acroform
        tool_context.state["services_list"] = services_list
        tool_context.state["pdf_path"] = pdf_path
        if already_done:
            tool_context.state["services_data"] = list(already_done.values())

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "services_list": services_list,
        "services_remaining": services_remaining,
        "services_count": len(services_list),
        "already_done_count": len(already_done),
        "acroform_has_acroform": acroform.get("has_acroform", False),
        "acroform_service_count": acroform.get("service_count", 0),
        "appendix_hints": appendix_hints,
    }


def correlate_services_and_taxonomy(
    services_list: list,
    pdf_path: str,
    tool_context: "ToolContext" = None,
) -> dict:
    """Step 3 of the real pipeline: correlate AcroForm positions to service
    names, then deterministically parse the HCBS taxonomy block for each
    service from the C-1/C-3 text (no LLM call).

    The AcroForm payload is read from session state (set by
    discover_service_list) rather than as an argument - do not pass it,
    the model is never given it back to copy.

    Returns:
        dict with status, confidence, count_match, hints, taxonomies
    """
    acroform = (tool_context.state.get("acroform") if tool_context is not None else None) or {}
    correlation = correlate_services(services_list, pdf_path, acroform)

    try:
        correlation["taxonomies"] = extract_service_taxonomies(pdf_path, services_list)
    except Exception:
        correlation["taxonomies"] = {}

    if tool_context is not None:
        tool_context.state["correlation"] = correlation

    return {
        "status": "success",
        "confidence": correlation.get("confidence"),
        "count_match": correlation.get("count_match"),
    }


def get_waiver_metadata(waiver_key: str) -> dict:
    """Step 4 of the real pipeline: get state/waiver_title from the
    waiver's overview.json, falling back to the state-code prefix of the
    waiver key if no overview exists yet.

    Returns:
        dict with state, waiver_title, file_name (dotted waiver key)
    """
    try:
        metadata = get_waiver_metadata_from_overview(waiver_key)
        state = metadata.get("state", "Unknown")
        waiver_title = metadata.get("waiver_title", "Unknown")
    except FileNotFoundError:
        state = waiver_key[:2] if len(waiver_key) >= 2 else "Unknown"
        waiver_title = "Unknown"

    return {
        "status": "success",
        "state": state,
        "waiver_title": waiver_title,
        "file_name": _format_file_name(waiver_key),
    }
