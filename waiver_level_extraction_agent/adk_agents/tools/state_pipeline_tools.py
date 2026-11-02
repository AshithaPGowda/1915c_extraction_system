"""
State-driven pipeline tools for the waiver-level ADK agents.

Thin wrappers over the same functions core/pipeline.py::run_overview_extraction
uses (checkbox resolver, overview model call, run_critic, run_reextraction,
merge_extractions, run_normalisation). The large objects - extraction, critique,
checkbox cache - live in ADK session state and are read/written by the tools
themselves, so the LLM only ever passes small identifiers (waiver_key,
attempt, field names). Having the model copy the full AcroForm/extraction
into a function-call argument made Gemini emit a code-interpreter call
('google:python_interpreter') and crash the run - the same failure the
services agent hit and fixed the same way.

State keys: pdf_path, checkbox_cache, extracted (current working extraction),
extracted_attempt1, critiques (list), reextracted_attempt2, corrections_all.
"""

import os
import json
from pathlib import Path

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover - tools stay importable outside ADK
    ToolContext = None

from .pdf_tools import get_pdf_path
from .stage_storage_tools import save_stage_output
from .constants import OUTPUT_DIR

PASS_THRESHOLD = 0.85
_EMPTY_CRITIQUE = {
    "verified_acroform": [], "verified_body": [], "corrected": [],
    "soft_flag": [], "hallucinated": [], "missing_child": [], "human_queue": [],
}


# ADK-only: 3.7 can take longer than the 2-minute default client timeout.
ADK_CLIENT_TIMEOUT_MS = int(os.getenv("ADK_GEMINI_TIMEOUT_MS", "600000"))  # 10 min
_TRANSIENT_MARKERS = ("DEADLINE_EXCEEDED", "504", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429")


def _adk_client(model: str):
    """Vertex client for the ADK extraction call: global endpoint for 3.7 and a
    generous HTTP timeout so a slow 3.7 response is not cut at 120s."""
    from google import genai
    from google.genai import types
    from ...shared_libraries.model_config import get_location

    return genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID"),
        location=get_location(model),
        http_options=types.HttpOptions(timeout=ADK_CLIENT_TIMEOUT_MS),
    )


def _retry_transient(func, retries: int = 2, base_delay: float = 15.0):
    """Retry a model call on transient API errors (504 DEADLINE_EXCEEDED,
    503, 429) with backoff. Non-transient errors are raised immediately."""
    import time
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt >= retries or not any(m in str(e) for m in _TRANSIENT_MARKERS):
                raise
            time.sleep(base_delay * (2 ** attempt))


def _resolve_pdf(waiver_key: str, pdf_path: str = "") -> dict:
    """Locate the PDF (explicit path, ADK_PDF_DIRS, then the default folder)."""
    return get_pdf_path(waiver_key, pdf_path)


def _is_base_field(key: str) -> bool:
    """A real schema field (not _evidence / _page / _confidence / _meta keys)."""
    return not key.startswith("_") and not key.endswith(("_evidence", "_page", "_confidence"))


def _state(tool_context):
    return tool_context.state if tool_context is not None else {}


def _stage(waiver_key: str, stage: str, data: dict):
    try:
        save_stage_output(waiver_key, stage, data)
    except Exception:
        pass  # checkpoint is best-effort; never kill the run for it


def _names(entries: list, limit: int = 25) -> list:
    return [e.get("field") for e in entries[:limit]]


def _breakdown(critique: dict) -> dict:
    return {k: len(critique.get(k, [])) for k in _EMPTY_CRITIQUE}


def prepare_waiver(waiver_key: str, pdf_path: str = "", tool_context: "ToolContext" = None) -> dict:
    """Step 1: locate the PDF and pre-resolve checkbox fields from AcroForm /
    spatial analysis (same as the CLI pipeline's checkbox-resolver step).

    Stores pdf_path and the checkbox cache in session state. Returns only a
    small summary - never the checkbox data itself.

    Args:
        waiver_key: The waiver identifier (e.g. "PA0324R0600"). If only a
            pdf_path is given, the key is derived from the file name.
        pdf_path: Optional full path to a PDF in ANY folder. Omit it to look
            the waiver_key up in the default PDF folders.
    """
    from ...shared_libraries.file_utils import load_layout, log_event, waiver_key_from_filename
    from ...sub_agents.checkbox_resolver import resolve_all_checkboxes, get_resolver_summary
    from ...parsers import AcroFormTree, get_metadata

    if not waiver_key and pdf_path:
        waiver_key = waiver_key_from_filename(pdf_path)
    found = _resolve_pdf(waiver_key, pdf_path)
    if found.get("status") != "found":
        return {"status": "error", "error": found.get("error", "PDF not found")}
    pdf_path = found["pdf_path"]

    try:
        layout = load_layout("layout_waiver_overview")
        checkbox_cache = resolve_all_checkboxes(
            pdf_path=pdf_path, layout=layout, page_hints=None, waiver_key=waiver_key,
        )
        summary = get_resolver_summary(checkbox_cache)
        tree = AcroFormTree(pdf_path)
        meta = get_metadata(pdf_path)
    except Exception as e:
        return {"status": "error", "error": f"prepare failed: {e}"}

    log_event(waiver_key, {
        "agent": "checkbox_resolver", "action": "pre_extraction_complete",
        "checkbox_fields_total": len(checkbox_cache),
        "resolved_acroform": summary["acroform"],
        "resolved_spatial": summary["spatial"],
        "no_resolution": summary["none"],
    })
    _stage(waiver_key, "checkbox_resolution", {
        "pdf_path": pdf_path,
        "summary": summary,
        "checkbox_cache": checkbox_cache,
        "pdf_meta": {"is_cms_form": meta.get("is_cms_form", False), "has_acroform": tree.has_acroform},
    })

    st = _state(tool_context)
    if tool_context is not None:
        st["pdf_path"] = pdf_path
        st["checkbox_cache"] = checkbox_cache
        st["critiques"] = []
        st["corrections_all"] = []
        st["resolver_summary"] = summary
        st["pdf_meta"] = {
            "is_cms_form": meta.get("is_cms_form", False),
            "has_acroform": tree.has_acroform,
        }

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "pdf_path": pdf_path,
        "is_cms_form": meta.get("is_cms_form", False),
        "has_acroform": tree.has_acroform,
        "checkbox_fields": len(checkbox_cache),
        "resolved_acroform": summary["acroform"],
        "resolved_spatial": summary["spatial"],
        "no_resolution": summary["none"],
    }


def extract_overview_fields(waiver_key: str, tool_context: "ToolContext" = None) -> dict:
    """Step 2 (Attempt 1): send the full PDF to Gemini and extract all
    overview fields (same call as the CLI pipeline). The extraction is stored
    in session state; only a small summary is returned.

    Args:
        waiver_key: The waiver identifier
    """
    from google.genai import types
    from ...core.pipeline import (
        _load_overview_prompt, _load_pdf_bytes, _parse_json_response,
        _retry_on_connection_error,
    )
    from ...shared_libraries.model_config import get_model_name
    from ..extraction_prompt_supplement import PROMPT_SUPPLEMENT
    from ...shared_libraries.file_utils import log_event

    st = _state(tool_context)
    pdf_path = st.get("pdf_path") or _resolve_pdf(waiver_key).get("pdf_path")
    if not pdf_path:
        return {"status": "error", "error": "PDF not found - run prepare_waiver first"}

    log_event(waiver_key, {
        "agent": "waiver_overview_agent", "action": "extraction_start",
        "pdf_path": pdf_path, "attempt": 1, "model": get_model_name(),
    })
    try:
        model = get_model_name()
        client = _adk_client(model)  # global endpoint for 3.7, long timeout
        pdf_bytes = _load_pdf_bytes(pdf_path)
        prompt = _load_overview_prompt() + PROMPT_SUPPLEMENT

        def _request():
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_bytes)),
                    types.Part(text=prompt),
                ])],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=65536),
            )
            return response.text.strip()

        raw = _retry_transient(lambda: _retry_on_connection_error(_request))
        extracted = _parse_json_response(raw)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "waiver_overview_agent", "action": "model_error",
            "attempt": 1, "error": str(e),
        })
        return {"status": "error", "waiver_key": waiver_key, "error": str(e)}

    log_event(waiver_key, {
        "agent": "waiver_overview_agent", "action": "extraction_complete",
        "attempt": 1,
        "fields_extracted": len([v for v in extracted.values() if v is not None]),
    })
    _stage(waiver_key, "extraction_attempt_1", extracted)

    if tool_context is not None:
        st["pdf_path"] = pdf_path
        st["extracted"] = extracted
        st["extracted_attempt1"] = dict(extracted)
        st["critiques"] = []
        st["corrections_all"] = []
        st["reextracted_attempt2"] = None

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "fields_extracted": len([k for k, v in extracted.items() if v is not None and _is_base_field(k)]),
        "fields_null": len([k for k, v in extracted.items() if v is None and _is_base_field(k)]),
    }


def critique_overview(waiver_key: str, attempt: int = 1, tool_context: "ToolContext" = None) -> dict:
    """Run the AcroForm critic on the current extraction in session state,
    then apply its AcroForm-authoritative corrections to that extraction.

    Args:
        waiver_key: The waiver identifier
        attempt: 1, 2 or 3 (which critic pass this is)

    Returns:
        Small summary: decision ("pass"/"attempt3"/"attempt2"/"human"),
        accuracy, bucket counts, names of fields still needing work.
    """
    from ...core.pipeline import _apply_acroform_corrections
    from ...sub_agents.critic_agent import run_critic
    from ...shared_libraries.file_utils import log_event

    st = _state(tool_context)
    extracted = st.get("extracted")
    pdf_path = st.get("pdf_path")
    if not extracted or not pdf_path:
        return {"status": "error", "error": "No extraction in state - run extract_overview_fields first"}

    try:
        critique = run_critic(
            pdf_path=pdf_path, extracted_data=extracted, attempt=attempt, waiver_key=waiver_key,
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "critic_agent", "action": "error", "attempt": attempt, "error": str(e),
        })
        critique = {**_EMPTY_CRITIQUE, "accuracy": 0.0, "decision": "attempt2", "error": str(e)}

    corrected, corrections = _apply_acroform_corrections(extracted, critique)
    if corrections:
        log_event(waiver_key, {
            "agent": "pipeline", "action": "acroform_corrections_applied",
            "attempt": attempt, "corrections": corrections,
        })
    breakdown = _breakdown(critique)
    log_event(waiver_key, {
        "agent": "pipeline", "action": f"critique{attempt}_complete",
        "breakdown": breakdown, "accuracy": critique.get("accuracy", 0),
        "decision": critique.get("decision", "unknown"),
    })
    _stage(waiver_key, f"critic_attempt_{attempt}", {
        "critique": critique,
        "extracted": corrected,  # working extraction after AcroForm corrections
        "corrections": corrections,
    })

    if tool_context is not None:
        st["extracted"] = corrected
        st["critiques"] = list(st.get("critiques", [])) + [critique]
        st["corrections_all"] = list(st.get("corrections_all", [])) + corrections

    needs_work = critique.get("soft_flag", []) + critique.get("missing_child", [])
    return {
        "status": "success",
        "attempt": attempt,
        "decision": critique.get("decision"),
        "accuracy": critique.get("accuracy"),
        "breakdown": breakdown,
        "acroform_corrections_applied": len(corrections),
        "fields_needing_reextraction": _names(needs_work),
        "fields_needing_reextraction_count": len(needs_work),
        "human_queue_fields": _names(critique.get("human_queue", [])),
        "error": critique.get("error"),
    }


def reextract_failing_fields(waiver_key: str, attempt: int, tool_context: "ToolContext" = None) -> dict:
    """Re-extract the fields the last critique flagged (soft_flag +
    missing_child), merge them into the working extraction with the
    critic-verified fields frozen. Attempt 2 is text-based, attempt 3 uses
    page images where needed. Reads critique/extraction from session state.

    Args:
        waiver_key: The waiver identifier
        attempt: 2 or 3 (which re-extraction pass this is)
    """
    from ...core.pipeline import _build_reextraction_context
    from ...sub_agents.reextraction_agent import run_reextraction, merge_extractions
    from ...parsers import AcroFormTree
    from ...shared_libraries.file_utils import log_event

    st = _state(tool_context)
    critiques = st.get("critiques") or []
    extracted = st.get("extracted")
    pdf_path = st.get("pdf_path")
    if not critiques or not extracted or not pdf_path:
        return {"status": "error", "error": "No critique/extraction in state - run critique_overview first"}
    critique = critiques[-1]

    needs = critique.get("soft_flag", []) + critique.get("missing_child", [])
    if not needs:
        log_event(waiver_key, {
            "agent": "pipeline", "action": "no_reextraction_needed",
            "reason": "no_soft_flag_or_missing_child",
        })
        return {"status": "skipped", "reason": "nothing flagged for re-extraction"}

    try:
        tree = AcroFormTree(pdf_path)
        context = _build_reextraction_context(critique, extracted, tree, waiver_key)
        if attempt == 2:
            failing = [{"field": f["field_name"], "reason": f["failure_reason"]}
                       for f in context if f["failure_reason"] != "missing_child"]
            low_conf = [{"field": f["field_name"], "reason": f["failure_reason"]}
                        for f in context if f["failure_reason"] == "missing_child"]
        else:
            failing = [{"field": f["field_name"], "reason": f["failure_reason"]} for f in context]
            low_conf = []
        log_event(waiver_key, {
            "agent": "pipeline", "action": "reextraction_fields", "attempt": attempt,
            "fields_count": len(context), "fields": [f["field_name"] for f in context],
        })
        reextracted = run_reextraction(
            waiver_key=waiver_key, pdf_path=pdf_path,
            failing_fields=failing, low_confidence_fields=low_conf, attempt=attempt,
            checkbox_cache=st.get("checkbox_cache"), fields_context=context,
        )
        verified = (critique.get("verified_acroform", []) + critique.get("verified_body", [])
                    + critique.get("corrected", []))
        merged = merge_extractions(base_extraction=extracted, passing_fields=verified, reextracted=reextracted)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "pipeline", "action": "reextraction_error", "attempt": attempt, "error": str(e),
        })
        return {"status": "error", "attempt": attempt, "error": str(e)}

    log_event(waiver_key, {
        "agent": "pipeline", "action": "merge_complete", "attempt": attempt,
        "frozen_fields_count": len(verified), "reextracted_fields_count": len(reextracted),
    })
    _stage(waiver_key, f"extraction_attempt_{attempt}", {
        "reextracted": reextracted,
        "merged": merged,  # working extraction after merge
    })

    if tool_context is not None:
        st["extracted"] = merged
        if attempt == 2:
            st["reextracted_attempt2"] = reextracted

    return {
        "status": "success",
        "attempt": attempt,
        "fields_reextracted": len([k for k in reextracted if not k.startswith("_")
                                   and not k.endswith(("_evidence", "_page"))]),
        "fields_attempted": [f["field_name"] for f in context][:25],
    }


def reextract_specific_fields(
    waiver_key: str,
    field_names: list[str] = None,
    guidance: str = "",
    field_numbers: str = "",
    tool_context: "ToolContext" = None,
) -> dict:
    """Targeted mode: re-extract ONLY the named fields on request (e.g. the
    user asks to redo specific fields), overwriting just those fields in the
    current extraction. If nothing is in session state yet, the last saved
    overview output for this waiver is loaded first.

    Args:
        waiver_key: The waiver identifier
        field_names: Exact schema field names to re-extract
        guidance: Optional free-text hint on what was wrong / where to look
        field_numbers: Alternative to field_names - the review-table row
            numbers, e.g. "33,34" or "40-44"
    """
    from ...sub_agents.reextraction_agent import run_reextraction, merge_extractions
    from ...shared_libraries.file_utils import log_event

    field_names = list(field_names or [])
    if field_numbers:
        order = _canonical_fields()
        nums, bad = _parse_numbers(field_numbers, len(order))
        if bad:
            return {"status": "error", "error": f"Not valid row numbers: {bad}"}
        field_names += [order[n - 1] for n in nums]
    if not field_names:
        return {"status": "error", "error": "No fields given"}

    st = _state(tool_context)
    pdf_path = st.get("pdf_path") or _resolve_pdf(waiver_key).get("pdf_path")
    if not pdf_path:
        return {"status": "error", "error": "PDF not found"}
    extracted = st.get("extracted")
    if not extracted:
        saved = OUTPUT_DIR / waiver_key / "overview.json"
        if saved.exists():
            extracted = json.loads(saved.read_text()).get("extracted_fields", {})
        else:
            extracted = {}

    reason = "user_requested" + (f": {guidance}" if guidance else "")
    failing = [{"field": f, "reason": reason} for f in field_names]
    try:
        reextracted = run_reextraction(
            waiver_key=waiver_key, pdf_path=pdf_path, failing_fields=failing,
            low_confidence_fields=[], attempt=2,
            checkbox_cache=st.get("checkbox_cache"), fields_context=None,
        )
        merged = merge_extractions(base_extraction=extracted, passing_fields=[], reextracted=reextracted)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    log_event(waiver_key, {
        "agent": "pipeline", "action": "targeted_reextraction",
        "fields": field_names, "guidance": guidance,
    })
    if tool_context is not None:
        st["pdf_path"] = pdf_path
        st["extracted"] = merged
    changed = {f: {"old": extracted.get(f), "new": merged.get(f)} for f in field_names}
    return {"status": "success", "fields_requested": field_names, "changes": changed}


def _save_overview_json(waiver_key: str, extracted_data: dict) -> str:
    """Same overview.json as tools.overview_tools.save_overview_output, but
    WITHOUT its automatic append to all_waivers.csv (ADK writes the CSV only
    on request - see write_overview_to_csv)."""
    from ...shared_libraries.file_utils import write_json_output, log_event

    data = {k: v for k, v in extracted_data.items() if not k.endswith("_confidence")}
    flags = {k: v for k, v in extracted_data.items() if k.endswith("_confidence")}
    output = {
        "waiver_key": waiver_key,
        "extracted_fields": data,
        "low_confidence_fields": [k.replace("_confidence", "") for k, v in flags.items() if v == "low"],
        "_discovered_services": data.get("_discovered_services", []),
    }
    path = write_json_output(waiver_key, "overview.json", output)
    log_event(waiver_key, {
        "agent": "waiver_overview_agent", "action": "extraction_complete",
        "fields_extracted": len([v for v in data.values() if v is not None]),
        "fields_null": len([v for v in data.values() if v is None]),
        "low_confidence_count": len(output["low_confidence_fields"]),
        "services_discovered": len(output["_discovered_services"]),
        "output_path": path,
    })
    return path


def write_overview_to_csv(waiver_key: str, dry_run: bool = False, output_csv: str = "") -> dict:
    """Merge this waiver's saved overview.json into the waiver-level results CSV
    (default data/outputs/waiver_level_extraction.csv).

    Replaces ONLY this waiver's existing row (if any) and appends the new one;
    every other row is left untouched, and the previous CSV is backed up to
    data/csv_backups/ first. Never writes the verified masters in data/verified/.
    Run it after finalize_waiver.

    Args:
        waiver_key: The waiver identifier
        dry_run: If True, report what would change without writing anything
        output_csv: Optional different CSV to write instead of the default - a
            bare file name (saved in data/outputs/) or a full path

    Returns:
        dict with rows_replaced, rows_written, rows_total_after, backup_path
    """
    from ...storage.csv_merge import merge_overview_into_csv
    return merge_overview_into_csv(waiver_key, dry_run=dry_run, output_csv=output_csv)


def finalize_waiver(waiver_key: str, tool_context: "ToolContext" = None) -> dict:
    """Final step: normalise the working extraction, build the evidence chain,
    save overview.json, SQLite and Chroma - same as the tail of the CLI
    pipeline. Reads everything from session state.

    Args:
        waiver_key: The waiver identifier
    """
    from ...sub_agents.normalisation_agent import run_normalisation
    from ...storage import save_extraction, save_corrections
    from ...shared_libraries.file_utils import log_event

    st = _state(tool_context)
    final_extracted = st.get("extracted")
    critiques = st.get("critiques") or []
    if not final_extracted:
        return {"status": "error", "error": "No extraction in state"}
    if not critiques:
        return {"status": "error", "error": "Critic has not run - run critique_overview first"}

    latest = critiques[-1]
    try:
        normalised, evidence_path = run_normalisation(
            waiver_key=waiver_key,
            final_extracted=final_extracted,
            attempt1_extracted=st.get("extracted_attempt1"),
            attempt1_critique=critiques[0],
            attempt2_extracted=st.get("reextracted_attempt2"),
            attempt2_critique=critiques[1] if len(critiques) > 1 else None,
        )
        output_path = _save_overview_json(waiver_key, normalised)
    except Exception as e:
        return {"status": "error", "error": f"normalisation/save failed: {e}"}

    accuracy = latest.get("accuracy", 0.0)
    decision = latest.get("decision", "unknown")
    if decision == "pass":
        status = "success"
    elif decision == "human":
        status = "human_review_required"
    elif accuracy >= PASS_THRESHOLD:
        status = "success"
    else:
        status = "validation_failed"

    norm_meta = normalised.get("_normalisation_meta", {})
    result = {
        "status": status,
        "waiver_key": waiver_key,
        "pdf_path": st.get("pdf_path"),
        "output_path": output_path,
        "evidence_path": evidence_path,
        "fields_extracted": len([k for k, v in normalised.items() if v is not None and not k.startswith("_")]),
        "attempt1_accuracy": critiques[0].get("accuracy", 0),
        "final_accuracy": accuracy,
        "attempts_used": len(critiques),
        "final_decision": decision,
        "acroform_corrections_count": len(st.get("corrections_all", [])),
        "norm_errors": norm_meta.get("norm_errors_count", 0),
        "unknown_fields": norm_meta.get("unknown_fields_count", 0),
        "checkbox_resolver_summary": st.get("resolver_summary"),
        "is_cms_form": (st.get("pdf_meta") or {}).get("is_cms_form"),
        "has_acroform": (st.get("pdf_meta") or {}).get("has_acroform"),
    }
    hq = latest.get("human_queue", [])
    if hq:
        result["human_queue_count"] = len(hq)
        result["human_queue_fields"] = _names(hq)

    sqlite_rows = chroma_docs = 0
    try:
        sqlite_rows = save_extraction(waiver_key, "overview", normalised, result)
    except Exception as e:
        log_event(waiver_key, {"agent": "storage", "action": "sqlite_error", "error": str(e)})
    try:
        chroma_docs = save_corrections(waiver_key, critiques, "overview")
    except BaseException as e:  # chroma can raise Rust panics
        log_event(waiver_key, {"agent": "storage", "action": "chroma_error", "error": str(e)})
    result["storage"] = {"sqlite_rows": sqlite_rows, "chroma_documents": chroma_docs}

    log_event(waiver_key, {
        "agent": "pipeline", "action": "extraction_complete", "final_status": status,
        "final_accuracy": accuracy, "attempts_used": len(critiques),
        "human_queue_count": len(hq), "output_path": output_path,
    })
    _stage(waiver_key, "normalization", {"result": result})
    if tool_context is not None:
        st["final_result"] = result
    return result


_ORDER = [
    "checkbox_resolution", "extraction_attempt_1", "critic_attempt_1",
    "extraction_attempt_2", "critic_attempt_2",
    "extraction_attempt_3", "critic_attempt_3", "normalization",
]
_NEXT = {
    "checkbox_resolution": "extract_overview_fields",
    "extraction_attempt_1": "critique_overview(attempt=1)",
    "critic_attempt_1": "reextract_failing_fields(attempt=2), or finalize_waiver if decision was pass/human",
    "extraction_attempt_2": "critique_overview(attempt=2)",
    "critic_attempt_2": "reextract_failing_fields(attempt=3) if decision was attempt3, else finalize_waiver",
    "extraction_attempt_3": "critique_overview(attempt=3)",
    "critic_attempt_3": "finalize_waiver",
    "normalization": "nothing - run is complete",
}


def resume_from_stage(waiver_key: str, stage: str, tool_context: "ToolContext" = None) -> dict:
    """Restart a run from its saved checkpoints: rebuild session state from
    the stage files on disk up to and including `stage` (the last stage you
    want to KEEP), so the pipeline can continue with the step after it.
    Later stage files are ignored and will be overwritten when re-run.

    Use get_pipeline_status(waiver_key) first to see which stages exist.

    Args:
        waiver_key: The waiver identifier
        stage: Last stage to keep - one of checkbox_resolution,
            extraction_attempt_1, critic_attempt_1, extraction_attempt_2,
            critic_attempt_2, extraction_attempt_3, critic_attempt_3

    Returns:
        Small summary incl. `next_step` (which tool/sub-agent to run next).
    """
    from .stage_storage_tools import load_stage_output

    if stage not in _ORDER:
        return {"status": "error", "error": f"Invalid stage '{stage}'. Valid: {_ORDER}"}
    keep = _ORDER[: _ORDER.index(stage) + 1]

    loaded = {}
    for name in keep:
        r = load_stage_output(waiver_key, name)
        if r.get("status") == "success":
            loaded[name] = r["data"]
    missing = [n for n in keep if n not in loaded and n != "normalization"]
    if missing:
        return {"status": "error", "error": f"Missing checkpoints: {missing}. Run get_pipeline_status."}

    cb = loaded["checkbox_resolution"]
    new = {
        "pdf_path": cb.get("pdf_path") or _resolve_pdf(waiver_key).get("pdf_path"),
        "checkbox_cache": cb.get("checkbox_cache", {}),
        "resolver_summary": cb.get("summary"),
        "pdf_meta": cb.get("pdf_meta", {}),
        "critiques": [],
        "corrections_all": [],
    }
    if "extraction_attempt_1" in loaded:
        new["extracted"] = loaded["extraction_attempt_1"]
        new["extracted_attempt1"] = dict(loaded["extraction_attempt_1"])
    for n in (1, 2, 3):
        c = loaded.get(f"critic_attempt_{n}")
        if c:
            new["critiques"].append(c["critique"])
            new["corrections_all"].extend(c.get("corrections", []))
            new["extracted"] = c["extracted"]
        e = loaded.get(f"extraction_attempt_{n}") if n > 1 else None
        if e:
            new["extracted"] = e["merged"]
            if n == 2:
                new["reextracted_attempt2"] = e["reextracted"]

    if tool_context is not None:
        tool_context.state.update(new)

    last = new["critiques"][-1] if new["critiques"] else None
    return {
        "status": "success",
        "waiver_key": waiver_key,
        "resumed_after": stage,
        "stages_loaded": list(loaded),
        "critiques_loaded": len(new["critiques"]),
        "last_decision": last.get("decision") if last else None,
        "last_accuracy": last.get("accuracy") if last else None,
        "has_extraction": "extracted" in new,
        "next_step": _NEXT[stage],
    }


# --------------------------------------------------------------------------
# Human review: show extracted values, then write only what the user approves
# --------------------------------------------------------------------------

def _canonical_fields() -> list:
    """The 163 schema fields in CSV column order. A field's position (1-based)
    is its stable review number, in every session."""
    import importlib.util
    from ...shared_libraries.paths import SCRIPTS_DIR

    spec = importlib.util.spec_from_file_location("_export_to_csv", SCRIPTS_DIR / "export_to_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CANONICAL_FIELDS[1:])  # skip waiver_key


def _parse_numbers(text: str, maximum: int):
    """'1-5, 9' -> ([1,2,3,4,5,9], []). Second value lists tokens that are invalid."""
    nums, bad = [], []
    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip().lstrip("#")
        if not tok:
            continue
        try:
            if "-" in tok:
                a, b = (int(x) for x in tok.split("-", 1))
                rng = list(range(min(a, b), max(a, b) + 1))
            else:
                rng = [int(tok)]
        except ValueError:
            bad.append(tok)
            continue
        for n in rng:
            (nums if 1 <= n <= maximum else bad).append(n if 1 <= n <= maximum else tok)
    return sorted(set(nums)), bad


def _cell(v, width: int) -> str:
    t = "" if v is None else str(v).replace("\n", " ").replace("|", "/").strip()
    return t if len(t) <= width else t[: width - 1] + "..."


def review_extraction(
    waiver_key: str, show: str = "all", offset: int = 0, limit: int = 50,
    search: str = "", numbers: str = "",
) -> dict:
    """Show the extracted values for the user to approve or reject, as a
    numbered markdown table (number, field, value, page, evidence). Print the
    returned `table` text to the user exactly as given.

    Args:
        waiver_key: The waiver identifier
        show: "all" (default - every schema field in order, empty ones marked
            "(empty)"), "filled" (only fields with a value) or "empty"
        offset: Start position in the selected list (for paging; the returned
            `next_hint` says exactly what to ask for next)
        limit: Rows per page (default 50)
        search: Only rows whose field name OR value contains any of these
            comma-separated terms (case-insensitive), e.g. "self-direction",
            "1.F", "spending, budget", "Miller". Works for any waiver that has
            been extracted, in any session. With `search`, show defaults to all
            matches, filled or empty.
        numbers: Only these row numbers, e.g. "10-20, 33"
    """
    from ...shared_libraries.paths import OUTPUTS_DIR, STAGES_DIR

    ov = OUTPUTS_DIR / waiver_key / "overview.json"
    if not ov.exists():
        return {"status": "error", "error": f"No overview.json for {waiver_key} - finalize it first"}
    values = json.loads(ov.read_text()).get("extracted_fields", {})
    evidence = {}
    evp = OUTPUTS_DIR / waiver_key / f"{waiver_key}_evidence.json"
    if evp.exists():
        evidence = json.loads(evp.read_text()).get("fields", {})
    pages = {}
    stage = STAGES_DIR / waiver_key / "extraction_attempt_1.json"
    if stage.exists():
        pages = json.loads(stage.read_text()).get("data", {})

    order = _canonical_fields()
    filled = [i for i, f in enumerate(order, 1) if values.get(f) not in (None, "")]
    empty = [i for i, f in enumerate(order, 1) if values.get(f) in (None, "")]
    pick = {"filled": filled, "empty": empty, "all": list(range(1, len(order) + 1))}.get(show, filled)
    if numbers:
        wanted, bad = _parse_numbers(numbers, len(order))
        if bad:
            return {"status": "error", "error": f"Invalid row numbers: {bad}"}
        pick = [n for n in pick if n in wanted] if show != "filled" or search else [n for n in wanted]
        if show == "filled" and not search:
            pick = wanted  # explicit numbers: show them whether filled or empty
    if search:
        terms = [t.strip().lower() for t in search.split(",") if t.strip()]
        base = pick if show in ("empty", "filled") and show != "filled" else list(range(1, len(order) + 1))
        if numbers:
            base = pick
        pick = [n for n in base
                if any(t in order[n - 1].lower() or t in str(values.get(order[n - 1]) or "").lower()
                       for t in terms)]
    page_rows = pick[offset: offset + limit]

    lines = ["| # | Field | Value | Pg | Evidence |", "|---|---|---|---|---|"]
    for n in page_rows:
        name = order[n - 1]
        ev = evidence.get(name, {}).get("attempt_1", {}).get("evidence", "")
        val = _cell(values.get(name), 70) if values.get(name) not in (None, "") else "*(empty)*"
        lines.append(f"| {n} | {_cell(name, 60)} | {val} | "
                     f"{pages.get(name + '_page', '')} | {_cell(ev, 60)} |")
    nxt = offset + limit if offset + limit < len(pick) else None

    total = len(order)
    what = {"all": "fields", "filled": "filled fields", "empty": "empty fields"}.get(show, "fields")
    if search or numbers:
        what = "matching fields"
    if page_rows:
        first, last = page_rows[0], page_rows[-1]
        summary = (f"Showing {what} #{first}-#{last}: {len(page_rows)} rows"
                   f" (of {len(pick)} {what}). "
                   f"Whole waiver: {total} schema fields = {len(filled)} filled + {len(empty)} empty. "
                   f"The # column is the field's position among all {total} fields, "
                   f"so it is the same in every view.")
    else:
        summary = f"No {what} found. Whole waiver: {total} fields = {len(filled)} filled + {len(empty)} empty."
    if nxt is not None:
        nxt_first = pick[nxt]
        nxt_last = pick[min(nxt + limit, len(pick)) - 1]
        next_hint = (f'More rows: say "show next page" (rows #{nxt_first}-#{nxt_last})'
                     + ("" if not (search or numbers) else " of this search")
                     + ".")
    else:
        next_hint = "That is the last page." if page_rows else ""
    return {
        "status": "success", "waiver_key": waiver_key, "show": show,
        "filled_count": len(filled), "empty_count": len(empty),
        "total_fields": total, "matching_rows": len(pick), "search": search, "numbers": numbers,
        "rows_in_this_page": len(page_rows), "next_offset": nxt,
        "summary": summary, "next_hint": next_hint,
        "table": "\n".join(lines),
    }


def apply_review(
    waiver_key: str,
    reject: str = "",
    approve_only: str = "",
    edits_json: str = "",
    write_csv: bool = True,
    output_csv: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply the user's review decisions, then write the approved row to the
    waiver-level results CSV. Row numbers are the ones in the review table.

    Args:
        waiver_key: The waiver identifier
        reject: Row numbers to reject (left blank), e.g. "5, 9, 20-22"
        approve_only: If given, ONLY these row numbers are kept and every other
            filled field is blanked, e.g. "1-20, 25"
        edits_json: Corrections as a JSON string mapping row number to new
            value, e.g. '{"14": "1", "22": "Case Manager"}'
        write_csv: Write the reviewed row to the CSV (default True)
        output_csv: Optional different CSV (file name -> data/outputs/, or a path)
        dry_run: Report what would change without writing anything
    """
    from ...shared_libraries.paths import OUTPUTS_DIR
    from ...shared_libraries.file_utils import log_event
    from ...storage.csv_merge import merge_overview_into_csv

    ov = OUTPUTS_DIR / waiver_key / "overview.json"
    if not ov.exists():
        return {"status": "error", "error": f"No overview.json for {waiver_key} - finalize it first"}
    doc = json.loads(ov.read_text())
    values = doc.get("extracted_fields", {})
    order = _canonical_fields()

    rej, bad1 = _parse_numbers(reject, len(order))
    keep, bad2 = _parse_numbers(approve_only, len(order))
    try:
        edits = {int(k): v for k, v in json.loads(edits_json).items()} if edits_json else {}
    except (ValueError, AttributeError) as e:
        return {"status": "error", "error": f"edits_json is not valid JSON: {e}"}
    bad_edit = [k for k in edits if not 1 <= k <= len(order)]
    if bad1 or bad2 or bad_edit:
        return {"status": "error", "error": f"Invalid row numbers: {bad1 + bad2 + bad_edit}"}

    def _coerce(v):
        t = str(v).strip()
        if t.lower() in ("null", "none", ""):
            return None
        try:
            return int(t)
        except ValueError:
            try:
                return float(t)
            except ValueError:
                return t

    new_values = dict(values)
    rejected, edited = [], []
    if keep:
        for i, f in enumerate(order, 1):
            if i not in keep and new_values.get(f) not in (None, ""):
                rejected.append(i)
    rejected = sorted(set(rejected) | set(rej))
    for n in rejected:
        new_values.pop(order[n - 1], None)
    for n, v in edits.items():
        new_values[order[n - 1]] = _coerce(v)
        edited.append(n)

    summary = {
        "waiver_key": waiver_key,
        "kept": len([f for f in order if new_values.get(f) not in (None, "")]),
        "rejected": [{"#": n, "field": _cell(order[n - 1], 60)} for n in rejected],
        "edited": [{"#": n, "field": _cell(order[n - 1], 60), "new_value": _cell(new_values[order[n - 1]], 60)}
                   for n in sorted(edited)],
    }
    if dry_run:
        return {"status": "dry_run", **summary}

    backup = ov.with_name("overview.pre_review.json")
    if not backup.exists():
        backup.write_text(ov.read_text())
    doc["extracted_fields"] = new_values
    ov.write_text(json.dumps(doc, indent=2, default=str))
    (ov.parent / "review.json").write_text(json.dumps({
        "reviewed_at": __import__("datetime").datetime.utcnow().isoformat(),
        "reject": rejected, "approve_only": keep, "edits": {str(k): v for k, v in edits.items()},
    }, indent=2))
    log_event(waiver_key, {
        "agent": "human_review", "action": "review_applied",
        "rejected_count": len(rejected), "edited_count": len(edited), "kept": summary["kept"],
    })

    # keep SQLite in step with what was approved (best effort)
    try:
        from ...storage import save_extraction
        from ...shared_libraries.paths import STAGES_DIR
        norm = STAGES_DIR / waiver_key / "normalization.json"
        res = json.loads(norm.read_text()).get("data", {}).get("result", {}) if norm.exists() else {}
        save_extraction(waiver_key, "overview", new_values, {**res, "status": "human_reviewed"})
        summary["db_updated"] = True
    except Exception as e:  # noqa: BLE001
        summary["db_updated"] = f"skipped: {e}"

    summary["status"] = "success"
    summary["overview_json"] = str(ov)
    summary["original_kept_at"] = str(backup)
    if write_csv:
        summary["csv"] = merge_overview_into_csv(waiver_key, output_csv=output_csv)
    return summary


# --------------------------------------------------------------------------
# Custom value: extract ONE user-defined value that is not in the schema
# --------------------------------------------------------------------------

_CUSTOM_PROMPT = """You are extracting ONE specific value from a 1915(c) HCBS Medicaid waiver application PDF.

## What to extract
- Name: {label}
- Meaning: {description}
- What it usually contains: {usual_contents}
- Value type: {value_type}
- Where to look (hint only - search the whole document if it is not there): {where_to_look}
- Allowed values: {allowed_values}
{hint_block}
## Rules
1. Read the whole document. The location hint may be wrong or missing.
2. Return the value exactly as the document states it. Do not paraphrase or summarise text values. For
   yes_no return 1 for yes/checked and 0 for no/unchecked. For number return only the number (no units
   or commas). For date use YYYY-MM-DD when possible. For list return a JSON array of strings.
3. If the value is not in the document, return found=false and value=null. Never guess.
4. "evidence" must be the VERBATIM text you used (for checkboxes include CHECKED / UNCHECKED and the label).
5. "page" is the 1-indexed PDF page of the evidence.
6. If the same item appears in several places with different values, give the best answer in "value" and
   list the others in "other_occurrences" with their pages.

Return ONLY a JSON object:
{{"found": true/false, "value": ..., "page": <int or null>, "evidence": "<verbatim>",
  "confidence": "high|medium|low", "notes": "<anything the reviewer should know>",
  "other_occurrences": [{{"value": ..., "page": <int>, "evidence": "<verbatim>"}}]}}
"""


def extract_custom_value(
    waiver_key: str,
    label: str,
    description: str,
    value_type: str = "text",
    usual_contents: str = "",
    where_to_look: str = "",
    allowed_values: str = "",
    pdf_path: str = "",
    hint: str = "",
    tool_context: "ToolContext" = None,
) -> dict:
    """Extract ONE user-defined value that is not one of the standard schema
    fields. ONLY call this after the user has confirmed the definition (what
    it is, what it usually contains, its type). One model call; nothing is
    saved - the user must approve the result first (see save_custom_value).

    Args:
        waiver_key: The waiver identifier
        label: Short name for the value, e.g. "Case management agency name"
        description: What the value means, in the user's words
        value_type: "text", "number", "yes_no" (1/0), "date" or "list"
        usual_contents: What it normally looks like / example values
        where_to_look: Appendix/section/page hint, or "" if unknown
        allowed_values: Fixed list of allowed answers, or "" for free
        pdf_path: Optional PDF in any folder (default: found by waiver_key,
            or the PDF already loaded in this session)
        hint: One-off guidance for a retry (e.g. "look in Appendix A").
            Used for this call only; it is NOT part of the saved definition,
            so never put it in description / where_to_look.
    """
    from google.genai import types
    from ...core.pipeline import _load_pdf_bytes, _parse_json_response, _retry_on_connection_error
    from ...shared_libraries.model_config import get_model_name
    from ...shared_libraries.file_utils import log_event

    st = _state(tool_context)
    pdf = pdf_path or st.get("pdf_path") or ""
    found = _resolve_pdf(waiver_key, pdf)
    if found.get("status") != "found":
        return {"status": "error", "error": found.get("error", "PDF not found")}
    pdf = found["pdf_path"]
    if not label or not description:
        return {"status": "error", "error": "label and description are required - confirm them with the user first"}

    prompt = _CUSTOM_PROMPT.format(
        label=label, description=description, value_type=value_type,
        usual_contents=usual_contents or "(not specified)",
        where_to_look=where_to_look or "(unknown)",
        allowed_values=allowed_values or "(free)",
        hint_block=f"- Extra guidance for this attempt: {hint}\n" if hint else "",
    )
    model = get_model_name()
    log_event(waiver_key, {"agent": "custom_value", "action": "extraction_start",
                           "label": label, "model": model, "hint": hint})
    try:
        client = _adk_client(model)
        pdf_bytes = _load_pdf_bytes(pdf)

        def _request():
            r = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_bytes)),
                    types.Part(text=prompt),
                ])],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16384),
            )
            return r.text.strip()

        result = _parse_json_response(_retry_transient(lambda: _retry_on_connection_error(_request)))
    except Exception as e:  # noqa: BLE001
        log_event(waiver_key, {"agent": "custom_value", "action": "model_error", "label": label, "error": str(e)})
        return {"status": "error", "error": str(e)}

    result = {**result, "label": label, "description": description, "value_type": value_type,
              "usual_contents": usual_contents, "where_to_look": where_to_look,
              "model": model, "pdf_path": pdf}
    log_event(waiver_key, {"agent": "custom_value", "action": "extraction_complete", "label": label,
                           "found": result.get("found"), "confidence": result.get("confidence"),
                           "page": result.get("page")})
    if tool_context is not None:
        pending = dict(st.get("custom_pending") or {})
        pending[label] = result
        st["custom_pending"] = pending
        st["pdf_path"] = pdf
    return {"status": "success", "waiver_key": waiver_key, **result}


def save_custom_value(
    waiver_key: str,
    label: str,
    value: str = "",
    output_csv: str = "",
    dry_run: bool = False,
    tool_context: "ToolContext" = None,
) -> dict:
    """Save a user-approved custom value: to the waiver's custom_extractions.json
    and as one row (waiver_key + label) in the long-format custom-values CSV
    (default data/outputs/custom_extractions.csv). Call only after the user
    approved the value shown by extract_custom_value.

    Args:
        waiver_key: The waiver identifier
        label: The label used in extract_custom_value
        value: Optional corrected value the user wants saved instead of the
            extracted one (leave "" to save the extracted value as is)
        output_csv: Optional different CSV (file name -> data/outputs/, or a path)
        dry_run: Report what would be written without writing
    """
    import datetime
    from ...shared_libraries.paths import OUTPUTS_DIR
    from ...shared_libraries.file_utils import log_event
    from ...storage.csv_merge import merge_custom_value

    st = _state(tool_context)
    res = (st.get("custom_pending") or {}).get(label)
    if not res:
        return {"status": "error", "error": f"No extracted value for '{label}' in this session - run extract_custom_value first"}
    final = value if value != "" else res.get("value")
    if isinstance(final, (list, dict)):
        final = json.dumps(final, ensure_ascii=False)
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "waiver_key": waiver_key, "label": label, "value": final, "page": res.get("page"),
        "evidence": res.get("evidence"), "confidence": res.get("confidence"),
        "description": res.get("description"), "value_type": res.get("value_type"),
        "model": res.get("model"), "extracted_at": now, "approved_at": now,
    }
    csv_result = merge_custom_value(row, output_csv=output_csv, dry_run=dry_run)
    if dry_run or csv_result.get("status") == "error":
        return {"status": csv_result["status"], "row": row, "csv": csv_result}

    jp = OUTPUTS_DIR / waiver_key / "custom_extractions.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(jp.read_text()) if jp.exists() else {}
    data[label] = {**row, "found": res.get("found"), "notes": res.get("notes"),
                   "other_occurrences": res.get("other_occurrences"),
                   "usual_contents": res.get("usual_contents"), "where_to_look": res.get("where_to_look")}
    jp.write_text(json.dumps(data, indent=2, default=str))
    log_event(waiver_key, {"agent": "custom_value", "action": "saved", "label": label})
    return {"status": "success", "saved_value": final, "json_path": str(jp), "csv": csv_result}
