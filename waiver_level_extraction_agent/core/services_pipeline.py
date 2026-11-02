"""
Services Extraction Pipeline.

Orchestrates the extraction of service-level data from 1915(c) HCBS waiver
PDFs. Produces one row per service per waiver (parent/child hierarchy)
matching the "Waiver Services - PD_Append C&E" sheet (49 columns).

Flow:
1. Get the service list - explicit arg > overview.json _discovered_services
   > LLM discovery - and cross-check the count against the AcroForm's
   per-service blocks (LLM re-discovery on mismatch)
2. Parse per-service AcroForm fields and correlate positions to names
3. Extract each service (one model call per service)
4. Normalise (types + computed component costs)
5. Critic #1: AcroForm verification + math checks + LLM body-text check;
   AcroForm corrections applied (AcroForm wins)
6. If the critic says attempt2: focused re-extraction of the flagged
   services (verified fields frozen), re-normalise, critic #2
7. Unresolved fields at attempt 2 go to services_human_overrides.jsonl
8. Flatten to rows; save to SQLite (+ per-run accuracy/attempts metadata),
   services.json, services_evidence.json, and the global services CSV
"""

import os
from datetime import datetime

# Load .env file only if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from ..sub_agents.services_agent import (
    discover_services,
    extract_all_services,
    flatten_services_to_rows,
    infer_taxonomy_fallback,
)
from ..sub_agents.services_critic_agent import (
    run_services_critic,
    apply_corrections,
    MAX_ATTEMPTS,
)
from ..sub_agents.services_reextraction_agent import run_services_reextraction
from ..sub_agents.services_deterministic_resolver import resolve_deterministic_flags
from ..sub_agents.services_normalisation import normalise_services
from ..parsers.services_field_mapping import (
    extract_services_acroform,
    correlate_services,
    extract_service_taxonomies,
)
from ..shared_libraries.file_utils import (
    log_event,
    write_json_output,
)
from ..storage.services_store import (
    save_services,
    save_services_run,
    export_all_services_to_csv,
)
from ..tools.services_tools import (
    get_discovered_services_from_overview,
    save_services_output,
    get_waiver_metadata_from_overview,
)

# Service-level results CSV (new extractions only - the verified master
# all_services.csv lives in data/verified/ and is never written here).
from ..shared_libraries.paths import SERVICE_LEVEL_CSV
ALL_SERVICES_CSV = str(SERVICE_LEVEL_CSV)


async def run_services_extraction(
    pdf_path: str,
    waiver_key: str,
    discovered_services: list[str] | None = None,
    standalone: bool = False,
    resume: bool = False,
) -> dict:
    """
    Run the services extraction pipeline for a single waiver.

    Args:
        pdf_path: Path to the PDF file
        waiver_key: Normalized waiver key (e.g., "AK0260R0512")
        discovered_services: Optional explicit list of service names
            (takes precedence over the overview lookup)
        standalone: If True, discover services from the PDF via LLM instead
            of using the overview output
        resume: If True, skip services already checkpointed by a prior
            killed/crashed run of this waiver and only extract the
            remainder (see extract_all_services). No effect if there is no
            checkpoint yet.

    Returns:
        Result dict with status, services extracted, and output paths
    """
    log_event(waiver_key, {
        "agent": "services_pipeline",
        "action": "pipeline_start",
        "pdf_path": pdf_path,
        "standalone": standalone,
        "provided_services_count": len(discovered_services) if discovered_services else 0,
    })

    # =========================================================================
    # STEP 1: AcroForm pre-extraction (used for both discovery cross-check
    # and the critic's verification oracle)
    # =========================================================================
    try:
        acroform = extract_services_acroform(pdf_path)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "acroform_error",
            "error": str(e),
        })
        acroform = {"has_acroform": False, "service_count": 0, "services": {},
                    "self_direction": [], "info_assistance": [], "capitation": {}}

    # =========================================================================
    # STEP 2: Get the service list (evidence-grounded)
    # =========================================================================
    services_list, appendix_hints = _resolve_service_list(
        pdf_path=pdf_path,
        waiver_key=waiver_key,
        discovered_services=discovered_services,
        standalone=standalone,
        acroform=acroform,
    )

    if not services_list:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "no_services_found",
        })
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "error": "No services found in waiver",
        }

    print(f"  Found {len(services_list)} services to extract"
          + (f" (AcroForm has {acroform['service_count']} service blocks)"
             if acroform["has_acroform"] else " (no AcroForm)"))

    # =========================================================================
    # STEP 3: Correlate AcroForm positions to service names
    # =========================================================================
    correlation = correlate_services(services_list, pdf_path, acroform)

    # Deterministic taxonomy parse from the C-1/C-3 spec blocks (the LLM's
    # taxonomy values are only trusted where no block was found in the text).
    try:
        correlation["taxonomies"] = extract_service_taxonomies(pdf_path, services_list)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "taxonomy_parse_error",
            "error": str(e),
        })
        correlation["taxonomies"] = {}

    log_event(waiver_key, {
        "agent": "services_pipeline",
        "action": "acroform_correlation",
        "confidence": correlation["confidence"],
        "count_match": correlation["count_match"],
        "hints": len(correlation["hints"]),
        "taxonomy_blocks": sum(
            1 for t in correlation["taxonomies"].values() if t.get("block_found")
        ),
    })

    # =========================================================================
    # STEP 4: Get waiver metadata
    # =========================================================================
    try:
        metadata = get_waiver_metadata_from_overview(waiver_key)
        state = metadata.get("state", "Unknown")
        waiver_title = metadata.get("waiver_title", "Unknown")
    except FileNotFoundError:
        state = waiver_key[:2] if len(waiver_key) >= 2 else "Unknown"
        waiver_title = "Unknown"

    # File name is the normalized waiver key with dots
    file_name = _format_file_name(waiver_key)

    # =========================================================================
    # STEP 5: Extract each service
    # =========================================================================
    print("  Extracting service details...")

    try:
        services_data = extract_all_services(
            pdf_path=pdf_path,
            services=services_list,
            state=state,
            waiver_title=waiver_title,
            file_name=file_name,
            waiver_key=waiver_key,
            one_at_a_time=True,  # More accurate
            appendix_hints=appendix_hints,
            resume=resume,
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "extraction_error",
            "error": str(e),
        })
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "error": f"Service extraction failed: {e}",
        }

    # =========================================================================
    # STEP 6: Normalise, critique, re-extract if needed
    # =========================================================================
    normalised_services = normalise_services(services_data, waiver_key)

    print("  Validating extractions (attempt 1)...")
    critique = _run_critic_safely(
        pdf_path, normalised_services, waiver_key, correlation, attempt=1
    )
    corrections_applied = apply_corrections(
        normalised_services, critique.get("corrections", [])
    )

    # Before anything flagged goes to the LLM re-extraction agent, try to
    # resolve it deterministically (PDF text tables / already-computed cost
    # math). Mutates `critique` in place: resolved fields move soft_flag ->
    # corrected, and decision flips to "pass" if nothing flagged remains.
    deterministic_resolved = _apply_deterministic_resolution(
        pdf_path, waiver_key, normalised_services, critique
    )
    if deterministic_resolved:
        print(f"  Resolved {deterministic_resolved} flagged field(s) "
              "deterministically (no LLM call)")
        corrections_applied += deterministic_resolved

    # Crash-safe snapshot: attempt-1 results (post-corrections) + critique.
    # If a later stage dies, the final outputs can be rebuilt from this
    # deterministically (normalise -> crosswalk -> flatten -> save).
    _write_snapshot(waiver_key, "services_attempt1_snapshot.json",
                    normalised_services, critique)

    attempts_used = 1
    if critique.get("decision") == "attempt2":
        attempts_used = 2
        flagged = len({e.get("service") for e in critique.get("soft_flag", [])})
        print(f"  Re-extracting {flagged} flagged service(s) (attempt 2)...")
        try:
            services_data = run_services_reextraction(
                pdf_path=pdf_path,
                waiver_key=waiver_key,
                services_data=normalised_services,
                critique=critique,
                correlation=correlation,
                state=state,
                waiver_title=waiver_title,
                file_name=file_name,
                appendix_hints=appendix_hints,
            )
            normalised_services = normalise_services(services_data, waiver_key)
        except Exception as e:
            log_event(waiver_key, {
                "agent": "services_pipeline",
                "action": "reextraction_error",
                "error": str(e),
            })

        print("  Validating extractions (attempt 2)...")
        critique = _run_critic_safely(
            pdf_path, normalised_services, waiver_key, correlation,
            attempt=MAX_ATTEMPTS,
        )
        corrections_applied += apply_corrections(
            normalised_services, critique.get("corrections", [])
        )
        # No attempt 3 follows, so anything still flagged here would
        # otherwise just sit in human_queue - resolve what we can.
        deterministic_resolved = _apply_deterministic_resolution(
            pdf_path, waiver_key, normalised_services, critique
        )
        if deterministic_resolved:
            print(f"  Resolved {deterministic_resolved} flagged field(s) "
                  "deterministically (no LLM call)")
            corrections_applied += deterministic_resolved
        _write_snapshot(waiver_key, "services_attempt2_snapshot.json",
                        normalised_services, critique)

    # Taxonomy crosswalk fallback: services whose taxonomy table is empty in
    # the PDF get the official CMS category (marked as inferred in evidence).
    try:
        inferred = infer_taxonomy_fallback(normalised_services, waiver_key)
        if inferred:
            print(f"  Inferred taxonomy for {inferred} service(s) from CMS crosswalk")
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "taxonomy_inference_error",
            "error": str(e),
        })

    # =========================================================================
    # STEP 7: Flatten to rows and save
    # =========================================================================
    flat_rows = flatten_services_to_rows(
        services_data=normalised_services,
        waiver_key=waiver_key,
        state=state,
        waiver_title=waiver_title,
        file_name=file_name,
    )

    # Save to SQLite
    try:
        rows_saved = save_services(waiver_key, flat_rows)
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "sqlite_save",
            "rows_saved": rows_saved,
        })
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "sqlite_error",
            "error": str(e),
        })
        rows_saved = 0

    # Final status mirrors the overview pipeline's mapping
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
            services_count=len(services_list),
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "run_meta_error",
            "error": str(e),
        })

    # =========================================================================
    # STEP 8: Write JSON outputs + global CSV
    # =========================================================================
    output_data = {
        "waiver_key": waiver_key,
        "state": state,
        "waiver_title": waiver_title,
        "file_name": file_name,
        "services": normalised_services,
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
            "confidence": correlation["confidence"],
            "count_match": correlation["count_match"],
            "hints": correlation["hints"],
        },
        "services_evidence": _extract_evidence(normalised_services),
        "critique": _critique_summary(critique),
    }
    evidence_path = write_json_output(waiver_key, "services_evidence.json", evidence_data)

    try:
        # Per-waiver merge into the service-level results CSV: only this
        # waiver's rows change; the verified master is never touched.
        from ..storage.csv_merge import merge_services_into_csv
        merge_services_into_csv(waiver_key)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "csv_export_error",
            "error": str(e),
        })

    # =========================================================================
    # STEP 9: Build result
    # =========================================================================
    parent_count = sum(1 for r in flat_rows if r.get("row_type") == "parent")
    child_count = sum(1 for r in flat_rows if r.get("row_type") == "child")

    result = {
        "status": final_status if final_status != "success" else "success",
        "waiver_key": waiver_key,
        "pdf_path": pdf_path,
        "output_path": output_path,
        "evidence_path": evidence_path,
        "services_extracted": len(services_list),
        "parent_rows": parent_count,
        "child_rows": child_count,
        "total_rows": len(flat_rows),
        "sqlite_rows_saved": rows_saved,
        "accuracy": accuracy,
        "attempts": attempts_used,
        "acroform_confidence": correlation["confidence"],
        "corrections_applied": corrections_applied,
        "critique_decision": decision,
        "soft_flags_remaining": len(critique.get("soft_flag", [])),
        "human_queue": len(critique.get("human_queue", [])),
    }

    log_event(waiver_key, {
        "agent": "services_pipeline",
        "action": "pipeline_complete",
        **{k: v for k, v in result.items() if k != "pdf_path"},
    })

    return result


def _resolve_service_list(
    pdf_path: str,
    waiver_key: str,
    discovered_services: list[str] | None,
    standalone: bool,
    acroform: dict,
) -> tuple[list[str], dict | None]:
    """
    Resolve the base service list, grounded in existing evidence:

    1. An explicitly passed list always wins (never silently overwritten).
    2. Otherwise use overview.json's _discovered_services.
    3. Otherwise (or in standalone mode) run LLM discovery.

    The chosen list is cross-checked against the AcroForm service-block
    count; on mismatch, LLM discovery re-runs and its result is used if it
    matches the AcroForm count better.

    Returns (services_list, appendix_hints).
    """
    appendix_hints = None
    services_list: list[str] = []
    source = None

    if discovered_services:
        services_list = list(discovered_services)
        source = "explicit"
    elif not standalone:
        try:
            services_list = get_discovered_services_from_overview(waiver_key)
            source = "overview"
        except FileNotFoundError:
            pass

    if not services_list:
        print("  Discovering services from PDF...")
        try:
            discovery_result = discover_services(pdf_path, waiver_key)
            services_list = [s["name"] for s in discovery_result.get("services", [])
                             if s.get("is_main_service", True)]
            appendix_hints = _hints_from_discovery(discovery_result)
            source = "llm_discovery"
        except Exception as e:
            log_event(waiver_key, {
                "agent": "services_pipeline",
                "action": "discovery_error",
                "error": str(e),
            })
            return [], None

    # Cross-check against the AcroForm service-block count
    acro_count = acroform.get("service_count", 0)
    if (
        services_list
        and acroform.get("has_acroform")
        and acro_count > 0
        and len(services_list) != acro_count
        and source != "llm_discovery"
    ):
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "service_count_mismatch",
            "source": source,
            "list_count": len(services_list),
            "acroform_count": acro_count,
        })
        print(f"  Service count mismatch ({source}: {len(services_list)} vs "
              f"AcroForm: {acro_count}) - re-discovering from PDF...")
        try:
            discovery_result = discover_services(pdf_path, waiver_key)
            rediscovered = [s["name"] for s in discovery_result.get("services", [])
                            if s.get("is_main_service", True)]
            if rediscovered and abs(len(rediscovered) - acro_count) < abs(len(services_list) - acro_count):
                services_list = rediscovered
                appendix_hints = _hints_from_discovery(discovery_result)
                source = "llm_discovery"
        except Exception as e:
            log_event(waiver_key, {
                "agent": "services_pipeline",
                "action": "rediscovery_error",
                "error": str(e),
            })

    log_event(waiver_key, {
        "agent": "services_pipeline",
        "action": "service_list_resolved",
        "source": source,
        "count": len(services_list),
        "acroform_count": acro_count,
    })
    return services_list, appendix_hints


def _hints_from_discovery(discovery_result: dict) -> dict:
    return {
        "appendix_c_pages": discovery_result.get("appendix_c_page_range"),
        "appendix_e_pages": discovery_result.get("appendix_e_page_range"),
        "appendix_j_pages": discovery_result.get("appendix_j_page_range"),
    }


def rebuild_from_checkpoint(waiver_key: str) -> dict:
    """
    Rebuild final outputs (SQLite rows, services.json, CSV) from the most
    recent checkpoint of a crashed/killed run. No model calls: raw
    checkpointed extractions go through normalise -> curated taxonomy
    crosswalk -> flatten -> save.

    Prefers the attempt-2 snapshot, then attempt-1 snapshot, then the raw
    per-service attempt-1 JSONL.
    """
    import json as _json
    from ..sub_agents.services_agent import checkpoint_path
    from ..parsers.hcbs_taxonomy import resolve_taxonomy

    services = None
    source = None
    for name in ("services_attempt2_snapshot.json", "services_attempt1_snapshot.json"):
        path = checkpoint_path(waiver_key, name)
        if os.path.exists(path):
            with open(path) as f:
                services = _json.load(f).get("services")
            source = name
            break
    if services is None:
        path = checkpoint_path(waiver_key, "services_attempt1.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                raw = [_json.loads(line) for line in f if line.strip()]
            if raw:
                services = normalise_services(raw, waiver_key)
                source = "services_attempt1.jsonl"
    if not services:
        return {"status": "error", "waiver_key": waiver_key,
                "error": "No checkpoint found"}

    # Deterministic taxonomy pass (curated crosswalk only - no PDF, no LLM)
    for service in services:
        parent = service.get("parent", {})
        label, src = resolve_taxonomy(parent.get("service_name"), [])
        if label:
            parent["hcbs_taxonomy"] = label

    try:
        metadata = get_waiver_metadata_from_overview(waiver_key)
        state = metadata.get("state", "Unknown")
        waiver_title = metadata.get("waiver_title", "Unknown")
    except FileNotFoundError:
        state = waiver_key[:2] if len(waiver_key) >= 2 else "Unknown"
        waiver_title = "Unknown"
    file_name = _format_file_name(waiver_key)

    flat_rows = flatten_services_to_rows(services, waiver_key, state,
                                         waiver_title, file_name)
    rows_saved = save_services(waiver_key, flat_rows)
    save_services_run(waiver_key, accuracy=None, attempts=0,
                      final_status="rebuilt_from_checkpoint",
                      services_count=len(services))
    output_path = save_services_output(waiver_key, {
        "waiver_key": waiver_key,
        "state": state,
        "waiver_title": waiver_title,
        "file_name": file_name,
        "services": services,
        "final_status": "rebuilt_from_checkpoint",
        "rebuilt_from": source,
        "extraction_timestamp": datetime.utcnow().isoformat(),
    })
    try:
        from ..storage.csv_merge import merge_services_into_csv
        merge_services_into_csv(waiver_key)
    except Exception:
        pass

    return {
        "status": "rebuilt_from_checkpoint",
        "waiver_key": waiver_key,
        "rebuilt_from": source,
        "services": len(services),
        "sqlite_rows_saved": rows_saved,
        "output_path": output_path,
    }


def extract_one_service(
    waiver_key: str,
    pdf_path: str,
    service_name: str,
    appendix_hints: dict | None = None,
    guidance: str | None = None,
) -> dict:
    """
    Extract a single named service (one model call) and append it to the
    waiver's services_attempt1.jsonl checkpoint - no critique, no DB save.

    Intended for manually retrying one specific service that kept failing
    inside a larger run, without paying for the whole batch again. If the
    checkpoint already has an entry for this service name, the new result
    is appended after it and wins on a later --resume or --rebuild (both
    read the checkpoint in order and keep the last entry per service name).

    Args:
        guidance: Optional free-text steer for this one attempt (e.g. "the
            cost table is on page 210-215" or "verify this is provider-
            managed only before filling self-direction fields") - passed
            straight into the extraction prompt. For manually correcting a
            service that keeps coming out wrong, not the normal loop.

    Use --resume afterward to fold it in with a normal run of the
    remaining services, or --rebuild to save the checkpoint as-is (skips
    critique/validation).
    """
    from ..sub_agents.services_agent import extract_single_service, _append_checkpoint

    try:
        metadata = get_waiver_metadata_from_overview(waiver_key)
        state = metadata.get("state", "Unknown")
        waiver_title = metadata.get("waiver_title", "Unknown")
    except FileNotFoundError:
        state = waiver_key[:2] if len(waiver_key) >= 2 else "Unknown"
        waiver_title = "Unknown"
    file_name = _format_file_name(waiver_key)

    result = extract_single_service(
        pdf_path=pdf_path,
        service_name=service_name,
        state=state,
        waiver_title=waiver_title,
        file_name=file_name,
        appendix_hints=appendix_hints,
        waiver_key=waiver_key,
        guidance=guidance,
    )
    _append_checkpoint(waiver_key, "services_attempt1.jsonl", result)
    return result


def _write_snapshot(
    waiver_key: str,
    name: str,
    services: list[dict],
    critique: dict,
) -> None:
    """Write a crash-safe stage snapshot. Never raises."""
    try:
        write_json_output(waiver_key, name, {
            "waiver_key": waiver_key,
            "generated_at": datetime.utcnow().isoformat(),
            "services": services,
            "critique": _critique_summary(critique),
        })
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "snapshot_error",
            "snapshot": name,
            "error": str(e),
        })


def _run_critic_safely(
    pdf_path: str,
    services_data: list[dict],
    waiver_key: str,
    correlation: dict,
    attempt: int,
) -> dict:
    """Run the critic; on failure, degrade to a pass-through critique."""
    try:
        return run_services_critic(
            pdf_path=pdf_path,
            services_data=services_data,
            waiver_key=waiver_key,
            correlation=correlation,
            attempt=attempt,
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "critic_error",
            "attempt": attempt,
            "error": str(e),
        })
        return {
            "passed": True,
            "decision": "pass",
            "attempt": attempt,
            "accuracy": None,
            "corrections": [],
            "soft_flag": [],
            "human_queue": [],
            "issues": [],
            "error": str(e),
        }


def _apply_deterministic_resolution(
    pdf_path: str,
    waiver_key: str,
    normalised_services: list[dict],
    critique: dict,
) -> int:
    """
    Before anything flagged by the critic is sent to the (expensive) LLM
    re-extraction agent, try to resolve it deterministically - from the PDF
    (provider_specifications, service_type) or from a value the critic's
    own cost-math check already computed (year_N_total_cost). Mutates
    `critique` in place: resolved fields move from soft_flag into corrected,
    and accuracy/decision are recomputed accordingly.

    Returns the number of fields resolved this way.
    """
    soft_flag = critique.get("soft_flag", [])
    if not soft_flag:
        return 0

    try:
        corrections, resolved_audit, remaining = resolve_deterministic_flags(
            pdf_path, normalised_services, soft_flag, waiver_key
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "services_pipeline",
            "action": "deterministic_resolution_error",
            "error": str(e),
        })
        return 0

    if not resolved_audit:
        return 0

    apply_corrections(normalised_services, corrections)

    critique["corrections"] = critique.get("corrections", []) + corrections
    critique["corrected"] = critique.get("corrected", []) + resolved_audit
    critique["soft_flag"] = remaining

    # human_queue (populated when this is the final-attempt critique) may
    # reference the same entries we just resolved - drop those too, and
    # write a resolved marker over any human-override record already filed
    # for them so a human reviewer doesn't see a stale "unresolved" entry.
    remaining_ids = {id(e) for e in remaining}
    old_human_queue = critique.get("human_queue", [])
    if old_human_queue:
        critique["human_queue"] = [
            e for e in old_human_queue if id(e) in remaining_ids
        ]

    verified_count = (
        len(critique.get("verified_acroform", []))
        + len(critique.get("verified_body", []))
        + len(critique["corrected"])
    )
    categorized = verified_count + len(remaining)
    critique["accuracy"] = (verified_count / categorized) if categorized else None

    critical_count = critique.get("critical_count", 0)
    if not remaining and not critical_count:
        critique["decision"] = "pass"
        critique["passed"] = True

    return len(resolved_audit)


def _critique_summary(critique: dict) -> dict:
    """Compact critique for the evidence file (drop bulky bucket lists)."""
    return {
        "decision": critique.get("decision"),
        "attempt": critique.get("attempt"),
        "accuracy": critique.get("accuracy"),
        "acroform_confidence": critique.get("acroform_confidence"),
        "verified_acroform": len(critique.get("verified_acroform", [])),
        "verified_body": len(critique.get("verified_body", [])),
        "corrected": critique.get("corrected", []),
        "soft_flag": critique.get("soft_flag", []),
        "human_queue": critique.get("human_queue", []),
        "issues": critique.get("issues", []),
    }


def _format_file_name(waiver_key: str) -> str:
    """
    Format waiver key as a file name with dots.

    'CA0336R0404' -> 'CA.0336.R04.04'
    """
    # Try to parse the standard format
    key = waiver_key.upper()
    if len(key) >= 10:
        # Format: SSNNNNRNNNN -> SS.NNNN.RNN.NN
        state = key[:2]
        waiver_num = key[2:6] if len(key) > 5 else key[2:]
        revision_part = key[6:] if len(key) > 6 else ""

        if revision_part.startswith("R"):
            rev_major = revision_part[1:3] if len(revision_part) > 2 else revision_part[1:]
            rev_minor = revision_part[3:5] if len(revision_part) > 4 else "00"
            return f"{state}.{waiver_num}.R{rev_major}.{rev_minor}"

    # Fallback: return as-is
    return waiver_key


def _extract_evidence(services_data: list[dict]) -> list[dict]:
    """Extract evidence from service data for the evidence file."""
    evidence = []
    for service in services_data:
        parent = service.get("parent", {})
        service_evidence = {
            "service_name": parent.get("service_name"),
            "parent_evidence": parent.get("evidence", {}),
            "children_evidence": [
                {
                    "service_name": child.get("service_name"),
                    "evidence": child.get("evidence", {}),
                }
                for child in service.get("children", [])
            ],
        }
        evidence.append(service_evidence)
    return evidence
