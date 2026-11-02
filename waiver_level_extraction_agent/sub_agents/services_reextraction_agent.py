"""
Re-extraction agent for the services pipeline.

Mirrors the overview re-extraction agent (sub_agents/reextraction_agent.py):
after the critic flags fields as soft_flag, the failing services are
re-extracted with a focused prompt that injects:
- which fields failed and why
- AcroForm hints (what the PDF form itself says - authoritative)
- page hints from the captured evidence

Verified fields are FROZEN: the merge only updates the failing fields
(and the children table when cost math failed), never fields the critic
already verified.

MAX_ATTEMPTS = 2 for services (extraction is per-service and much more
focused than the 163-field overview extraction).
"""

import time

from .services_agent import _get_client, _parse_json_response, PdfCache, SERVICE_CALL_DELAY_SECONDS
from .services_prompt import build_service_extraction_prompt
from ..shared_libraries.file_utils import log_event


# Fields whose failure means the Appendix J children table must be redone.
_COST_FIELD_MARKERS = ("total_cost", "users", "unit", "avg_", "component_cost")


def _is_cost_field(field: str | None) -> bool:
    if not field:
        return False
    return any(marker in field for marker in _COST_FIELD_MARKERS)


def _format_acroform_hints(hint: dict | None) -> str:
    """Render the AcroForm hints for one service as prompt text."""
    if not hint:
        return "  (no AcroForm data available for this service)"
    lines = []
    if hint.get("delivery_method") is not None:
        lines.append(
            f"  - Service Delivery Method = {hint['delivery_method']} "
            "(1=Participant-directed, 2=Provider managed, 3=Both)"
        )
    if hint.get("provider_eligibility") is not None:
        lines.append(
            f"  - Provider eligibility digits = {hint['provider_eligibility']} "
            "(1=Legally Responsible Person, 2=Relative, 3=Legal Guardian)"
        )
    if hint.get("self_direction_authority") is not None:
        lines.append(
            f"  - E-1g Self-direction Authority = {hint['self_direction_authority']} "
            "(1=Employer, 2=Budget, 3=Both)"
        )
    if hint.get("information_assistance") is not None:
        lines.append(
            f"  - E-1j Information & Assistance = {hint['information_assistance']}"
        )
    for year, cap in (hint.get("capitation") or {}).items():
        lines.append(f"  - J-2d Capitation Year {year} = {cap}")
    return "\n".join(lines) if lines else "  (no AcroForm data available for this service)"


def _build_focused_prompt(
    service_name: str,
    state: str,
    waiver_title: str,
    file_name: str,
    failing_entries: list[dict],
    acroform_hint: dict | None,
    evidence: dict | None,
    appendix_hints: dict | None,
) -> str:
    """The standard extraction prompt plus a focused correction block."""
    base = build_service_extraction_prompt(
        service_name=service_name,
        state=state,
        waiver_title=waiver_title,
        file_name=file_name,
        appendix_hints=appendix_hints,
    )

    failures = "\n".join(
        f"  - `{e.get('field')}`: previous value {e.get('llm_value')!r} - "
        f"{e.get('reason') or 'conflicted with the PDF form (AcroForm)'}"
        for e in failing_entries
    )

    pages = []
    for key in ("appendix_c_page", "appendix_e_page", "appendix_j_page"):
        page = (evidence or {}).get(key)
        if page:
            pages.append(f"{key.replace('_', ' ')}: {page}")
    page_note = (
        f"\nPages where this service was previously found: {', '.join(pages)}"
        if pages else ""
    )

    correction_block = f"""

## RE-EXTRACTION - THIS IS ATTEMPT 2

A verification pass found problems with the previous extraction of
"{service_name}". Focus on getting these fields right this time:

{failures}

## AcroForm ground truth (from the PDF's own form fields - AUTHORITATIVE)

The PDF form checkboxes say:
{_format_acroform_hints(acroform_hint)}

If your reading of the body text conflicts with the AcroForm values above,
the AcroForm values are correct - the form fields ARE the document's data.
{page_note}

Re-extract the COMPLETE service (parent and children) in the same JSON format,
paying particular attention to the failing fields listed above. Re-check the
Appendix J tables row by row: every sub-service row belongs to a child, and
users x avg units per user x avg cost per unit must reproduce the printed
component/total cost.
"""
    return base + correction_block


def merge_service_extraction(
    original: dict,
    reextracted: dict,
    failing_fields: set[str],
    replace_children: bool,
) -> dict:
    """
    Merge a re-extracted service into the original: only the failing fields
    are updated; everything the critic verified stays frozen.

    Args:
        original: The original {'parent', 'children'} dict
        reextracted: The re-extracted {'parent', 'children'} dict
        failing_fields: Parent field names allowed to change
        replace_children: If True, the children array (and parent per-year
            cost fields) come from the re-extraction

    Returns:
        Merged service dict
    """
    merged_parent = dict(original.get("parent", {}))
    new_parent = reextracted.get("parent", {})

    for field in failing_fields:
        if field in new_parent:
            merged_parent[field] = new_parent[field]

    if replace_children:
        for year in range(1, 6):
            for suffix in ("capitation", "unit", "users", "avg_units_per_user",
                           "avg_cost_per_unit", "total_cost"):
                key = f"year_{year}_{suffix}"
                if key in new_parent:
                    merged_parent[key] = new_parent[key]
        children = reextracted.get("children", original.get("children", []))
    else:
        children = original.get("children", [])

    # Refresh evidence for the re-extracted fields
    if isinstance(new_parent.get("evidence"), dict):
        evidence = dict(merged_parent.get("evidence") or {})
        evidence.update(new_parent["evidence"])
        merged_parent["evidence"] = evidence

    return {"parent": merged_parent, "children": children}


def run_services_reextraction(
    pdf_path: str,
    waiver_key: str,
    services_data: list[dict],
    critique: dict,
    correlation: dict,
    state: str,
    waiver_title: str,
    file_name: str,
    appendix_hints: dict | None = None,
) -> list[dict]:
    """
    Re-extract the services flagged by the critic and merge the results.

    Args:
        pdf_path: Path to the PDF
        waiver_key: Waiver key for logging
        services_data: Current extraction (list of {'parent','children'})
        critique: Critic output (uses its 'soft_flag' entries)
        correlation: AcroForm correlation (hints injected into prompts)
        state, waiver_title, file_name: Waiver metadata for the prompt
        appendix_hints: Optional appendix page ranges

    Returns:
        New services_data list with failing services re-extracted and merged.
    """
    soft_flags = critique.get("soft_flag", [])
    if not soft_flags:
        return services_data

    # Group failures by service index
    by_service: dict[int, list[dict]] = {}
    name_to_index = {
        s.get("parent", {}).get("service_name"): i
        for i, s in enumerate(services_data)
    }
    for entry in soft_flags:
        idx = entry.get("service_index")
        if idx is None:
            idx = name_to_index.get(entry.get("service"))
        if idx is None or idx >= len(services_data):
            continue
        by_service.setdefault(idx, []).append(entry)

    log_event(waiver_key, {
        "agent": "services_reextraction",
        "action": "reextraction_start",
        "services_flagged": len(by_service),
        "fields_flagged": len(soft_flags),
    })

    hints = correlation.get("hints", {})
    merged_data = list(services_data)

    flagged_items = sorted(by_service.items())
    with PdfCache(_get_client(), pdf_path, waiver_key) as cache:
        for position, (idx, entries) in enumerate(flagged_items, 1):
            original = services_data[idx]
            parent = original.get("parent", {})
            service_name = parent.get("service_name")
            if not service_name:
                continue

            failing_fields = {e.get("field") for e in entries if e.get("field")}
            replace_children = any(_is_cost_field(f) for f in failing_fields)

            prompt = _build_focused_prompt(
                service_name=service_name,
                state=state,
                waiver_title=waiver_title,
                file_name=file_name,
                failing_entries=entries,
                acroform_hint=hints.get(service_name),
                evidence=parent.get("evidence"),
                appendix_hints=appendix_hints,
            )

            try:
                raw_text = cache.call(prompt)
                reextracted = _parse_json_response(raw_text)
                merged_data[idx] = merge_service_extraction(
                    original=original,
                    reextracted=reextracted,
                    failing_fields=failing_fields,
                    replace_children=replace_children,
                )
                log_event(waiver_key, {
                    "agent": "services_reextraction",
                    "action": "service_reextracted",
                    "service_name": service_name,
                    "fields": sorted(failing_fields),
                    "children_replaced": replace_children,
                })
            except Exception as e:
                log_event(waiver_key, {
                    "agent": "services_reextraction",
                    "action": "reextraction_error",
                    "service_name": service_name,
                    "error": str(e),
                })
                # Keep the original extraction for this service.

            if position < len(flagged_items):
                time.sleep(SERVICE_CALL_DELAY_SECONDS)

    return merged_data
