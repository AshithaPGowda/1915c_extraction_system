"""
Deterministic resolver for critic-flagged (soft_flag) fields.

Rule: before any flagged field is sent to the re-extraction agent (an LLM
call), check whether it can be resolved directly from the PDF or from a
value the critic already computed. Only fields nothing here knows how to
resolve fall through to the LLM.

This is a registry, not a one-off special case: each field name maps to a
resolver function. Field types discovered in future waivers that aren't
handled here simply pass through unchanged to `remaining` - same as if
this module didn't run at all - so the LLM path never disappears, it's
just no longer hit for things we already know how to answer deterministically.

Coverage as of the 60-waiver sample this was built against: every soft_flag
field seen (`provider_specifications`, `service_type`, `year_N_total_cost`)
has a resolver here.
"""

from collections import defaultdict

from ..parsers.services_field_mapping import (
    extract_provider_specifications,
    extract_service_type,
)
from ..shared_libraries.file_utils import log_event


def _resolve_provider_specifications(pdf_path, services_data, entries):
    """Parse the 'Provider Category' table for the flagged services."""
    names = sorted({e["service"] for e in entries if e.get("service")})
    if not names:
        return {}
    parsed = extract_provider_specifications(pdf_path, names)
    resolved = {}
    for e in entries:
        info = parsed.get(e.get("service")) or {}
        value = info.get("value")
        if value is not None:
            resolved[id(e)] = {
                "corrected_value": value,
                "source": "provider_category_table",
                "detail": info.get("categories"),
            }
    return resolved


def _resolve_service_type(pdf_path, services_data, entries):
    """Parse the 'Service Type:' label for the flagged services."""
    names = sorted({e["service"] for e in entries if e.get("service")})
    if not names:
        return {}
    parsed = extract_service_type(pdf_path, names)
    resolved = {}
    for e in entries:
        info = parsed.get(e.get("service")) or {}
        value = info.get("value")
        if value is not None:
            resolved[id(e)] = {
                "corrected_value": value,
                "source": "service_type_label",
                "detail": info.get("raw_label"),
            }
    return resolved


def _resolve_cost_math(pdf_path, services_data, entries):
    """
    Use the value the critic's cost-math check already computed
    (parent total = sum of child component costs, or
    users x avg_units x avg_cost) instead of asking the LLM to re-derive it.
    Pure arithmetic on fields already in the record - no PDF read needed.
    """
    resolved = {}
    for e in entries:
        expected = e.get("expected")
        if expected is not None:
            resolved[id(e)] = {
                "corrected_value": expected,
                "source": "cost_math",
                "detail": e.get("reason"),
            }
    return resolved


# field_name -> resolver(pdf_path, services_data, entries_for_that_field)
# Add new entries here as new soft_flag field types are discovered; anything
# not registered safely falls through to the LLM re-extraction agent.
_RESOLVERS = {
    "provider_specifications": _resolve_provider_specifications,
    "service_type": _resolve_service_type,
}


def _is_cost_field(field_name: str) -> bool:
    return bool(field_name) and field_name.endswith("_total_cost")


def resolve_deterministic_flags(
    pdf_path: str,
    services_data: list[dict],
    soft_flag: list[dict],
    waiver_key: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Try to resolve each critic-flagged field deterministically before it
    reaches the re-extraction agent.

    Returns:
        (corrections, resolved_audit, remaining)
        corrections:    [{service_index, field, corrected_value}] - ready
                         for apply_corrections()
        resolved_audit: audit-trail entries mirroring the critic's
                         `corrected` bucket shape (for logging/accuracy)
        remaining:      soft_flag entries nothing here could resolve -
                         unchanged, ready to hand to run_services_reextraction
    """
    by_field: dict[str, list[dict]] = defaultdict(list)
    for entry in soft_flag:
        by_field[entry.get("field")].append(entry)

    resolved_by_id: dict[int, dict] = {}

    for field, entries in by_field.items():
        if _is_cost_field(field):
            resolved_by_id.update(_resolve_cost_math(pdf_path, services_data, entries))
            continue
        resolver = _RESOLVERS.get(field)
        if resolver is None:
            continue
        resolved_by_id.update(resolver(pdf_path, services_data, entries))

    corrections = []
    resolved_audit = []
    remaining = []

    for entry in soft_flag:
        hit = resolved_by_id.get(id(entry))
        if hit is None:
            remaining.append(entry)
            continue
        corrections.append({
            "service_index": entry.get("service_index"),
            "field": entry.get("field"),
            "corrected_value": hit["corrected_value"],
        })
        resolved_audit.append({
            "service": entry.get("service"),
            "service_index": entry.get("service_index"),
            "field": entry.get("field"),
            "llm_value": entry.get("llm_value"),
            "corrected_value": hit["corrected_value"],
            "source": hit["source"],
            "detail": hit.get("detail"),
        })

    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_deterministic_resolver",
            "action": "resolve_complete",
            "flagged": len(soft_flag),
            "resolved": len(resolved_audit),
            "remaining": len(remaining),
            "resolved_fields": sorted({r["field"] for r in resolved_audit}),
        })

    return corrections, resolved_audit, remaining
