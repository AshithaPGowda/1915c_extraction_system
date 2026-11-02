"""
Services Critic Agent - verification oracle for services extraction.

Mirrors the overview critic (sub_agents/critic_agent.py) with three steps:

1. AcroForm verification (DETERMINISTIC) - per-service checkboxes from the
   PDF form are the ground truth for delivery method, provider eligibility,
   E-1g self-direction authority, E-1j information & assistance, and J-2d
   capitation. Requires the position<->name correlation from
   parsers/services_field_mapping.correlate_services().

2. Deterministic math & consistency checks - component cost (users x avg
   units x avg cost/unit) vs extracted total cost, parent total vs sum of
   child component costs, coded-field ranges, required fields, parent/child
   consistency.

3. LLM body-text verification - for fields with no AcroForm mapping
   (service type, HCBS taxonomy, provider specifications), checks whether
   the captured evidence supports the extracted value.

Classification buckets per (service, field), following the overview critic:
    verified_acroform  AcroForm confirmed the LLM value
    corrected          AcroForm overrode the LLM value (AcroForm wins)
    verified_body      evidence supports the value (LLM check)
    soft_flag          conflict or unsupported - candidate for re-extraction
    human_queue        still failing at the final attempt

Decision: "pass" | "attempt2" | "human"
Accuracy: (verified_acroform + verified_body + corrected) / total categorized
"""

import os
import json
import fcntl
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types

from ..shared_libraries.file_utils import log_event
from ..parsers.services_field_mapping import provider_eligibility_set
from ..parsers.hcbs_taxonomy import resolve_taxonomy
from .services_agent import _retry_on_connection_error

MAX_ATTEMPTS = 2
COST_TOLERANCE = 0.02  # 2% tolerance for component-cost vs total-cost math

from ..shared_libraries.paths import SERVICES_HUMAN_OVERRIDES_PATH
HUMAN_OVERRIDES_PATH = str(SERVICES_HUMAN_OVERRIDES_PATH)

# Fields verified against AcroForm hints (extraction key -> hint key)
ACROFORM_FIELDS = {
    "service_delivery_method": "delivery_method",
    "provider_eligibility": "provider_eligibility",
    "self_direction_authority": "self_direction_authority",
    "information_assistance": "information_assistance",
}

# Fields verified against captured evidence text via LLM.
# hcbs_taxonomy is NOT here: it is parsed deterministically from the
# C-1/C-3 spec blocks (extract_service_taxonomies) and corrected like an
# AcroForm field.
BODY_TEXT_FIELDS = ["service_type", "provider_specifications"]


BODY_TEXT_CRITIC_SYSTEM = """You are a verification agent for Medicaid 1915(c) waiver service extraction.
You are given extracted field values together with the verbatim PDF evidence
captured for them. For each field decide whether the evidence supports the
extracted value.

Rules:
- The evidence is verbatim text from the waiver PDF. Judge ONLY from it.
- "supported" means the evidence, read plainly, justifies the value.
- If evidence is missing, empty, or unrelated to the field, it cannot support
  a non-null value.
- A null value with no evidence is fine to mark supported (nothing claimed).
- Return confidence "high" only when the evidence is explicit."""


def _get_client():
    """Get Gemini client. Checks GOOGLE_GENAI_USE_VERTEXAI env var."""
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE"
    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION", "us-central1"),
        )
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)


def _write_services_human_override(
    waiver_key: str,
    service_name: str,
    service_index: int | None,
    field_name: str,
    acroform_value: Any,
    llm_value: Any,
    conflict_type: str,
    attempt_reached: int,
) -> None:
    """Append a record to services_human_overrides.jsonl (flock-protected)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "waiver_key": waiver_key,
        "service_name": service_name,
        "service_index": service_index,
        "field_name": field_name,
        "acroform_value": acroform_value,
        "llm_value": llm_value,
        "conflict_type": conflict_type,
        "attempt_reached": attempt_reached,
        "human_decision": None,
        "final_value": None,
    }
    os.makedirs(os.path.dirname(HUMAN_OVERRIDES_PATH), exist_ok=True)
    with open(HUMAN_OVERRIDES_PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# =============================================================================
# Step 1: AcroForm verification
# =============================================================================

def _values_match(field: str, llm_value: Any, acro_value: Any) -> bool:
    if field == "provider_eligibility":
        # Sheet coding varies ("2" vs "020" vs "023") - compare digit sets.
        return provider_eligibility_set(llm_value) == provider_eligibility_set(acro_value)
    if llm_value is None and acro_value is None:
        return True
    if llm_value is None or acro_value is None:
        return False
    try:
        return int(llm_value) == int(acro_value)
    except (ValueError, TypeError):
        return str(llm_value) == str(acro_value)


def verify_against_acroform(
    services_data: list[dict],
    correlation: dict,
) -> dict:
    """
    Compare extracted parent fields against correlated AcroForm hints.

    Returns dict with lists: verified_acroform, corrected, soft_flag,
    plus `corrections` entries the pipeline applies to the extraction
    ({service_index, field, corrected_value}).
    """
    out = {"verified_acroform": [], "corrected": [], "soft_flag": [],
           "corrections": []}

    # --- HCBS taxonomy: deterministic resolution oracle -------------------
    # Precedence: curated crosswalk (sheet vocabulary) > PDF taxonomy table
    # (when it maps to something more specific than "Other") > defer to the
    # LLM crosswalk inference (which runs after the critic).
    # Independent of AcroForm presence/confidence.
    taxonomies = correlation.get("taxonomies") or {}
    for idx, service in enumerate(services_data):
        parent = service.get("parent", {})
        name = parent.get("service_name")
        tax = taxonomies.get(name) or {}
        pdf_cats = [
            d.get("category") for d in tax.get("detail", []) if d.get("category")
        ]
        expected, source = resolve_taxonomy(name, pdf_cats)
        llm_value = parent.get("hcbs_taxonomy")

        if expected is None:
            if tax.get("block_found") and llm_value is not None:
                # Spec block exists but nothing deterministic supports the
                # LLM's value - null it and let the inference fallback
                # (sheet vocabulary, "Other" as last resort) decide.
                out["corrected"].append({
                    "service": name,
                    "service_index": idx,
                    "field": "hcbs_taxonomy",
                    "llm_value": llm_value,
                    "acroform_value": None,
                    "source": "taxonomy_resolution",
                    "conflict_type": "unsupported_taxonomy",
                })
                out["corrections"].append({
                    "service_index": idx,
                    "field": "hcbs_taxonomy",
                    "corrected_value": None,
                })
            continue

        entry = {
            "service": name,
            "service_index": idx,
            "field": "hcbs_taxonomy",
            "llm_value": llm_value,
            "acroform_value": expected,
            "source": source,
        }
        norm = lambda v: str(v).strip().lower() if v is not None else None
        if norm(llm_value) == norm(expected):
            out["verified_acroform"].append(entry)
        else:
            out["corrected"].append({**entry, "conflict_type": (
                "missing_value" if llm_value is None else "wrong_value"
            )})
            out["corrections"].append({
                "service_index": idx,
                "field": "hcbs_taxonomy",
                "corrected_value": expected,
            })

    hints = correlation.get("hints", {})
    if correlation.get("confidence") not in ("high", "medium"):
        return out

    for idx, service in enumerate(services_data):
        parent = service.get("parent", {})
        name = parent.get("service_name")
        hint = hints.get(name)
        if hint is None:
            continue

        checks = dict(ACROFORM_FIELDS)
        for field, hint_key in checks.items():
            acro_value = hint.get(hint_key)
            if acro_value is None and hint_key in (
                "self_direction_authority", "information_assistance",
            ):
                # Absence in the E-1 tables genuinely means "not applicable";
                # verify that the LLM also left it blank.
                pass
            llm_value = parent.get(field)
            entry = {
                "service": name,
                "service_index": idx,
                "field": field,
                "llm_value": llm_value,
                "acroform_value": acro_value,
            }
            if _values_match(field, llm_value, acro_value):
                out["verified_acroform"].append(entry)
            else:
                # AcroForm wins - correct the value.
                out["corrected"].append({**entry, "conflict_type": (
                    "missing_value" if llm_value is None else "wrong_value"
                )})
                out["corrections"].append({
                    "service_index": idx,
                    "field": field,
                    "corrected_value": acro_value,
                })

        # J-2d capitation per year (only when the AcroForm has the J tables)
        for year, cap_value in (hint.get("capitation") or {}).items():
            field = f"year_{year}_capitation"
            llm_value = parent.get(field)
            entry = {
                "service": name,
                "service_index": idx,
                "field": field,
                "llm_value": llm_value,
                "acroform_value": cap_value,
            }
            if _values_match(field, llm_value, cap_value):
                out["verified_acroform"].append(entry)
            else:
                out["corrected"].append({**entry, "conflict_type": (
                    "missing_value" if llm_value is None else "wrong_value"
                )})
                out["corrections"].append({
                    "service_index": idx,
                    "field": field,
                    "corrected_value": cap_value,
                })

    return out


# =============================================================================
# Step 2: deterministic math & consistency checks
# =============================================================================

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().upper() in ("N/A", "NA", ""):
            return None
        try:
            return float(value.replace(",", "").replace("$", ""))
        except ValueError:
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def validate_cost_math(service_data: dict) -> list[dict]:
    """
    Cost-math checks for one service:
    - child total costs must be "N/A"
    - parent component cost (users x avg units x avg cost) must be within
      tolerance of the extracted total cost (when the parent has no children)
    - with children: parent total must be within tolerance of the sum of
      child component costs
    """
    issues = []
    parent = service_data.get("parent", {})
    children = service_data.get("children", [])
    name = parent.get("service_name")

    for year in range(1, 6):
        cost_key = f"year_{year}_total_cost"
        parent_total = _as_float(parent.get(cost_key))

        for child in children:
            child_cost = child.get(cost_key)
            if child_cost is not None and str(child_cost).upper() != "N/A":
                issues.append({
                    "type": "child_should_have_na_cost",
                    "field": cost_key,
                    "service": child.get("service_name"),
                    "parent_service": name,
                    "value": child_cost,
                    "message": f"Child service should have 'N/A' for cost, got: {child_cost}",
                })

        if parent_total is None:
            continue

        if children:
            child_components = [
                _as_float(c.get(f"year_{year}_component_cost"))
                for c in children
            ]
            known = [c for c in child_components if c is not None]
            # Only meaningful when every child has computable components.
            if known and len(known) == len(children):
                total = sum(known)
                if total > 0 and abs(total - parent_total) / max(parent_total, 1) > COST_TOLERANCE:
                    issues.append({
                        "type": "parent_total_mismatch",
                        "field": cost_key,
                        "service": name,
                        "value": parent_total,
                        "expected": round(total, 2),
                        "message": (
                            f"Year {year}: parent total {parent_total} != "
                            f"sum of child component costs {round(total, 2)}"
                        ),
                    })
        else:
            component = _as_float(parent.get(f"year_{year}_component_cost"))
            if component is not None and component > 0:
                if abs(component - parent_total) / max(parent_total, 1) > COST_TOLERANCE:
                    issues.append({
                        "type": "component_cost_mismatch",
                        "field": cost_key,
                        "service": name,
                        "value": parent_total,
                        "expected": component,
                        "message": (
                            f"Year {year}: users x avg units x avg cost = {component} "
                            f"but extracted total cost is {parent_total}"
                        ),
                    })

    return issues


def validate_coded_fields(service_data: dict) -> list[dict]:
    """Range checks for coded fields on the parent row."""
    issues = []
    parent = service_data.get("parent", {})
    name = parent.get("service_name")

    ranges = {
        "service_type": (1, 4),
        "service_delivery_method": (1, 3),
        "provider_specifications": (1, 3),
        "self_direction_authority": (1, 3),
        "information_assistance": (1, 1),
    }
    for field, (lo, hi) in ranges.items():
        value = parent.get(field)
        if value is None:
            continue
        try:
            v = int(value)
            if v < lo or v > hi:
                raise ValueError
        except (ValueError, TypeError):
            issues.append({
                "type": f"invalid_{field}",
                "field": field,
                "service": name,
                "value": value,
                "message": f"{field} must be integer {lo}-{hi}, got: {value}",
            })

    for year in range(1, 6):
        cap = parent.get(f"year_{year}_capitation")
        if cap is not None and cap not in (0, 1, "0", "1"):
            issues.append({
                "type": "invalid_capitation",
                "field": f"year_{year}_capitation",
                "service": name,
                "value": cap,
                "message": f"Capitation must be 0 or 1, got: {cap}",
            })

    eligibility = parent.get("provider_eligibility")
    if eligibility is not None and eligibility != "":
        if not all(c in "0123" for c in str(eligibility)):
            issues.append({
                "type": "invalid_provider_eligibility",
                "field": "provider_eligibility",
                "service": name,
                "value": eligibility,
                "message": f"Provider eligibility must be digits 0-3, got: {eligibility}",
            })

    return issues


def validate_required_fields(service_data: dict) -> list[dict]:
    """Required-field checks for the parent row."""
    issues = []
    parent = service_data.get("parent", {})

    if not parent.get("service_name"):
        issues.append({
            "type": "missing_service_name",
            "field": "service_name",
            "message": "Service name is required",
        })

    has_cost = any(
        parent.get(f"year_{y}_total_cost") not in (None, "N/A")
        for y in range(1, 6)
    )
    if not has_cost:
        issues.append({
            "type": "no_cost_data",
            "field": "year_X_total_cost",
            "service": parent.get("service_name"),
            "message": "No cost data found for any year",
        })

    return issues


def validate_parent_child_consistency(service_data: dict) -> list[dict]:
    """Parent/child linkage checks."""
    issues = []
    parent = service_data.get("parent", {})
    children = service_data.get("children", [])
    parent_name = parent.get("service_name")

    for child in children:
        child_parent = child.get("parent_service")
        if child_parent and child_parent != parent_name:
            issues.append({
                "type": "parent_mismatch",
                "field": "parent_service",
                "service": child.get("service_name"),
                "value": child_parent,
                "expected": parent_name,
                "message": f"Child parent_service doesn't match: {child_parent} != {parent_name}",
            })
        if child.get("row_type") != "child":
            issues.append({
                "type": "invalid_row_type",
                "field": "row_type",
                "service": child.get("service_name"),
                "value": child.get("row_type"),
                "message": f"Child should have row_type='child', got: {child.get('row_type')}",
            })

    if parent.get("row_type") != "parent":
        issues.append({
            "type": "invalid_row_type",
            "field": "row_type",
            "service": parent_name,
            "value": parent.get("row_type"),
            "message": f"Parent should have row_type='parent', got: {parent.get('row_type')}",
        })

    return issues


# Issue types that flag a field for re-extraction (vs. informational).
_SOFT_FLAG_ISSUE_TYPES = {
    "component_cost_mismatch",
    "parent_total_mismatch",
    "child_should_have_na_cost",
    "no_cost_data",
    "invalid_service_type",
    "invalid_service_delivery_method",
    "invalid_provider_specifications",
    "invalid_self_direction_authority",
    "invalid_information_assistance",
    "invalid_capitation",
    "invalid_provider_eligibility",
}
_CRITICAL_ISSUE_TYPES = {"missing_service_name", "no_cost_data"}


# =============================================================================
# Step 3: LLM body-text verification
# =============================================================================

def _build_body_text_payload(services_data: list[dict]) -> list[dict]:
    """Collect (service, field, value, evidence) tuples for LLM verification."""
    payload = []
    for idx, service in enumerate(services_data):
        parent = service.get("parent", {})
        evidence = parent.get("evidence") or {}
        for field in BODY_TEXT_FIELDS:
            value = parent.get(field)
            if value is None:
                continue  # nothing claimed, nothing to verify
            payload.append({
                "service_index": idx,
                "service": parent.get("service_name"),
                "field": field,
                "value": value,
                "evidence": evidence.get(field) or evidence.get("source"),
            })
    return payload


def _verify_body_text_fields(
    services_data: list[dict],
    waiver_key: str | None = None,
) -> dict:
    """
    LLM check: does the captured evidence support each extracted value?

    One model call per waiver. On any model failure, all checked fields
    become soft_flag (same failure posture as the overview critic).
    """
    out = {"verified_body": [], "soft_flag": []}
    payload = _build_body_text_payload(services_data)
    if not payload:
        return out

    items = [
        {
            "id": i,
            "service": p["service"],
            "field": p["field"],
            "extracted_value": p["value"],
            "evidence": p["evidence"],
        }
        for i, p in enumerate(payload)
    ]

    user_prompt = (
        "For each item, decide if the evidence supports the extracted value.\n"
        "Return ONLY a JSON array like "
        '[{"id": 0, "supported": true, "confidence": "high", "reason": "..."}].\n\n'
        + json.dumps(items, indent=1, default=str)
    )

    def _make_request():
        client = _get_client()
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part(text=BODY_TEXT_CRITIC_SYSTEM + "\n\n" + user_prompt)
            ])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16384,
            ),
        )

    try:
        # Transient errors (rate limits, DNS blips) get retried here rather
        # than falling straight to the except below - previously *any*
        # failure on this single call flagged every field it was checking
        # as "unsupported", even when the value was fine and the call just
        # needed a retry.
        response = _retry_on_connection_error(_make_request)
        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        verdicts = {v["id"]: v for v in json.loads(text)}
    except Exception as e:
        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_critic",
                "action": "body_text_verification_error",
                "error": str(e),
            })
        for p in payload:
            out["soft_flag"].append({
                "service": p["service"],
                "service_index": p["service_index"],
                "field": p["field"],
                "llm_value": p["value"],
                "reason": f"body-text verification failed: {e}",
            })
        return out

    for i, p in enumerate(payload):
        verdict = verdicts.get(i, {})
        supported = verdict.get("supported", False)
        confidence = verdict.get("confidence", "low")
        entry = {
            "service": p["service"],
            "service_index": p["service_index"],
            "field": p["field"],
            "llm_value": p["value"],
            "confidence": confidence,
            "reason": verdict.get("reason"),
        }
        if supported and confidence in ("high", "medium"):
            out["verified_body"].append(entry)
        else:
            entry["reason"] = entry["reason"] or "evidence does not support value"
            out["soft_flag"].append(entry)

    return out


# =============================================================================
# Main entry
# =============================================================================

def run_services_critic(
    pdf_path: str,
    services_data: list[dict],
    waiver_key: str | None = None,
    correlation: dict | None = None,
    attempt: int = 1,
    verify_body_text: bool = True,
) -> dict:
    """
    Run the services critic on extracted service data.

    Args:
        pdf_path: Path to the PDF file
        services_data: List of service dicts with 'parent' and 'children'
        waiver_key: Waiver key for logging and human overrides
        correlation: Output of correlate_services() (AcroForm hints); None
            disables AcroForm verification
        attempt: Which extraction attempt this critique follows (1-based)
        verify_body_text: Set False to skip the LLM verification step

    Returns:
        Critique dict with buckets, corrections, issues, accuracy, decision.
    """
    correlation = correlation or {"confidence": "none", "hints": {}}

    # Step 1: AcroForm verification
    acro = verify_against_acroform(services_data, correlation)

    # Step 2: deterministic checks
    all_issues = []
    for service in services_data:
        all_issues.extend(validate_cost_math(service))
        all_issues.extend(validate_coded_fields(service))
        all_issues.extend(validate_required_fields(service))
        all_issues.extend(validate_parent_child_consistency(service))

    issue_soft_flags = []
    name_to_index = {
        s.get("parent", {}).get("service_name"): i
        for i, s in enumerate(services_data)
    }
    for issue in all_issues:
        if issue.get("type") in _SOFT_FLAG_ISSUE_TYPES:
            issue_soft_flags.append({
                "service": issue.get("service") or issue.get("parent_service"),
                "service_index": name_to_index.get(
                    issue.get("parent_service") or issue.get("service")
                ),
                "field": issue.get("field"),
                "llm_value": issue.get("value"),
                "reason": issue.get("message"),
                # Deterministic checks (cost math) already compute the
                # correct value; carry it through so it can be applied
                # directly instead of re-derived by the LLM.
                "expected": issue.get("expected"),
                "issue_type": issue.get("type"),
            })

    # Step 3: LLM body-text verification
    if verify_body_text:
        body = _verify_body_text_fields(services_data, waiver_key)
    else:
        body = {"verified_body": [], "soft_flag": []}

    soft_flag = acro["soft_flag"] + issue_soft_flags + body["soft_flag"]

    # De-duplicate soft flags per (service, field)
    seen = set()
    deduped_soft = []
    for entry in soft_flag:
        key = (entry.get("service"), entry.get("field"))
        if key not in seen:
            seen.add(key)
            deduped_soft.append(entry)
    soft_flag = deduped_soft

    critical_issues = [i for i in all_issues if i.get("type") in _CRITICAL_ISSUE_TYPES]

    # Accuracy over categorized fields
    verified_count = (
        len(acro["verified_acroform"])
        + len(body["verified_body"])
        + len(acro["corrected"])
    )
    categorized = verified_count + len(soft_flag)
    accuracy = (verified_count / categorized) if categorized else None

    # Decision
    human_queue = []
    if not soft_flag and not critical_issues:
        decision = "pass"
    elif attempt < MAX_ATTEMPTS:
        decision = "attempt2"
    else:
        decision = "human"
        human_queue = soft_flag
        if waiver_key:
            for entry in human_queue:
                _write_services_human_override(
                    waiver_key=waiver_key,
                    service_name=entry.get("service"),
                    service_index=entry.get("service_index"),
                    field_name=entry.get("field"),
                    acroform_value=entry.get("acroform_value"),
                    llm_value=entry.get("llm_value"),
                    conflict_type="unresolved_soft_flag",
                    attempt_reached=attempt,
                )

    # Audit trail: every AcroForm correction is recorded
    if waiver_key:
        for entry in acro["corrected"]:
            _write_services_human_override(
                waiver_key=waiver_key,
                service_name=entry.get("service"),
                service_index=entry.get("service_index"),
                field_name=entry.get("field"),
                acroform_value=entry.get("acroform_value"),
                llm_value=entry.get("llm_value"),
                conflict_type=entry.get("conflict_type", "wrong_value"),
                attempt_reached=attempt,
            )

    result = {
        "passed": decision == "pass",
        "decision": decision,
        "attempt": attempt,
        "accuracy": accuracy,
        "acroform_confidence": correlation.get("confidence"),
        "verified_acroform": acro["verified_acroform"],
        "corrected": acro["corrected"],
        "corrections": acro["corrections"],
        "verified_body": body["verified_body"],
        "soft_flag": soft_flag,
        "human_queue": human_queue,
        "issues": all_issues,
        "critical_count": len(critical_issues),
        "services_validated": len(services_data),
    }

    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_critic",
            "action": "critique_complete",
            "attempt": attempt,
            "decision": decision,
            "accuracy": accuracy,
            "verified_acroform": len(acro["verified_acroform"]),
            "corrected": len(acro["corrected"]),
            "verified_body": len(body["verified_body"]),
            "soft_flag": len(soft_flag),
            "human_queue": len(human_queue),
            "total_issues": len(all_issues),
        })

    return result


def apply_corrections(services_data: list[dict], corrections: list[dict]) -> int:
    """
    Apply AcroForm corrections to the extraction in place (AcroForm wins).

    Returns the number of fields corrected.
    """
    applied = 0
    for corr in corrections:
        idx = corr.get("service_index")
        field = corr.get("field")
        if idx is None or field is None or idx >= len(services_data):
            continue
        parent = services_data[idx].setdefault("parent", {})
        parent[field] = corr.get("corrected_value")
        applied += 1
    return applied
