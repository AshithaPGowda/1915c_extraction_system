"""
AcroForm verification tools for the critic agent.
"""

from typing import Any

from ...parsers import AcroFormTree, get_acroform_path, SCHEMA_TO_ACROFORM


def get_acroform_tree(pdf_path: str) -> dict:
    """Get the AcroForm structure from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        dict with AcroForm info or error
    """
    try:
        tree = AcroFormTree(pdf_path)
        return {
            "status": "success",
            "has_acroform": tree.has_acroform,
            "field_count": len(tree.flat_fields) if tree.has_acroform else 0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_acroform_value(pdf_path: str, field_name: str) -> dict:
    """Get the AcroForm value for a specific schema field.

    Args:
        pdf_path: Path to the PDF file
        field_name: Schema field name (e.g., "self_direction_budget_authority")

    Returns:
        dict with acroform_value, acroform_path, or null if not mapped
    """
    try:
        tree = AcroFormTree(pdf_path)

        if not tree.has_acroform:
            return {"status": "no_acroform", "value": None}

        acroform_path = get_acroform_path(field_name)
        if not acroform_path:
            return {"status": "not_mapped", "value": None, "field": field_name}

        value = tree.get_value(acroform_path)
        return {
            "status": "success",
            "field": field_name,
            "acroform_path": acroform_path,
            "value": value,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def verify_field_against_acroform(
    pdf_path: str,
    field_name: str,
    extracted_value: Any
) -> dict:
    """Verify an extracted value against AcroForm metadata.

    Args:
        pdf_path: Path to the PDF file
        field_name: Schema field name
        extracted_value: The value extracted by the LLM

    Returns:
        dict with verification result (match, mismatch, no_acroform, not_mapped)
    """
    acroform_result = get_acroform_value(pdf_path, field_name)

    if acroform_result["status"] != "success":
        return {
            "field": field_name,
            "verification": acroform_result["status"],
            "extracted_value": extracted_value,
            "acroform_value": None,
        }

    acroform_value = acroform_result["value"]

    # Normalize for comparison
    def normalize(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("true", "yes", "on", "1"):
                return 1
            if v_lower in ("false", "no", "off", "0", ""):
                return 0
            return v_lower
        return v

    norm_extracted = normalize(extracted_value)
    norm_acroform = normalize(acroform_value)

    if norm_extracted == norm_acroform:
        return {
            "field": field_name,
            "verification": "match",
            "extracted_value": extracted_value,
            "acroform_value": acroform_value,
        }
    else:
        return {
            "field": field_name,
            "verification": "mismatch",
            "extracted_value": extracted_value,
            "acroform_value": acroform_value,
            "acroform_path": acroform_result.get("acroform_path"),
        }


def get_all_acroform_mappings() -> dict:
    """Get all schema-to-AcroForm field mappings.

    Returns:
        dict with all mappings
    """
    return {
        "status": "success",
        "mapping_count": len(SCHEMA_TO_ACROFORM),
        "mappings": SCHEMA_TO_ACROFORM,
    }


def verify_all_fields(waiver_key: str, attempt: int = 1) -> dict:
    """Verify ALL extracted fields against AcroForm metadata at once.

    This tool does the mechanical comparison for all fields and returns
    categorized buckets. The agent can then analyze results and provide
    explanations and recommendations.

    Args:
        waiver_key: The waiver identifier (e.g., "IA0242R0702")
        attempt: Current attempt number (1, 2, or 3)

    Returns:
        dict with:
            - verified_acroform: Fields matching AcroForm
            - verified_body: Non-AcroForm fields (assumed correct)
            - corrected: Fields corrected by AcroForm
            - soft_flag: Fields with conflicts needing review
            - missing_child: Expected fields not extracted
            - hallucinated: Fields that shouldn't have values
            - human_queue: Fields needing human review
            - accuracy: Percentage of correct fields
            - decision: "pass", "attempt2", "attempt3", or "human"
            - summary: Quick stats for the agent to analyze
    """
    from .stage_storage_tools import load_stage_output, load_acroform_cache

    # Load extracted data
    extraction_result = load_stage_output(waiver_key, f"extraction_attempt_{attempt}")
    if extraction_result.get("status") != "success":
        # Try previous attempts
        for prev_attempt in range(attempt - 1, 0, -1):
            extraction_result = load_stage_output(waiver_key, f"extraction_attempt_{prev_attempt}")
            if extraction_result.get("status") == "success":
                break
        else:
            return {
                "status": "error",
                "error": f"No extraction data found for {waiver_key}",
            }

    extracted_data = extraction_result.get("data", {})

    # Load AcroForm cache
    acroform_result = load_acroform_cache(waiver_key)
    if acroform_result.get("status") != "success":
        return {
            "status": "error",
            "error": f"No AcroForm cache found for {waiver_key}. Run extract_and_cache_acroform first.",
        }

    acroform_data = acroform_result.get("acroform", {})
    checkbox_values = acroform_data.get("checkbox_values", {})

    # Initialize buckets
    verified_acroform = []
    verified_body = []
    corrected = []
    soft_flag = []
    missing_child = []
    hallucinated = []
    human_queue = []

    # Helper to normalize values for comparison
    def normalize(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("true", "yes", "on", "1"):
                return 1
            if v_lower in ("false", "no", "off", "0", ""):
                return 0
            return v_lower
        return v

    # Helper to get page and evidence for a field
    def get_field_context(field_name: str) -> dict:
        page = extracted_data.get(f"{field_name}_page")
        evidence = extracted_data.get(f"{field_name}_evidence")
        return {
            "page": page,
            "evidence": evidence[:200] if evidence and len(evidence) > 200 else evidence,
        }

    # Track processed fields
    processed_fields = set()

    # 1. Verify AcroForm-mapped fields
    for field_name, acroform_path in SCHEMA_TO_ACROFORM.items():
        processed_fields.add(field_name)

        extracted_value = extracted_data.get(field_name)
        acroform_value = checkbox_values.get(field_name)

        # Get page and evidence context
        context = get_field_context(field_name)

        # Case: AcroForm has no value (unresolved checkbox)
        if acroform_value is None:
            if extracted_value is None:
                # Both missing - could be legitimately empty or missing_child
                missing_child.append({
                    "field": field_name,
                    "acroform_path": acroform_path,
                    "reason": "Both AcroForm and extraction returned None",
                    **context,
                })
            else:
                # AcroForm couldn't resolve, but LLM extracted something
                soft_flag.append({
                    "field": field_name,
                    "llm_value": extracted_value,
                    "acroform_value": None,
                    "conflict_type": "acroform_unresolved",
                    "reason": "AcroForm couldn't resolve this checkbox, LLM extracted a value. Needs visual OCR.",
                    **context,
                })
            continue

        # Case: Compare values
        norm_extracted = normalize(extracted_value)
        norm_acroform = normalize(acroform_value)

        if norm_extracted == norm_acroform:
            # Match!
            verified_acroform.append({
                "field": field_name,
                "value": extracted_value,
                "acroform_value": acroform_value,
                "source": "acroform",
            })
        elif extracted_value is None:
            # LLM missed it, AcroForm has it
            missing_child.append({
                "field": field_name,
                "acroform_path": acroform_path,
                "expected_value": acroform_value,
                "reason": "LLM returned None but AcroForm has a value. Check if field exists on expected page.",
                **context,
            })
        elif acroform_value in (0, False, "0", "false", "no"):
            # AcroForm says OFF but LLM said ON - hallucination
            hallucinated.append({
                "field": field_name,
                "hallucinated_value": extracted_value,
                "acroform_value": acroform_value,
                "reason": "AcroForm definitively OFF, LLM hallucinated ON. May need visual confirmation.",
                **context,
            })
        else:
            # Mismatch - AcroForm wins, correct it
            corrected.append({
                "field": field_name,
                "extracted_value": extracted_value,
                "corrected_value": acroform_value,
                "acroform_path": acroform_path,
                **context,
            })

    # 2. Process non-AcroForm fields (body text fields)
    for field_name, value in extracted_data.items():
        # Skip metadata fields
        if field_name.endswith(("_evidence", "_page", "_confidence")):
            continue
        if field_name.startswith("_"):
            continue
        if field_name in processed_fields:
            continue

        # Non-AcroForm field - assume correct if has value
        if value is not None and value != "":
            verified_body.append({
                "field": field_name,
                "value": value,
                "source": "body_text",
            })

    # 3. Calculate accuracy
    total_verified = len(verified_acroform) + len(verified_body) + len(corrected)
    total_fields = (
        total_verified +
        len(soft_flag) +
        len(hallucinated) +
        len(missing_child)
    )

    accuracy = total_verified / total_fields if total_fields > 0 else 1.0

    # 4. Determine decision
    has_issues = (
        len(soft_flag) > 0 or
        len(missing_child) > 0 or
        len(hallucinated) > 0
    )

    if not has_issues and accuracy >= 0.85:
        decision = "pass"
    elif attempt == 1:
        decision = "attempt2"
    elif attempt == 2:
        decision = "attempt3"
    else:
        decision = "human"
        # Move remaining issues to human_queue
        for item in soft_flag + missing_child:
            human_queue.append(item)

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "attempt": attempt,
        "verified_acroform": verified_acroform,
        "verified_body": verified_body,
        "corrected": corrected,
        "soft_flag": soft_flag,
        "missing_child": missing_child,
        "hallucinated": hallucinated,
        "human_queue": human_queue,
        "accuracy": round(accuracy, 4),
        "decision": decision,
        "summary": {
            "total_fields": total_fields,
            "verified_count": total_verified,
            "corrected_count": len(corrected),
            "soft_flag_count": len(soft_flag),
            "missing_count": len(missing_child),
            "hallucinated_count": len(hallucinated),
            "needs_reextraction": len(soft_flag) + len(missing_child),
        },
    }
