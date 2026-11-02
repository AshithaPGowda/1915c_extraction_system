"""
Error analysis tools for understanding extraction patterns.

These tools analyze human_overrides.jsonl to find error patterns
and help the agent learn from past mistakes.
"""

import json

from .constants import DATA_DIR, OUTPUT_DIR, LOGS_DIR
from waiver_level_extraction_agent.shared_libraries.file_utils import load_layout


def _dedupe_overrides(records: list) -> dict:
    """Deduplicate human_overrides records by (waiver, field).

    Keeps only the latest entry per (waiver_number, field_name) combination.

    Returns:
        dict mapping (waiver_number, field_name) -> record
    """
    unique = {}
    for r in records:
        key = (r.get("waiver_number"), r.get("field_name"))
        if key not in unique or r.get("timestamp", "") > unique[key].get("timestamp", ""):
            unique[key] = r
    return unique


def get_common_extraction_errors() -> dict:
    """Analyze human overrides to find common extraction error patterns."""
    try:
        overrides_path = DATA_DIR / "human_overrides.jsonl"

        if not overrides_path.exists():
            return {"status": "success", "patterns": [], "message": "No errors recorded"}

        records = []
        with open(overrides_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Deduplicate by (waiver_number, field_name)
        unique = _dedupe_overrides(records)

        field_errors = {}
        conflict_counts = {}

        for record in unique.values():
            field = record.get("field_name", "unknown")
            conflict = record.get("conflict_type", "unknown")

            field_errors[field] = field_errors.get(field, 0) + 1
            conflict_counts[conflict] = conflict_counts.get(conflict, 0) + 1

        top_error_fields = sorted(field_errors.items(), key=lambda x: -x[1])[:20]
        top_conflict_types = sorted(conflict_counts.items(), key=lambda x: -x[1])

        return {
            "status": "success",
            "total_errors": len(unique),
            "top_error_fields": [{"field": f, "count": c} for f, c in top_error_fields],
            "conflict_type_breakdown": [{"type": t, "count": c} for t, c in top_conflict_types],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_errors_by_waiver(limit: int = 20) -> dict:
    """Get error counts grouped by waiver - find most problematic PDFs."""
    try:
        overrides_path = DATA_DIR / "human_overrides.jsonl"

        if not overrides_path.exists():
            return {"status": "success", "waivers": []}

        records = []
        with open(overrides_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Deduplicate by (waiver_number, field_name)
        unique = _dedupe_overrides(records)

        waiver_errors = {}
        waiver_conflict_types = {}

        for record in unique.values():
            waiver_num = record.get("waiver_number", "unknown")
            waiver_key = waiver_num.replace(".", "")
            conflict = record.get("conflict_type", "unknown")

            waiver_errors[waiver_key] = waiver_errors.get(waiver_key, 0) + 1

            if waiver_key not in waiver_conflict_types:
                waiver_conflict_types[waiver_key] = {}
            waiver_conflict_types[waiver_key][conflict] = \
                waiver_conflict_types[waiver_key].get(conflict, 0) + 1

        sorted_waivers = sorted(waiver_errors.items(), key=lambda x: -x[1])[:limit]

        result = []
        for waiver_key, count in sorted_waivers:
            result.append({
                "waiver_key": waiver_key,
                "error_count": count,
                "conflict_types": waiver_conflict_types.get(waiver_key, {}),
            })

        return {
            "status": "success",
            "total_waivers_with_errors": len(waiver_errors),
            "most_problematic": result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_error_classification() -> dict:
    """Classify all errors into categories."""
    try:
        overrides_path = DATA_DIR / "human_overrides.jsonl"

        if not overrides_path.exists():
            return {"status": "success", "classifications": {}}

        records = []
        with open(overrides_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Deduplicate by (waiver_number, field_name)
        unique = _dedupe_overrides(records)

        classifications = {
            "acroform_mismatch": [],
            "missing_value": [],
            "wrong_value": [],
            "hallucination": [],
            "other": [],
        }

        for record in unique.values():
            conflict = record.get("conflict_type", "")
            field = record.get("field_name", "unknown")
            acroform_val = record.get("acroform_value")
            llm_val = record.get("llm_value")

            if "mismatch" in conflict.lower() or (acroform_val is not None and llm_val != acroform_val):
                classifications["acroform_mismatch"].append(field)
            elif llm_val is None and acroform_val is not None:
                classifications["missing_value"].append(field)
            elif conflict == "wrong_value":
                classifications["wrong_value"].append(field)
            elif conflict == "hallucination" or (acroform_val is None and llm_val is not None):
                classifications["hallucination"].append(field)
            else:
                classifications["other"].append(field)

        summary = {}
        for cat, fields in classifications.items():
            if fields:
                field_counts = {}
                for f in fields:
                    field_counts[f] = field_counts.get(f, 0) + 1
                top_fields = sorted(field_counts.items(), key=lambda x: -x[1])[:5]
                summary[cat] = {
                    "total": len(fields),
                    "top_fields": [{"field": f, "count": c} for f, c in top_fields],
                }

        return {"status": "success", "classifications": summary}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_field_error_breakdown(field_name: str) -> dict:
    """Get deduplicated error breakdown for a specific field.

    Args:
        field_name: Field to analyze (supports partial match)

    Returns:
        dict with:
        - raw_count: records in file (may have duplicates)
        - unique_count: actual unique waivers affected
        - breakdown: count by conflict_type
        - flagged_for_review: count needing human decision
        - sample_waivers: first 10 affected waiver keys
    """
    overrides_path = DATA_DIR / "human_overrides.jsonl"

    if not overrides_path.exists():
        return {"status": "error", "error": "No overrides file"}

    records = []
    with open(overrides_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter to matching field
    matching = [r for r in records if field_name.lower() in r.get("field_name", "").lower()]

    # Deduplicate
    unique = _dedupe_overrides(matching)

    # Count by conflict type
    breakdown = {}
    flagged = 0
    sample_waivers = []
    seen_waivers = set()

    for (waiver, field), r in unique.items():
        conflict = r.get("conflict_type", "unknown")
        breakdown[conflict] = breakdown.get(conflict, 0) + 1

        if r.get("human_decision") is None:
            flagged += 1

        waiver_key = waiver.replace(".", "")
        if len(sample_waivers) < 10 and waiver_key not in seen_waivers:
            sample_waivers.append(waiver_key)
            seen_waivers.add(waiver_key)

    return {
        "status": "success",
        "field_name": field_name,
        "raw_count": len(matching),
        "unique_count": len(unique),
        "breakdown": breakdown,
        "flagged_for_review": flagged,
        "sample_waivers": sample_waivers,
    }


def get_extraction_prompt() -> dict:
    """Get the extraction prompt/instructions used for waiver overview extraction."""
    try:
        from waiver_level_extraction_agent.sub_agents.overview_prompt import OVERVIEW_AGENT_PROMPT

        layout = load_layout("layout_waiver_overview")
        # Layout is a dict: field_name -> {Header, isInTable, context}
        field_names = list(layout.keys())

        return {
            "status": "success",
            "prompt_length": len(OVERVIEW_AGENT_PROMPT),
            "prompt_preview": OVERVIEW_AGENT_PROMPT[:2000] + "..." if len(OVERVIEW_AGENT_PROMPT) > 2000 else OVERVIEW_AGENT_PROMPT,
            "total_fields_to_extract": len(field_names),
            "field_names": field_names,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_field_schema(field_name: str) -> dict:
    """Get the schema definition for a specific field."""
    try:
        layout = load_layout("layout_waiver_overview")
        # Layout is a dict: field_name -> {Header, isInTable, context}

        # Exact match
        if field_name in layout:
            field_info = layout[field_name]
            return {
                "status": "success",
                "field_name": field_name,
                "field": {
                    "name": field_name,
                    "header": field_info.get("Header", ""),
                    "context": field_info.get("context", ""),
                    "is_in_table": field_info.get("isInTable", False),
                },
            }

        # Partial match
        matches = [name for name in layout.keys() if field_name.lower() in name.lower()]
        if matches:
            return {
                "status": "partial_match",
                "searched_for": field_name,
                "matches": matches[:5],
            }

        return {
            "status": "not_found",
            "field_name": field_name,
            "available_fields": list(layout.keys())[:20],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_field_evidence(waiver_key: str, field_name: str) -> dict:
    """Read the evidence chain for a specific field from a waiver's output.

    This shows the actual extracted evidence/snippet that led to the value.
    Use this to understand WHY the LLM extracted a particular value.

    Args:
        waiver_key: The waiver identifier (e.g., "AK0260R0608")
        field_name: The field to look up (can be partial match)

    Returns:
        dict with evidence data including snippets
    """
    try:
        # Try to find evidence file
        evidence_path = OUTPUT_DIR / waiver_key / f"{waiver_key}_evidence.json"

        if not evidence_path.exists():
            # Try alternate name
            evidence_path = OUTPUT_DIR / waiver_key / "evidence_chain.json"

        if not evidence_path.exists():
            return {
                "status": "not_found",
                "error": f"No evidence file found for {waiver_key}",
            }

        with open(evidence_path) as f:
            data = json.load(f)

        fields = data.get("fields", {})

        # Find matching field
        matched_field = None
        matched_data = None

        for fname, fdata in fields.items():
            if fname == field_name:
                matched_field = fname
                matched_data = fdata
                break
            elif field_name.lower() in fname.lower():
                matched_field = fname
                matched_data = fdata
                break

        if not matched_data:
            return {
                "status": "field_not_found",
                "waiver_key": waiver_key,
                "field_name": field_name,
                "available_fields": list(fields.keys())[:20],
            }

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "field_name": matched_field,
            "evidence": matched_data,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_evidence_for_errors(field_name: str, limit: int = 10) -> dict:
    """Get evidence snippets from waivers where a field had errors.

    This is the KEY tool for understanding WHY extractions fail.
    It reads the actual evidence/snippets that led to wrong values.

    Args:
        field_name: The field to investigate (can be partial match)
        limit: Max number of examples to return

    Returns:
        dict with evidence from affected waivers showing what LLM saw
    """
    try:
        # First, find waivers with errors for this field
        overrides_path = DATA_DIR / "human_overrides.jsonl"

        if not overrides_path.exists():
            return {"status": "success", "message": "No errors recorded"}

        records = []
        with open(overrides_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Filter to matching field
        matching_records = [
            r for r in records if field_name.lower() in r.get("field_name", "").lower()
        ]

        # Deduplicate by (waiver_number, field_name)
        unique = _dedupe_overrides(matching_records)

        affected_waivers = []
        for record in unique.values():
            waiver_num = record.get("waiver_number", "").replace(".", "")
            if waiver_num:
                affected_waivers.append({
                    "waiver_key": waiver_num,
                    "llm_value": record.get("llm_value"),
                    "correct_value": record.get("acroform_value") or record.get("corrected_value"),
                    "conflict_type": record.get("conflict_type"),
                })
                if len(affected_waivers) >= limit:
                    break

        if not affected_waivers:
            return {"status": "success", "message": "No errors found for this field"}

        # Now get evidence for each affected waiver
        evidence_samples = []

        for waiver_info in affected_waivers:
            waiver_key = waiver_info["waiver_key"]
            evidence_result = get_field_evidence(waiver_key, field_name)

            if evidence_result.get("status") == "success":
                evidence_data = evidence_result.get("evidence", {})

                sample = {
                    "waiver_key": waiver_key,
                    "llm_extracted": waiver_info["llm_value"],
                    "should_be": waiver_info["correct_value"],
                    "final_value_in_output": evidence_data.get("final_value"),
                    "attempt_1_evidence": evidence_data.get("attempt_1", {}).get("evidence"),
                    "attempt_2_verbatim": evidence_data.get("attempt_2", {}).get("verbatim_source"),
                    "resolution": evidence_data.get("resolution"),
                }
                evidence_samples.append(sample)

        return {
            "status": "success",
            "field_name": field_name,
            "error_count": len(affected_waivers),
            "evidence_samples": evidence_samples,
            "analysis_hint": "Look at 'attempt_1_evidence' and 'attempt_2_verbatim' to see exactly what the LLM saw and interpreted.",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def investigate_field_errors(field_name: str) -> dict:
    """Deep investigation of why a specific field is causing errors.

    This is a COMPREHENSIVE tool that gathers all relevant data about
    a field's extraction errors in one call. Use this when someone asks
    "Why is field X failing?" or "What's causing errors in field Y?"

    Returns:
        - Field schema/definition
        - Error count and types for this field
        - Sample of actual corrections made (LLM value vs corrected value)
        - Which waivers have this error
        - Root cause analysis hints

    Args:
        field_name: The field to investigate (can be partial match)
    """
    try:
        results = {
            "status": "success",
            "field_name": field_name,
            "investigation": {},
        }

        # 1. Get field schema
        # Layout is a dict: field_name -> {Header, isInTable, context}
        layout = load_layout("layout_waiver_overview")
        field_schema = None
        matched_field_name = field_name

        # Exact match first
        if field_name in layout:
            field_info = layout[field_name]
            field_schema = {
                "name": field_name,
                "header": field_info.get("Header", ""),
                "context": field_info.get("context", ""),
                "is_in_table": field_info.get("isInTable", False),
            }
        else:
            # Partial match
            for name, info in layout.items():
                if field_name.lower() in name.lower():
                    field_schema = {
                        "name": name,
                        "header": info.get("Header", ""),
                        "context": info.get("context", ""),
                        "is_in_table": info.get("isInTable", False),
                    }
                    matched_field_name = name
                    break

        results["investigation"]["schema"] = field_schema
        results["investigation"]["matched_field_name"] = matched_field_name

        # 2. Analyze human overrides for this field
        overrides_path = DATA_DIR / "human_overrides.jsonl"
        field_overrides = []
        value_patterns = {"llm_values": {}, "corrected_values": {}}
        affected_waivers = set()

        if overrides_path.exists():
            records = []
            with open(overrides_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            # Filter to matching field first
            matching_records = [
                r for r in records
                if r.get("field_name", "") == matched_field_name
                or field_name.lower() in r.get("field_name", "").lower()
            ]

            # Deduplicate by (waiver_number, field_name)
            unique = _dedupe_overrides(matching_records)

            for record in unique.values():
                field_overrides.append(record)
                affected_waivers.add(record.get("waiver_number", "").replace(".", ""))

                # Track value patterns
                llm_val = str(record.get("llm_value", "None"))
                corr_val = str(record.get("acroform_value") or record.get("corrected_value", "None"))

                value_patterns["llm_values"][llm_val] = value_patterns["llm_values"].get(llm_val, 0) + 1
                value_patterns["corrected_values"][corr_val] = value_patterns["corrected_values"].get(corr_val, 0) + 1

        results["investigation"]["error_count"] = len(field_overrides)
        results["investigation"]["affected_waivers"] = list(affected_waivers)[:20]
        results["investigation"]["affected_waiver_count"] = len(affected_waivers)

        # 3. Analyze the correction patterns
        if field_overrides:
            # Most common LLM mistakes
            sorted_llm = sorted(value_patterns["llm_values"].items(), key=lambda x: -x[1])
            sorted_corr = sorted(value_patterns["corrected_values"].items(), key=lambda x: -x[1])

            results["investigation"]["value_analysis"] = {
                "llm_commonly_extracts": sorted_llm[:5],
                "should_actually_be": sorted_corr[:5],
            }

            # Sample corrections for context
            results["investigation"]["sample_corrections"] = [
                {
                    "waiver": r.get("waiver_number"),
                    "llm_extracted": r.get("llm_value"),
                    "correct_value": r.get("acroform_value") or r.get("corrected_value"),
                    "conflict_type": r.get("conflict_type"),
                }
                for r in field_overrides[:10]
            ]

            # 4. Generate root cause hints
            hints = []

            # Check for checkbox inversion (LLM says 1 but should be 0, or vice versa)
            if value_patterns["llm_values"].get("1", 0) > 0 and value_patterns["corrected_values"].get("0", 0) > 0:
                hints.append("CHECKBOX INVERSION: LLM extracts 1 when it should be 0. Check if checkbox logic is inverted.")
            if value_patterns["llm_values"].get("0", 0) > 0 and value_patterns["corrected_values"].get("1", 0) > 0:
                hints.append("CHECKBOX INVERSION: LLM extracts 0 when it should be 1. Check if checkbox logic is inverted.")

            # Check for None/missing issues
            if value_patterns["llm_values"].get("None", 0) > len(field_overrides) * 0.3:
                hints.append("MISSING VALUE: LLM frequently returns None. Field may be hard to locate in PDF.")

            # Check for hallucination
            if value_patterns["corrected_values"].get("None", 0) > len(field_overrides) * 0.3:
                hints.append("HALLUCINATION: LLM extracts values that should be None. It may be finding wrong data.")

            # Check field type from schema
            if field_schema:
                field_type = field_schema.get("type", "")
                if field_type == "boolean" or "0 = No" in str(field_schema):
                    hints.append(f"BOOLEAN FIELD: This is a checkbox/boolean. Schema expects 0 or 1.")

            results["investigation"]["root_cause_hints"] = hints if hints else ["No obvious pattern detected. Review sample corrections for clues."]

        else:
            results["investigation"]["message"] = "No errors found for this field in human_overrides.jsonl"

        return results

    except Exception as e:
        return {"status": "error", "error": str(e)}


def find_empty_fields(field_name: str, limit: int = 20) -> dict:
    """Find waivers where a specific field is empty or null.

    Use this to answer questions like "Which waivers have empty 1.C values?"
    or "Show me all PDFs where a specific column is missing".

    Args:
        field_name: The field/column name to check (supports partial matching).
                    If multiple columns match, searches all of them.
        limit: Maximum number of results to return per column (default 20)

    Returns:
        dict with list of waivers having empty values for this field.
        If multiple columns match, returns results for all matching columns.
    """
    import csv

    from waiver_level_extraction_agent.shared_libraries.paths import (
        VERIFIED_WAIVERS_CSV, PDF_DIR as _PDF_DIR,
    )
    CSV_PATH = VERIFIED_WAIVERS_CSV  # empty-field analysis runs over the verified master
    PDF_DIRS = [_PDF_DIR]

    if not CSV_PATH.exists():
        return {"status": "error", "error": f"CSV not found at {CSV_PATH}"}

    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            if not headers:
                return {"status": "error", "error": "CSV has no headers"}

            # Find all matching columns (fuzzy match)
            matching_cols = [h for h in headers if field_name.lower() in h.lower()]
            if not matching_cols:
                return {
                    "status": "error",
                    "error": f"No column matching '{field_name}'",
                    "available_columns": headers[:20],  # Show first 20 for reference
                }

            # Track empty waivers per column
            empty_by_column = {col: [] for col in matching_cols}
            total_rows = 0

            for row in reader:
                total_rows += 1
                waiver_key = row.get("waiver_key", "unknown")

                for col in matching_cols:
                    value = row.get(col, "").strip()

                    if not value or value.lower() in ("null", "none", "n/a"):
                        # Check if PDF exists (only once per waiver)
                        pdf_exists = any(
                            (d / f"{waiver_key}.pdf").exists() or (d / f"{waiver_key}.PDF").exists()
                            for d in PDF_DIRS if d.exists()
                        )

                        empty_by_column[col].append({
                            "waiver_key": waiver_key,
                            "pdf_exists": pdf_exists,
                        })

        # Build results - if single column, return simple format; if multiple, return per-column
        if len(matching_cols) == 1:
            target_col = matching_cols[0]
            empty_waivers = empty_by_column[target_col]
            return {
                "status": "success",
                "field_name": target_col,
                "query": field_name,
                "total_empty": len(empty_waivers),
                "total_processed": total_rows,
                "empty_percentage": round(len(empty_waivers) / total_rows * 100, 1) if total_rows else 0,
                "waivers_with_empty": empty_waivers[:limit],
                "showing": min(limit, len(empty_waivers)),
            }
        else:
            # Multiple columns match - return results for each
            results_by_column = []
            for col in matching_cols:
                empty_waivers = empty_by_column[col]
                results_by_column.append({
                    "field_name": col,
                    "total_empty": len(empty_waivers),
                    "empty_percentage": round(len(empty_waivers) / total_rows * 100, 1) if total_rows else 0,
                    "waivers_with_empty": empty_waivers[:limit],
                    "showing": min(limit, len(empty_waivers)),
                })

            return {
                "status": "success",
                "query": field_name,
                "total_processed": total_rows,
                "multiple_columns_matched": True,
                "columns_found": len(matching_cols),
                "results_by_column": results_by_column,
                "hint": "Multiple columns matched. Use a more specific query to target one column.",
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def diagnose_empty_field(waiver_key: str, field_name: str) -> dict:
    """Investigate why a field is empty for a specific waiver.

    Use this after find_empty_fields() to understand WHY a field is empty.
    Checks overview.json, evidence.json, and logs to explain the empty value.

    Args:
        waiver_key: The waiver identifier (e.g., "TX1234R0100")
        field_name: The field to investigate (supports partial matching)

    Returns:
        dict with diagnosis explaining why the field is empty, including:
        - extraction_attempts: what the LLM tried
        - evidence_detail: what text was found
        - relevant_log_entries: any errors or decisions
        - possible_reasons: likely explanations
    """
    LOG_DIR = LOGS_DIR

    waiver_dir = OUTPUT_DIR / waiver_key
    overview_path = waiver_dir / "overview.json"
    evidence_path = waiver_dir / f"{waiver_key}_evidence.json"
    log_path = LOG_DIR / f"{waiver_key}.jsonl"

    diagnosis = {
        "waiver_key": waiver_key,
        "field_name": field_name,
        "files_checked": [],
        "extraction_attempts": [],
        "possible_reasons": [],
    }

    try:
        # Check overview.json
        if overview_path.exists():
            diagnosis["files_checked"].append("overview.json")
            with open(overview_path) as f:
                overview = json.load(f)

            # The structure might be {field: {final_value, attempt_1, ...}}
            # or it might be {extracted_fields: {field: value}}
            fields_data = overview.get("fields", overview.get("extracted_fields", overview))

            # Find field (fuzzy match)
            matching_fields = [k for k in fields_data.keys() if field_name.lower() in k.lower()]
            if matching_fields:
                field_key = matching_fields[0]
                field_data = fields_data.get(field_key, {})

                diagnosis["matched_field"] = field_key

                # Handle different data structures
                if isinstance(field_data, dict):
                    diagnosis["final_value"] = field_data.get("final_value", field_data.get("value"))
                    diagnosis["resolution"] = field_data.get("resolution")

                    # Check each attempt
                    for attempt in ["attempt_1", "attempt_2", "attempt_3"]:
                        if attempt in field_data:
                            attempt_data = field_data[attempt]
                            if isinstance(attempt_data, dict):
                                diagnosis["extraction_attempts"].append({
                                    "attempt": attempt,
                                    "value": attempt_data.get("value"),
                                    "evidence": str(attempt_data.get("evidence", ""))[:200],
                                    "decision": attempt_data.get("decision"),
                                    "reason": str(attempt_data.get("critic_reason", ""))[:200],
                                })
                else:
                    # Simple value
                    diagnosis["final_value"] = field_data

        # Check evidence.json for more detail
        if evidence_path.exists():
            diagnosis["files_checked"].append("evidence.json")
            with open(evidence_path) as f:
                evidence = json.load(f)

            # Evidence structure: {fields: {field_name: {...}}}
            fields_evidence = evidence.get("fields", evidence)

            matching_fields = [k for k in fields_evidence.keys() if field_name.lower() in k.lower()]
            if matching_fields:
                field_key = matching_fields[0]
                field_evidence = fields_evidence.get(field_key, {})
                if isinstance(field_evidence, dict):
                    diagnosis["evidence_detail"] = {
                        "final_value": field_evidence.get("final_value"),
                        "evidence_snippet": str(field_evidence.get("attempt_1", {}).get("evidence", ""))[:500],
                        "resolution": field_evidence.get("resolution"),
                    }

        # Check logs for errors
        if log_path.exists():
            diagnosis["files_checked"].append("logs")
            relevant_logs = []
            with open(log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Look for critic decisions or errors related to this field
                        entry_str = str(entry).lower()
                        if (
                            "missing_child" in entry_str
                            or "hallucinated" in entry_str
                            or "error" in entry_str
                            or field_name.lower() in entry_str
                        ):
                            relevant_logs.append({
                                "agent": entry.get("agent"),
                                "action": entry.get("action"),
                                "data": str(entry.get("data", ""))[:200],
                            })
                    except json.JSONDecodeError:
                        pass
            diagnosis["relevant_log_entries"] = relevant_logs[:5]

        # Determine possible reasons based on what we found
        if not diagnosis.get("extraction_attempts") and not diagnosis.get("evidence_detail"):
            diagnosis["possible_reasons"].append("Field was never extracted (not in schema or PDF)")
        elif all(a.get("value") is None for a in diagnosis.get("extraction_attempts", [])):
            diagnosis["possible_reasons"].append("Field not found in PDF text across all attempts")
            diagnosis["possible_reasons"].append("Field may be in image-only section not OCR'd")
        elif diagnosis.get("resolution") == "missing_child":
            diagnosis["possible_reasons"].append("Field should exist but extraction failed - marked for human review")

        final_val = diagnosis.get("final_value")
        if final_val is None or final_val == "" or str(final_val).lower() in ("none", "null"):
            if not diagnosis["possible_reasons"]:
                diagnosis["possible_reasons"].append("Value extracted as empty/null - check evidence for details")

        return {"status": "success", **diagnosis}

    except Exception as e:
        return {"status": "error", "error": str(e), **diagnosis}
