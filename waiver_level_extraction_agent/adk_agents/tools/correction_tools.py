"""
Manual correction tools for updating extraction results.
"""

import json
import datetime
from typing import Any, Optional

from .constants import DATA_DIR, OUTPUT_DIR
from .output_tools import log_agent_event


def apply_manual_correction(
    waiver_key: str,
    field_name: str,
    corrected_value: Any,
    reason: Optional[str] = None
) -> dict:
    """Apply a manual correction to a field in a processed waiver.

    Records the correction in human_overrides.jsonl and updates
    the overview.json output file.

    Args:
        waiver_key: The waiver identifier
        field_name: The field to correct
        corrected_value: The new value to set
        reason: Optional reason for the correction

    Returns:
        dict with correction result
    """
    # Load existing output
    output_path = OUTPUT_DIR / waiver_key / "overview.json"

    if not output_path.exists():
        return {
            "status": "error",
            "error": f"No output found for {waiver_key}. Process the waiver first.",
        }

    with open(output_path) as f:
        data = json.load(f)

    extracted = data.get("extracted_fields", {})
    old_value = extracted.get(field_name)

    # Apply correction
    extracted[field_name] = corrected_value
    data["extracted_fields"] = extracted

    # Add correction metadata
    if "_manual_corrections" not in data:
        data["_manual_corrections"] = []

    data["_manual_corrections"].append({
        "field": field_name,
        "old_value": old_value,
        "new_value": corrected_value,
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    })

    # Save updated output
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Record in human_overrides.jsonl
    override_record = {
        "waiver_number": ".".join([
            waiver_key[:2],
            waiver_key[2:6],
            waiver_key[6:8],
            waiver_key[8:10]
        ]) if len(waiver_key) >= 10 else waiver_key,
        "field_name": field_name,
        "old_value": old_value,
        "corrected_value": corrected_value,
        "conflict_type": "manual_correction",
        "source": "human",
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    overrides_path = DATA_DIR / "human_overrides.jsonl"
    with open(overrides_path, "a") as f:
        f.write(json.dumps(override_record) + "\n")

    log_agent_event(waiver_key, "orchestrator", "manual_correction", {
        "field": field_name,
        "old_value": old_value,
        "new_value": corrected_value,
        "reason": reason,
    })

    # Update CSV with corrected value
    from ...tools.overview_tools import append_to_csv
    append_to_csv(waiver_key)

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "field": field_name,
        "old_value": old_value,
        "new_value": corrected_value,
        "output_path": str(output_path),
    }


def get_field_value(waiver_key: str, field_name: str) -> dict:
    """Get the current value of a field from a processed waiver.

    Useful for checking values before applying corrections.

    Args:
        waiver_key: The waiver identifier
        field_name: The field to look up

    Returns:
        dict with field value and metadata
    """
    output_path = OUTPUT_DIR / waiver_key / "overview.json"

    if not output_path.exists():
        return {
            "status": "not_found",
            "error": f"No output found for {waiver_key}",
        }

    with open(output_path) as f:
        data = json.load(f)

    extracted = data.get("extracted_fields", {})

    if field_name not in extracted:
        # Try partial match
        matches = [k for k in extracted.keys() if field_name.lower() in k.lower()]
        if matches:
            return {
                "status": "partial_match",
                "field_name": field_name,
                "possible_matches": matches[:10],
            }
        return {"status": "field_not_found", "field_name": field_name}

    # Check if field was manually corrected
    manual_corrections = data.get("_manual_corrections", [])
    field_corrections = [c for c in manual_corrections if c.get("field") == field_name]

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "field": field_name,
        "value": extracted[field_name],
        "has_manual_corrections": len(field_corrections) > 0,
        "corrections_history": field_corrections,
    }
