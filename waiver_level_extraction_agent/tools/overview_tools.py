"""Tools for the Waiver Overview agent."""

import os
import sys
import json
import csv
import base64
import subprocess
from pathlib import Path
from ..shared_libraries.file_utils import (
    waiver_key_from_filename,
    write_json_output,
    log_event,
)

# Path setup for CSV auto-append
from ..shared_libraries.paths import PROJECT_ROOT, WAIVER_LEVEL_CSV as CSV_PATH, SCRIPTS_DIR
EXPORT_SCRIPT = SCRIPTS_DIR / "export_to_csv.py"


def append_to_csv(waiver_key: str) -> bool:
    """
    Append a waiver's overview data to the global CSV.
    Calls export_to_csv.py with --waiver flag.
    Returns True if successful.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(EXPORT_SCRIPT), "--waiver", waiver_key],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print(f"  Auto-appended {waiver_key} to CSV")
            return True
        else:
            print(f"  Warning: CSV append failed for {waiver_key}: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Warning: CSV append error for {waiver_key}: {e}")
        return False


def load_pdf_as_base64(pdf_path: str) -> str:
    """
    Load a PDF file and return as base64 string for passing to Gemini.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_overview_output(waiver_key: str, extracted_data: dict) -> str:
    """
    Save the overview extraction result to outputs/{waiver_key}/overview.json.
    Logs the event. Returns the output file path.
    """
    # Separate confidence flags from actual data for cleaner output
    data = {k: v for k, v in extracted_data.items() if not k.endswith("_confidence")}
    confidence_flags = {k: v for k, v in extracted_data.items() if k.endswith("_confidence")}

    output = {
        "waiver_key": waiver_key,
        "extracted_fields": data,
        "low_confidence_fields": [
            k.replace("_confidence", "") for k, v in confidence_flags.items() if v == "low"
        ],
        "_discovered_services": data.get("_discovered_services", []),
    }

    path = write_json_output(waiver_key, "overview.json", output)

    log_event(waiver_key, {
        "agent": "waiver_overview_agent",
        "action": "extraction_complete",
        "fields_extracted": len([v for v in data.values() if v is not None]),
        "fields_null": len([v for v in data.values() if v is None]),
        "low_confidence_count": len(output["low_confidence_fields"]),
        "services_discovered": len(output["_discovered_services"]),
        "output_path": path,
    })

    # Auto-append to global CSV
    append_to_csv(waiver_key)

    return path


def get_discovered_services(waiver_key: str) -> list[str]:
    """
    Read the overview.json output and return the discovered service names list.
    Used by the orchestrator to build the Services agent prompt.
    """
    from ..shared_libraries.paths import OUTPUTS_DIR
    path = os.path.join(str(OUTPUTS_DIR), waiver_key, "overview.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Overview output not found for {waiver_key}. "
            f"Run the overview agent first."
        )
    with open(path) as f:
        data = json.load(f)
    return data.get("_discovered_services", [])
