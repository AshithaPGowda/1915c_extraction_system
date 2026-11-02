"""
Function tools for the Waiver Extraction ADK agent.

Provides tools for:
- Discovering available waiver PDFs
- Processing single or batch waivers
- Checking processing status and results
- Reviewing extraction output
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional

# Resolve paths
from ..shared_libraries.paths import (
    PROJECT_ROOT, DATA_ROOT as DATA_DIR, OUTPUTS_DIR as OUTPUT_DIR, PDF_DIR,
)
AGENT_PKG_DIR = Path(__file__).resolve().parent.parent

# Import pipeline
from ..core.pipeline import run_overview_extraction
from ..shared_libraries.file_utils import waiver_key_from_filename


def _find_pdf_path(waiver_key: str) -> Optional[str]:
    """Find the PDF path for a given waiver key."""
    # Check multiple possible locations
    possible_paths = [
        PDF_DIR / f"{waiver_key}.pdf",
        PDF_DIR / f"{waiver_key.upper()}.pdf",
        PDF_DIR / f"{waiver_key.lower()}.pdf",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # Try glob search
    matches = list(PDF_DIR.glob(f"*{waiver_key}*.pdf"))
    if matches:
        return str(matches[0])

    return None


# ===========================================================================
# DISCOVERY TOOLS
# ===========================================================================


def list_available_waivers(limit: int = 50) -> dict:
    """List waiver PDFs available for processing.

    Scans the pdfs_267_latest/ directory for PDF files.

    Args:
        limit: Maximum number of waivers to return (default 50).

    Returns:
        dict with count and list of waiver_keys.
    """
    if not PDF_DIR.exists():
        return {"count": 0, "waivers": [], "error": f"PDF directory not found: {PDF_DIR}"}

    waivers = []
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        waiver_key = waiver_key_from_filename(pdf_file.name)
        if waiver_key:
            waivers.append(waiver_key)
        if len(waivers) >= limit:
            break

    return {
        "count": len(waivers),
        "total_available": len(list(PDF_DIR.glob("*.pdf"))),
        "waivers": waivers,
    }


def list_processed_waivers(limit: int = 50) -> dict:
    """List waivers that have already been processed.

    Scans the outputs/ directory for completed extractions.

    Args:
        limit: Maximum number to return (default 50).

    Returns:
        dict with count and list of processed waiver info.
    """
    if not OUTPUT_DIR.exists():
        return {"count": 0, "waivers": []}

    processed = []
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if folder.is_dir():
            overview_path = folder / "overview.json"
            if overview_path.exists():
                processed.append({
                    "waiver_key": folder.name,
                    "has_overview": True,
                })
        if len(processed) >= limit:
            break

    return {
        "count": len(processed),
        "waivers": processed,
    }


def list_unprocessed_waivers(limit: int = 50) -> dict:
    """List waivers that are available but not yet processed.

    Args:
        limit: Maximum number to return (default 50).

    Returns:
        dict with count and list of unprocessed waiver_keys.
    """
    available = list_available_waivers(limit=9999)
    processed = list_processed_waivers(limit=9999)

    processed_keys = {w["waiver_key"] for w in processed.get("waivers", [])}
    unprocessed = [k for k in available.get("waivers", []) if k not in processed_keys]

    return {
        "count": len(unprocessed),
        "total_available": available.get("total_available", 0),
        "total_processed": len(processed_keys),
        "waivers": unprocessed[:limit],
    }


# ===========================================================================
# PROCESSING TOOLS
# ===========================================================================


def process_waiver(waiver_key: str, pdf_path: Optional[str] = None) -> dict:
    """Process a single waiver PDF through the extraction pipeline.

    Runs the full pipeline: checkbox resolver -> extraction -> critic ->
    reextraction -> normalization.

    Args:
        waiver_key: The waiver identifier (e.g., "NV2426R0001").
        pdf_path: Optional explicit PDF path. If not provided, will search for it.

    Returns:
        dict with status, accuracy, output_path, and pipeline details.
    """
    # Find PDF if not provided
    if not pdf_path:
        pdf_path = _find_pdf_path(waiver_key)
        if not pdf_path:
            return {
                "status": "error",
                "waiver_key": waiver_key,
                "error": f"PDF not found for waiver {waiver_key}. Searched in {PDF_DIR}",
            }

    if not os.path.exists(pdf_path):
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": f"PDF file not found: {pdf_path}",
        }

    # Run the async pipeline synchronously
    try:
        result = asyncio.run(run_overview_extraction(pdf_path, waiver_key))
        return result
    except Exception as e:
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": str(e),
        }


def process_batch(waiver_keys: list[str], stop_on_error: bool = False) -> dict:
    """Process multiple waivers in sequence.

    Args:
        waiver_keys: List of waiver identifiers to process.
        stop_on_error: If True, stop processing on first error. Default False.

    Returns:
        dict with summary statistics and per-waiver results.
    """
    results = []
    success_count = 0
    error_count = 0

    for waiver_key in waiver_keys:
        result = process_waiver(waiver_key)
        results.append({
            "waiver_key": waiver_key,
            "status": result.get("status"),
            "accuracy": result.get("final_accuracy"),
            "error": result.get("error"),
        })

        if result.get("status") == "success":
            success_count += 1
        else:
            error_count += 1
            if stop_on_error:
                break

    return {
        "total": len(waiver_keys),
        "processed": len(results),
        "success": success_count,
        "errors": error_count,
        "results": results,
    }


def process_from_file(file_path: str, limit: Optional[int] = None) -> dict:
    """Process waivers listed in a text file (one per line).

    Args:
        file_path: Path to file containing waiver keys, one per line.
        limit: Optional maximum number to process.

    Returns:
        dict with batch processing results.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        # Try relative to project root
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        return {"status": "error", "error": f"File not found: {file_path}"}

    with open(file_path) as f:
        waiver_keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if limit:
        waiver_keys = waiver_keys[:limit]

    return process_batch(waiver_keys)


# ===========================================================================
# STATUS & RESULT TOOLS
# ===========================================================================


def get_waiver_status(waiver_key: str) -> dict:
    """Check the processing status of a waiver.

    Args:
        waiver_key: The waiver identifier.

    Returns:
        dict with processing status and available outputs.
    """
    output_folder = OUTPUT_DIR / waiver_key

    if not output_folder.exists():
        # Check if PDF exists
        pdf_path = _find_pdf_path(waiver_key)
        if pdf_path:
            return {
                "waiver_key": waiver_key,
                "status": "not_processed",
                "pdf_available": True,
                "pdf_path": pdf_path,
            }
        else:
            return {
                "waiver_key": waiver_key,
                "status": "not_found",
                "pdf_available": False,
            }

    # Check what outputs exist
    outputs = {
        "overview": (output_folder / "overview.json").exists(),
        "evidence": (output_folder / "evidence_chain.json").exists(),
    }

    # Load overview to get accuracy
    overview_path = output_folder / "overview.json"
    accuracy = None
    if overview_path.exists():
        try:
            with open(overview_path) as f:
                data = json.load(f)
                accuracy = data.get("final_accuracy")
        except:
            pass

    return {
        "waiver_key": waiver_key,
        "status": "processed",
        "outputs": outputs,
        "accuracy": accuracy,
    }


def get_extraction_result(waiver_key: str) -> dict:
    """Get the extraction result for a processed waiver.

    Args:
        waiver_key: The waiver identifier.

    Returns:
        dict with extracted fields or error.
    """
    overview_path = OUTPUT_DIR / waiver_key / "overview.json"

    if not overview_path.exists():
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": f"No extraction found. Run process_waiver('{waiver_key}') first.",
        }

    with open(overview_path) as f:
        data = json.load(f)

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "extracted_fields": data.get("extracted_fields", {}),
        "low_confidence_fields": data.get("low_confidence_fields", []),
    }


def get_evidence_chain(waiver_key: str) -> dict:
    """Get the evidence chain showing how each field was extracted.

    Args:
        waiver_key: The waiver identifier.

    Returns:
        dict with per-field evidence or error.
    """
    evidence_path = OUTPUT_DIR / waiver_key / "evidence_chain.json"

    if not evidence_path.exists():
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": "No evidence chain found.",
        }

    with open(evidence_path) as f:
        data = json.load(f)

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "evidence": data,
    }


def get_failed_fields(waiver_key: str) -> dict:
    """Get fields that failed validation or have low confidence.

    Args:
        waiver_key: The waiver identifier.

    Returns:
        dict with failed/low-confidence field details.
    """
    evidence_path = OUTPUT_DIR / waiver_key / "evidence_chain.json"

    if not evidence_path.exists():
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": "No evidence chain found.",
        }

    with open(evidence_path) as f:
        evidence = json.load(f)

    failed = []
    low_confidence = []

    for field_name, field_data in evidence.get("fields", {}).items():
        if field_data.get("status") == "failed":
            failed.append({
                "field": field_name,
                "reason": field_data.get("failure_reason"),
            })
        elif field_data.get("confidence") == "low":
            low_confidence.append({
                "field": field_name,
                "value": field_data.get("final_value"),
            })

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "failed_count": len(failed),
        "low_confidence_count": len(low_confidence),
        "failed_fields": failed,
        "low_confidence_fields": low_confidence,
    }
