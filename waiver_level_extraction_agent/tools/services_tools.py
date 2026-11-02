"""Tools for the Waiver Services extraction agent."""

import os
import json
from pathlib import Path

from ..shared_libraries.file_utils import (
    write_json_output,
    log_event,
    get_output_dir,
)


def get_discovered_services_from_overview(waiver_key: str) -> list[str]:
    """
    Read the overview.json output and return the discovered service names list.

    Args:
        waiver_key: Normalized waiver key (e.g., "AK0260R0512")

    Returns:
        List of service names discovered during overview extraction

    Raises:
        FileNotFoundError: If overview.json doesn't exist for this waiver
    """
    output_dir = get_output_dir(waiver_key)
    path = os.path.join(output_dir, "overview.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Overview output not found for {waiver_key}. "
            f"Run the overview pipeline first, or use --standalone."
        )

    with open(path) as f:
        data = json.load(f)

    # Handle both formats: direct list or nested in extracted_fields
    services = data.get("_discovered_services", [])
    if not services:
        extracted = data.get("extracted_fields", {})
        services = extracted.get("_discovered_services", [])

    return services


def get_waiver_metadata_from_overview(waiver_key: str) -> dict:
    """
    Get waiver metadata (state, title, etc.) from overview extraction.

    Args:
        waiver_key: Normalized waiver key

    Returns:
        Dict with state, waiver_title, waiver_number, effective_date

    Raises:
        FileNotFoundError: If overview.json doesn't exist
    """
    output_dir = get_output_dir(waiver_key)
    path = os.path.join(output_dir, "overview.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Overview output not found for {waiver_key}. "
            f"Run the overview pipeline first."
        )

    with open(path) as f:
        data = json.load(f)

    # Extract from the nested structure
    extracted = data.get("extracted_fields", data)

    return {
        "state": extracted.get("1.A State"),
        "waiver_title": extracted.get("1.B Waiver Title"),
        "waiver_number": extracted.get("1.C Waiver number"),
        "effective_date": extracted.get("1.E. Approved Effective Date"),
    }


def save_services_output(waiver_key: str, services_data: dict) -> str:
    """
    Save the services extraction result to outputs/{waiver_key}/services.json.

    Args:
        waiver_key: Normalized waiver key
        services_data: Dict with services extraction results

    Returns:
        Path to the output file
    """
    path = write_json_output(waiver_key, "services.json", services_data)

    log_event(waiver_key, {
        "agent": "services_tools",
        "action": "output_saved",
        "output_path": path,
        "services_count": len(services_data.get("services", [])),
    })

    return path


def get_service_page_hints(pdf_path: str, service_name: str) -> dict:
    """
    Get page hints for a specific service.

    Uses text search to find likely pages where the service appears.

    Args:
        pdf_path: Path to the PDF file
        service_name: The service name to search for

    Returns:
        Dict with appendix_c_pages, appendix_e_pages, appendix_j_pages
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        # PyMuPDF not available, return empty hints
        return {}

    hints = {
        "service_pages": [],
        "appendix_c_pages": None,
        "appendix_e_pages": None,
        "appendix_j_pages": None,
    }

    try:
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_lower = text.lower()

            # Find service mentions
            if service_name.lower() in text_lower:
                hints["service_pages"].append(page_num + 1)  # 1-indexed

            # Find appendix markers
            if "appendix c" in text_lower or "c-1" in text_lower:
                if hints["appendix_c_pages"] is None:
                    hints["appendix_c_pages"] = []
                hints["appendix_c_pages"].append(page_num + 1)

            if "appendix e" in text_lower or "e-1" in text_lower:
                if hints["appendix_e_pages"] is None:
                    hints["appendix_e_pages"] = []
                hints["appendix_e_pages"].append(page_num + 1)

            if "appendix j" in text_lower or "j-2" in text_lower:
                if hints["appendix_j_pages"] is None:
                    hints["appendix_j_pages"] = []
                hints["appendix_j_pages"].append(page_num + 1)

        doc.close()

        # Convert page lists to ranges
        for key in ["appendix_c_pages", "appendix_e_pages", "appendix_j_pages"]:
            pages = hints.get(key)
            if pages:
                hints[key] = [min(pages), max(pages)]

    except Exception as e:
        # On any error, return empty hints
        print(f"Warning: Could not get page hints: {e}")

    return hints


def list_waivers_with_overview() -> list[str]:
    """
    List all waivers that have overview.json (ready for services extraction).

    Returns:
        List of waiver keys
    """
    from ..shared_libraries.paths import OUTPUTS_DIR
    base_output_dir = str(OUTPUTS_DIR)

    waivers = []

    if not os.path.exists(base_output_dir):
        return waivers

    for item in os.listdir(base_output_dir):
        item_path = os.path.join(base_output_dir, item)
        if os.path.isdir(item_path):
            overview_path = os.path.join(item_path, "overview.json")
            if os.path.exists(overview_path):
                waivers.append(item)

    return sorted(waivers)


def get_pdf_path_for_waiver(waiver_key: str, pdf_dir: str | None = None) -> str | None:
    """
    Find the PDF file for a waiver key.

    Args:
        waiver_key: Normalized waiver key
        pdf_dir: Optional directory to search (defaults to data/pdfs)

    Returns:
        Path to the PDF file, or None if not found
    """
    if pdf_dir is None:
        from ..shared_libraries.paths import PDF_DIR
        pdf_dir = str(PDF_DIR)

    # Try various filename patterns
    patterns = [
        f"{waiver_key}.pdf",
        f"{waiver_key}.PDF",
        # With dots: CA0336R0404 -> CA.0336.R04.04.pdf
        _format_with_dots(waiver_key) + ".pdf",
        _format_with_dots(waiver_key) + ".PDF",
    ]

    for pattern in patterns:
        path = os.path.join(pdf_dir, pattern)
        if os.path.exists(path):
            return path

    # Try glob search
    import glob
    for ext in ["*.pdf", "*.PDF"]:
        matches = glob.glob(os.path.join(pdf_dir, ext))
        for match in matches:
            basename = os.path.basename(match)
            # Normalize the basename and compare
            normalized = _normalize_filename(basename)
            if normalized == waiver_key:
                return match

    return None


def _format_with_dots(waiver_key: str) -> str:
    """Format waiver key with dots: CA0336R0404 -> CA.0336.R04.04"""
    key = waiver_key.upper()
    if len(key) >= 10:
        state = key[:2]
        waiver_num = key[2:6] if len(key) > 5 else key[2:]
        revision_part = key[6:] if len(key) > 6 else ""

        if revision_part.startswith("R"):
            rev_major = revision_part[1:3] if len(revision_part) > 2 else revision_part[1:]
            rev_minor = revision_part[3:5] if len(revision_part) > 4 else "00"
            return f"{state}.{waiver_num}.R{rev_major}.{rev_minor}"

    return waiver_key


def _normalize_filename(filename: str) -> str:
    """Normalize a PDF filename to a waiver key."""
    import re
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Remove dots, spaces, uppercase
    return re.sub(r'[.\s]', '', name).upper().strip()
