"""
PDF and layout tools for ADK agents.
"""

import os
import base64

from .constants import PDF_DIR
from ...shared_libraries.file_utils import load_layout


def get_pdf_path(waiver_key: str, pdf_path: str = "") -> dict:
    """Find the PDF for a waiver.

    An explicit `pdf_path` (a PDF in ANY folder) wins. Otherwise the
    waiver_key is looked up in the folders listed in the ADK_PDF_DIRS
    environment variable (os.pathsep-separated), then in the default PDF
    folder (data/pdfs_267_latest).

    Args:
        waiver_key: The waiver identifier (e.g., "NV2426R0001")
        pdf_path: Optional full path to a PDF file anywhere on disk

    Returns:
        dict with pdf_path or error
    """
    from pathlib import Path

    if pdf_path:
        p = Path(pdf_path).expanduser()
        if p.is_dir():
            return {"status": "error", "error": f"{p} is a directory - give the path of the PDF file"}
        if not p.exists():
            return {"status": "error", "error": f"PDF not found: {p}"}
        if p.suffix.lower() != ".pdf":
            return {"status": "error", "error": f"Not a PDF: {p}"}
        return {"status": "found", "pdf_path": str(p.resolve())}

    dirs = [Path(d).expanduser() for d in os.getenv("ADK_PDF_DIRS", "").split(os.pathsep) if d]
    dirs.append(PDF_DIR)
    for base in dirs:
        for name in (waiver_key, waiver_key.upper(), waiver_key.lower()):
            if (base / f"{name}.pdf").exists():
                return {"status": "found", "pdf_path": str((base / f"{name}.pdf").resolve())}
        matches = sorted(base.glob(f"*{waiver_key}*.pdf")) if base.exists() else []
        if matches:
            return {"status": "found", "pdf_path": str(matches[0].resolve())}

    return {"status": "not_found", "error": f"PDF not found for {waiver_key}"}


def list_pdfs_in_folder(folder: str, limit: int = 50) -> dict:
    """List the PDF files in a folder (any location), e.g. to process a whole
    folder of waivers one at a time.

    Args:
        folder: Path of the folder
        limit: Max number of PDFs to return (default 50)

    Returns:
        dict with count and a list of {waiver_key, pdf_path}; each waiver_key
        is derived from the file name.
    """
    from pathlib import Path
    from ...shared_libraries.file_utils import waiver_key_from_filename

    base = Path(folder).expanduser()
    if not base.is_dir():
        return {"status": "error", "error": f"Not a folder: {base}"}
    pdfs = sorted(p for p in base.iterdir() if p.suffix.lower() == ".pdf")
    return {
        "status": "success",
        "folder": str(base.resolve()),
        "count": len(pdfs),
        "returned": min(len(pdfs), limit),
        "pdfs": [{"waiver_key": waiver_key_from_filename(p.name), "pdf_path": str(p.resolve())}
                 for p in pdfs[:limit]],
    }


def get_extraction_schema() -> dict:
    """Get the field schema for waiver overview extraction.

    Returns:
        dict with field names and their definitions
    """
    layout = load_layout("layout_waiver_overview")

    # Layout is a dict: field_name -> {Header, isInTable, context}
    field_names = list(layout.keys())

    # Build field definitions for the agent
    fields = {}
    for field_name, field_info in layout.items():
        fields[field_name] = {
            "header": field_info.get("Header", ""),
            "context": field_info.get("context", ""),
            "is_in_table": field_info.get("isInTable", False),
        }

    return {
        "status": "success",
        "total_fields_to_extract": len(field_names),
        "field_names": field_names,
        "fields": fields,
    }


def load_pdf_bytes(pdf_path: str) -> dict:
    """Load PDF file as base64 for passing to Gemini.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        dict with base64 encoded PDF or error
    """
    if not os.path.exists(pdf_path):
        return {"status": "error", "error": f"PDF not found: {pdf_path}"}

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    return {
        "status": "success",
        "size_bytes": len(pdf_bytes),
        "base64": base64.b64encode(pdf_bytes).decode("utf-8"),
    }
