"""
Checkbox resolver tools for pre-extraction of boolean fields.
"""

from ...shared_libraries.file_utils import load_layout
from ...sub_agents.checkbox_resolver import resolve_all_checkboxes, get_resolver_summary


def resolve_checkboxes(pdf_path: str, waiver_key: str) -> dict:
    """Pre-extract boolean fields using checkbox resolver.

    Uses 3-layer resolution: AcroForm → Spatial Analysis → None

    Args:
        pdf_path: Path to the PDF file
        waiver_key: Waiver identifier for logging

    Returns:
        dict with resolved checkbox values
    """
    try:
        layout = load_layout("layout_waiver_overview")

        checkbox_cache = resolve_all_checkboxes(
            pdf_path=pdf_path,
            layout=layout,
            page_hints=None,
            waiver_key=waiver_key,
        )

        summary = get_resolver_summary(checkbox_cache)

        return {
            "status": "success",
            "total_fields": len(checkbox_cache),
            "resolved_acroform": summary["acroform"],
            "resolved_spatial": summary["spatial"],
            "no_resolution": summary["none"],
            "cache": checkbox_cache,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
