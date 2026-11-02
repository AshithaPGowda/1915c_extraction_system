"""
Page Image Extractor for Attempt 2 checkbox OCR.

Extracts specific pages from PDF as base64-encoded JPEG images
for sending to the LLM for visual checkbox inspection.
"""

import base64
import io
import re
from pdf2image import convert_from_path


def get_page_images_base64(pdf_path: str, page_num: int) -> list[dict]:
    """
    Render pages page_num-1, page_num, page_num+1 at 150 DPI.

    Args:
        pdf_path: Path to the PDF file
        page_num: Target page number (0-indexed)

    Returns:
        List of dicts: [{page_num: int, base64: str, media_type: "image/jpeg"}]
        Pages are clamped to valid range (no below 0, no above last page)
    """
    # Ensure page_num is int (callers may pass strings from LLM output)
    page_num = int(page_num)

    # Get total page count first
    from pdf2image.pdf2image import pdfinfo_from_path
    try:
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
    except Exception:
        # Fallback: try to convert first page to get page count
        total_pages = 1000  # Will be clamped anyway

    # Calculate page range (clamp to valid bounds)
    first_page = max(0, page_num - 1)
    last_page = min(total_pages - 1, page_num + 1)

    # pdf2image uses 1-indexed pages
    first_page_1idx = first_page + 1
    last_page_1idx = last_page + 1

    # Convert pages to images at 150 DPI
    images = convert_from_path(
        pdf_path,
        first_page=first_page_1idx,
        last_page=last_page_1idx,
        dpi=150,
        fmt="jpeg"
    )

    results = []
    current_page = first_page

    for img in images:
        # Convert PIL Image to base64 JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        b64_data = base64.standard_b64encode(buffer.read()).decode("utf-8")

        results.append({
            "page_num": current_page,
            "base64": b64_data,
            "media_type": "image/jpeg"
        })
        current_page += 1

    return results


def parse_page_from_evidence(evidence_str: str) -> int | None:
    """
    Extract page number from evidence string.

    Scans for patterns like:
    - "page 14"
    - "p.14"
    - "Page 14"
    - "on page 14"
    - "pg 14"
    - "pg. 14"

    Args:
        evidence_str: The evidence string from extraction

    Returns:
        Page number as 0-indexed integer, or None if not found
    """
    if not evidence_str:
        return None

    # Pattern to match various page references
    # Matches: page 14, Page 14, p.14, p. 14, pg 14, pg. 14, on page 14
    patterns = [
        r'[Pp]age\s*(\d+)',      # "page 14", "Page 14"
        r'[Pp]\.?\s*(\d+)',       # "p.14", "p. 14", "P14"
        r'[Pp]g\.?\s*(\d+)',      # "pg 14", "pg. 14", "Pg14"
        r'on\s+page\s+(\d+)',     # "on page 14"
    ]

    for pattern in patterns:
        match = re.search(pattern, evidence_str)
        if match:
            page_1idx = int(match.group(1))
            # Convert to 0-indexed
            return page_1idx - 1

    return None
