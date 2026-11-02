"""
Page image extraction tools for visual re-extraction.
"""

from ...sub_agents.page_image_extractor import get_page_images_base64


def get_page_images(pdf_path: str, pages: list[int]) -> dict:
    """Extract specific pages as images for visual re-extraction.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers (1-indexed)

    Returns:
        dict with base64-encoded images
    """
    try:
        all_images = []
        for page_num in pages:
            page_images = get_page_images_base64(pdf_path, page_num)
            for img in page_images:
                img["page"] = page_num
                all_images.append(img)

        return {
            "status": "success",
            "page_count": len(all_images),
            "images": all_images,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_field_page_hint(extracted_data: dict, field_name: str) -> dict:
    """Get the page number where a field was extracted from.

    Args:
        extracted_data: The extraction dict with _page keys
        field_name: The field to look up

    Returns:
        dict with page number or None
    """
    page_key = f"{field_name}_page"
    page = extracted_data.get(page_key)

    if page:
        # Convert to int (LLM may return strings like "5" instead of 5)
        page = int(page)
        return {
            "field": field_name,
            "page": page,
            "page_window": [max(1, page - 1), page, page + 1],
        }
    else:
        return {
            "field": field_name,
            "page": None,
            "page_window": [],
        }
