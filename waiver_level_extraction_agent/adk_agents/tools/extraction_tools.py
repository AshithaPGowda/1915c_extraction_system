"""
Extraction tools for ADK agents.

These tools actually send PDFs to Gemini for field extraction.
"""

import json
from google import genai
from google.genai import types

from .pdf_tools import get_extraction_schema
from ...shared_libraries.file_utils import load_layout


def _get_client():
    """Get the Gemini client."""
    return genai.Client()


def _load_pdf_bytes(pdf_path: str) -> bytes:
    """Load PDF file as bytes."""
    with open(pdf_path, "rb") as f:
        return f.read()


def _load_overview_prompt() -> str:
    """Load the overview extraction prompt."""
    from ...sub_agents.overview_prompt import OVERVIEW_AGENT_PROMPT
    return OVERVIEW_AGENT_PROMPT


def _parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


def extract_all_fields(pdf_path: str, waiver_key: str, checkbox_values: dict = None) -> dict:
    """Extract all fields from a PDF using Gemini multimodal.

    This is the main extraction tool that sends the PDF to Gemini
    with the extraction prompt and returns structured field data.

    Args:
        pdf_path: Path to the PDF file
        waiver_key: Waiver identifier for logging
        checkbox_values: Optional pre-resolved checkbox values to include

    Returns:
        dict with extracted fields or error
    """
    try:
        client = _get_client()
        pdf_bytes = _load_pdf_bytes(pdf_path)
        base_prompt = _load_overview_prompt()

        # Add checkbox hints if available
        if checkbox_values:
            checkbox_hint = "\n\n## Pre-resolved Checkbox Values (AUTHORITATIVE)\n"
            checkbox_hint += "The following checkbox fields have been resolved from PDF metadata. Use these values:\n"
            for field, value in checkbox_values.items():
                if value is not None:
                    checkbox_hint += f"- {field}: {value}\n"
            base_prompt = base_prompt + checkbox_hint

        # Call Gemini with PDF
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_bytes,
                            )
                        ),
                        types.Part(text=base_prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,
            ),
        )

        raw_text = response.text.strip()

        # Parse JSON response
        try:
            extracted_data = _parse_json_response(raw_text)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Failed to parse JSON response: {e}",
                "raw_response_preview": raw_text[:1000],
            }

        # Count fields
        schema = get_extraction_schema()
        expected_fields = schema.get("total_fields_to_extract", 0)
        extracted_count = len([k for k in extracted_data.keys() if not k.startswith("_")])

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "extracted_fields": extracted_count,
            "expected_fields": expected_fields,
            "data": extracted_data,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def extract_specific_fields(pdf_path: str, field_names: list, page_hints: dict = None) -> dict:
    """Extract specific fields from a PDF (for re-extraction).

    Use this when you need to re-extract only certain fields,
    typically after critic verification identified issues.

    Args:
        pdf_path: Path to the PDF file
        field_names: List of field names to extract
        page_hints: Optional dict of field_name -> page_number hints

    Returns:
        dict with extracted field values
    """
    try:
        client = _get_client()
        pdf_bytes = _load_pdf_bytes(pdf_path)

        # Build focused extraction prompt
        layout = load_layout("layout_waiver_overview")

        field_descriptions = []
        for field_name in field_names:
            if field_name in layout:
                info = layout[field_name]
                desc = f"- **{field_name}**: {info.get('context', 'No description')}"
                if page_hints and field_name in page_hints:
                    desc += f" (likely on page {page_hints[field_name]})"
                field_descriptions.append(desc)

        prompt = f"""Extract ONLY the following fields from this PDF document.

## Fields to Extract:
{chr(10).join(field_descriptions)}

## Instructions:
1. Search carefully for each field
2. Return exact values as they appear in the document
3. For checkbox/boolean fields: use 1 for checked, 0 for unchecked
4. For dates: use YYYY-MM-DD format
5. If a field is not found, return null

Return a JSON object with field names as keys and extracted values.
Include a "_page" suffix key for each field with the page number where found.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_bytes,
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,  # Increased from 8192 to handle large extractions
            ),
        )

        raw_text = response.text.strip()

        try:
            extracted_data = _parse_json_response(raw_text)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Failed to parse JSON: {e}",
                "raw_response_preview": raw_text[:500],
            }

        return {
            "status": "success",
            "fields_requested": len(field_names),
            "fields_extracted": len([k for k in extracted_data.keys() if not k.endswith("_page")]),
            "data": extracted_data,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
