"""
Re-extraction Agent for focused field extraction.

Same underlying model capability as overview agent, but with a focused prompt.
Receives: full PDF inline, list of failing/low-confidence fields with hints.
Returns: JSON with only the re-extracted subset of fields.

Enhanced with AcroForm context and page window support.
For checkbox fields with needs_image_ocr=True, sends page images instead of PDF.
"""

import os
import json
import base64

from google import genai
from google.genai import types

from ..shared_libraries.paths import OUTPUTS_DIR

from ..shared_libraries.model_config import get_model_name, get_location

from ..shared_libraries.file_utils import load_layout, log_event
from .reextraction_prompt import build_reextraction_prompt
from .page_image_extractor import get_page_images_base64, parse_page_from_evidence


def _get_client():
    """Get Gemini client. Checks GOOGLE_GENAI_USE_VERTEXAI env var."""
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE"

    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=get_location(),
        )
    else:
        # Use Google AI API with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)


def _load_pdf_bytes(pdf_path: str) -> bytes:
    """Read PDF file as bytes."""
    with open(pdf_path, "rb") as f:
        return f.read()


def _parse_json_response(raw_text: str) -> dict | None:
    """Parse JSON from model response, handling markdown fences and extra data."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (possibly with language hint like ```json)
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Use JSONDecoder to parse only the first JSON object
    # This handles cases where LLM outputs extra text after the JSON
    decoder = json.JSONDecoder()
    text = text.lstrip()  # Remove leading whitespace
    try:
        result, end_idx = decoder.raw_decode(text)
        return result
    except json.JSONDecodeError:
        # Fallback to regular parsing for better error messages
        return json.loads(text)


def _get_field_page(field: dict) -> int | None:
    """
    Get page number for a field from various sources.

    Checks in order:
    1. field["page"] - explicit page from context
    2. field["_page"] - alternate key
    3. Parses from field["evidence"] text (fallback)

    Returns:
        Page number (0-indexed) or None if not found
    """
    # Try explicit page keys first
    page = field.get("page")
    if page is not None:
        # Convert to int (LLM may return strings like "5" instead of 5)
        page = int(page)
        # Convert to 0-indexed if 1-indexed
        return page - 1 if page > 0 else page

    page = field.get("_page")
    if page is not None:
        page = int(page)
        return page - 1 if page > 0 else page

    # Try pages list from context (take middle page)
    pages = field.get("pages")
    if pages and len(pages) > 0:
        # Take middle page, convert to 0-indexed
        middle_idx = len(pages) // 2
        page_val = int(pages[middle_idx])
        return page_val - 1 if page_val > 0 else page_val

    # Fallback to evidence parsing
    evidence = field.get("evidence", "")
    if evidence:
        parsed = parse_page_from_evidence(evidence)
        if parsed is not None:
            return parsed

    return None


def _build_image_ocr_prompt(field: dict) -> tuple[str, str]:
    """
    Build a focused prompt for image-based checkbox OCR.

    Args:
        field: Field dict with field name, extracted value, reason, etc.

    Returns:
        Tuple of (prompt string, short_key) where short_key is used in JSON output
    """
    field_name = field.get("field", "")
    previous_value = field.get("extracted") or field.get("llm_value")
    rejection_reason = field.get("reason") or field.get("failure_reason", "Value did not match expected")
    acroform_value = field.get("acroform_value")

    # Use a short key for JSON output to avoid truncation issues
    short_key = "value"

    # Build AcroForm hint if available
    acroform_hint = ""
    if acroform_value is not None:
        acroform_hint = f"""
## ACROFORM AUTHORITATIVE VALUE
The PDF form metadata says this field's value is: **{acroform_value}**
Your visual extraction should reconcile with this authoritative value.
"""

    prompt = f"""You are looking at page images from a 1915(c) Medicaid waiver PDF.

## YOUR SINGLE TASK

Find and read the checkbox for this ONE field:
**{field_name}**

## WHAT ATTEMPT 1 EXTRACTED
Previous value: {previous_value}
Why it was rejected: {rejection_reason}
{acroform_hint}
## INSTRUCTIONS

1. Locate the checkbox for this field in the page images
2. Visually inspect the checkbox state:
   - Filled checkbox/circle (☒, ●, X, filled) = CHECKED = 1
   - Empty checkbox/circle (☐, ○, empty) = UNCHECKED = 0
3. For Miller Trust: INVERTED coding (0=Yes, 1=No)
4. Return ONLY a JSON object with the value, evidence, and page

## REQUIRED OUTPUT FORMAT

Return ONLY this JSON structure, no markdown fences, no explanation:
{{"value": 0 or 1, "evidence": "description of what you see", "page": <page_number>}}

Example output:
{{"value": 0, "evidence": "CHECKED box next to Yes", "page": 15}}
{{"value": 1, "evidence": "UNCHECKED empty circle next to Yes, CHECKED circle next to No", "page": 3}}

Return ONLY the JSON object, nothing else."""

    return prompt, short_key


def _process_image_path_field(
    client,
    pdf_path: str,
    field: dict,
    checkbox_cache: dict | None,
    waiver_key: str,
    attempt: int
) -> dict:
    """
    Process a single field via image OCR path.

    Args:
        client: Gemini client
        pdf_path: Path to PDF file
        field: Field dict with needs_image_ocr=True
        checkbox_cache: Optional cache from checkbox resolver
        waiver_key: Waiver key for logging
        attempt: Attempt number

    Returns:
        Dict with extracted field value, evidence, and page
    """
    field_name = field.get("field", "")

    # Check if checkbox_cache has high-confidence value for this field
    if checkbox_cache and field_name in checkbox_cache:
        cache_entry = checkbox_cache[field_name]
        if cache_entry.get("confidence") == "high":
            value = cache_entry.get("value")
            log_event(waiver_key, {
                "agent": "reextraction_agent",
                "action": "image_ocr_skipped",
                "field": field_name,
                "reason": "high_confidence_resolver",
                "resolver_value": value,
            })
            return {
                field_name: value,
                f"{field_name}_evidence": f"Resolver (high confidence): {cache_entry.get('raw_acroform_value', 'N/A')}",
                f"{field_name}_page": cache_entry.get("page"),  # May be None
            }

    # Get page number from field context
    page_num = _get_field_page(field)
    if page_num is None:
        page_num = 0  # Fallback to first page

    log_event(waiver_key, {
        "agent": "reextraction_agent",
        "action": "image_ocr_start",
        "field": field_name,
        "page_num": page_num,
        "attempt": attempt,
    })

    # Get page images
    try:
        page_images = get_page_images_base64(pdf_path, page_num)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "reextraction_agent",
            "action": "image_ocr_error",
            "field": field_name,
            "error": f"Failed to extract page images: {e}",
        })
        return {}

    # Build focused prompt
    prompt, short_key = _build_image_ocr_prompt(field)

    # Build content parts: images + prompt
    parts = []
    for img_data in page_images:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type=img_data["media_type"],
                    data=base64.standard_b64decode(img_data["base64"]),
                )
            )
        )
    parts.append(types.Part(text=prompt))

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,
            ),
        )

        raw_text = response.text.strip()

        # Debug: write raw response (unique per waiver)
        debug_path = os.path.join(
            str(OUTPUTS_DIR),
            f"debug_{waiver_key}_attempt{attempt}_image_ocr_{field_name[:30]}.txt"
        )
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w") as f:
            f.write(raw_text)

        raw_result = _parse_json_response(raw_text)

        # Map short key back to full field name
        result = {}
        if raw_result:
            value = raw_result.get("value")
            evidence = raw_result.get("evidence", "")
            page = raw_result.get("page")
            if value is not None:
                result[field_name] = value
                result[f"{field_name}_evidence"] = f"Image OCR: {evidence}"
                # Use page from LLM response, or fallback to what we sent
                # Always convert to int (LLM may return strings like "15" instead of 15)
                result[f"{field_name}_page"] = int(page) if page else (page_num + 1)  # Convert back to 1-indexed

        log_event(waiver_key, {
            "agent": "reextraction_agent",
            "action": "image_ocr_complete",
            "field": field_name,
            "pages_used": [img["page_num"] for img in page_images],
            "result": result.get(field_name) if result else None,
        })

        return result

    except Exception as e:
        log_event(waiver_key, {
            "agent": "reextraction_agent",
            "action": "image_ocr_error",
            "field": field_name,
            "error": str(e),
        })
        return {}


def run_reextraction(
    waiver_key: str,
    pdf_path: str,
    failing_fields: list[dict],
    low_confidence_fields: list[dict],
    attempt: int,
    checkbox_cache: dict | None = None,
    fields_context: list[dict] | None = None,
) -> dict:
    """
    Run re-extraction for failing and low-confidence fields.

    Args:
        waiver_key: Normalized waiver key
        pdf_path: Path to the PDF file
        failing_fields: List of failing field dicts from critic
        low_confidence_fields: List of low-confidence field dicts from critic
        attempt: The attempt number this re-extraction is for
        checkbox_cache: Optional checkbox resolver cache for high-confidence values
        fields_context: Optional rich context from pipeline._build_reextraction_context()
            Each dict contains: field_name, failure_reason, acroform_value,
            llm_value, acroform_subtree, ancestor_path, pages, parent_path

    Returns:
        Dict with re-extracted field values (subset of fields only)
    """
    all_fields = failing_fields + low_confidence_fields

    if not all_fields:
        log_event(waiver_key, {
            "agent": "reextraction_agent",
            "action": "skipped",
            "attempt": attempt,
            "reason": "no_fields_to_reextract",
        })
        return {}

    # Build context lookup for enriching fields
    context_lookup = {}
    if fields_context:
        for ctx in fields_context:
            field_name = ctx.get("field_name")
            if field_name:
                context_lookup[field_name] = ctx

    # Enrich fields with context (for image path processing)
    enriched_fields = []
    for field in all_fields:
        field_name = field.get("field", "")
        ctx = context_lookup.get(field_name, {})

        # Merge context into field dict
        enriched = dict(field)
        if ctx:
            enriched["acroform_value"] = ctx.get("acroform_value")
            enriched["pages"] = ctx.get("pages")
            enriched["failure_reason"] = ctx.get("failure_reason")
            enriched["llm_value"] = ctx.get("llm_value")
            enriched["needs_image_ocr"] = ctx.get("needs_image_ocr", False)

        enriched_fields.append(enriched)

    # Split fields into text path and image path
    text_path_fields = []
    image_path_fields = []

    for field in enriched_fields:
        if field.get("needs_image_ocr"):
            image_path_fields.append(field)
        else:
            text_path_fields.append(field)

    # Log the split
    log_event(waiver_key, {
        "agent": "reextraction_agent",
        "action": "reextraction_start",
        "attempt": attempt,
        "text_path_count": len(text_path_fields),
        "image_path_count": len(image_path_fields),
        "text_path_fields": [f.get("field") for f in text_path_fields],
        "image_path_fields": [f.get("field") for f in image_path_fields],
        "has_fields_context": fields_context is not None,
    })

    results = {}
    client = _get_client()

    # === TEXT PATH: Send full PDF inline ===
    if text_path_fields:
        # Load layout schema for location hints
        layout_schema = load_layout("layout_waiver_overview")

        # Build the focused prompt for text-path fields only
        text_failing = [f for f in failing_fields if not f.get("needs_image_ocr")]
        text_low_conf = [f for f in low_confidence_fields if not f.get("needs_image_ocr")]

        reextraction_prompt = build_reextraction_prompt(
            text_failing,
            text_low_conf,
            layout_schema,
            fields_context=fields_context,  # Pass rich context
        )

        # Load PDF
        pdf_bytes = _load_pdf_bytes(pdf_path)

        try:
            # Build initial content with PDF and prompt
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_bytes,
                            )
                        ),
                        types.Part(text=reextraction_prompt),
                    ],
                )
            ]

            response = client.models.generate_content(
                model=get_model_name(),
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=65536,  # Increased from 16384 to max
                ),
            )

            raw_text = response.text.strip()

            # Debug: write raw response (unique per waiver)
            debug_path = os.path.join(
                str(OUTPUTS_DIR),
                f"debug_{waiver_key}_attempt{attempt}_reextraction.txt"
            )
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, "w") as f:
                f.write(raw_text)

            # Try to parse JSON, with retry for truncated responses
            text_results = None
            max_continuation_attempts = 2

            for continuation_attempt in range(max_continuation_attempts + 1):
                try:
                    text_results = _parse_json_response(raw_text)
                    break  # Success - exit retry loop
                except json.JSONDecodeError as json_err:
                    # Check if it looks like truncated output (ends mid-string or missing closing braces)
                    stripped = raw_text.rstrip()
                    is_truncated = (
                        not stripped.endswith("}") or
                        '"' in stripped[-50:] and not stripped.rstrip().endswith('"}')
                    )

                    if is_truncated and continuation_attempt < max_continuation_attempts:
                        log_event(waiver_key, {
                            "agent": "reextraction_agent",
                            "action": "text_path_truncated_retry",
                            "attempt": attempt,
                            "continuation_attempt": continuation_attempt + 1,
                            "error": str(json_err),
                        })

                        # Ask LLM to continue from where it left off
                        continuation_prompt = (
                            "Your previous JSON response was truncated. "
                            "Please continue EXACTLY from where you left off. "
                            "Do not restart - just output the remaining JSON to complete the object. "
                            f"Your response ended with: ...{raw_text[-200:]}"
                        )

                        # Add previous response and continuation request to conversation
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part(text=raw_text)],
                        ))
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=continuation_prompt)],
                        ))

                        continuation_response = client.models.generate_content(
                            model=get_model_name(),
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                max_output_tokens=65536,
                            ),
                        )

                        continuation_text = continuation_response.text.strip()

                        # Append continuation to original (avoiding duplicate overlap)
                        raw_text = raw_text + continuation_text

                        # Update debug file
                        with open(debug_path, "w") as f:
                            f.write(raw_text)
                    else:
                        # Not truncated or max retries reached - re-raise
                        raise json_err

            if text_results:
                results.update(text_results)

            log_event(waiver_key, {
                "agent": "reextraction_agent",
                "action": "text_path_complete",
                "attempt": attempt,
                "fields_returned": list(text_results.keys()) if text_results else [],
            })

        except json.JSONDecodeError as e:
            log_event(waiver_key, {
                "agent": "reextraction_agent",
                "action": "text_path_json_error",
                "attempt": attempt,
                "error": str(e),
            })

        except Exception as e:
            log_event(waiver_key, {
                "agent": "reextraction_agent",
                "action": "text_path_error",
                "attempt": attempt,
                "error": str(e),
            })

    # === IMAGE PATH: Process each checkbox field with page images ===
    if image_path_fields:
        for field in image_path_fields:
            field_result = _process_image_path_field(
                client=client,
                pdf_path=pdf_path,
                field=field,
                checkbox_cache=checkbox_cache,
                waiver_key=waiver_key,
                attempt=attempt
            )
            results.update(field_result)

    # Log completion
    log_event(waiver_key, {
        "agent": "reextraction_agent",
        "action": "reextraction_complete",
        "attempt": attempt,
        "total_fields_returned": len([k for k in results.keys() if not k.endswith(("_evidence", "_page"))]),
    })

    return results


def merge_extractions(
    base_extraction: dict,
    passing_fields: list[dict],
    reextracted: dict
) -> dict:
    """
    Merge re-extracted fields with frozen passing fields from base extraction.

    Preserves _evidence and _page keys for re-extracted fields.

    Args:
        base_extraction: The full extraction from Attempt 1
        passing_fields: List of passing field dicts from Attempt 1 critic
        reextracted: The re-extracted subset dict (includes _evidence and _page keys)

    Returns:
        Merged dict with passing fields frozen and failing fields updated
    """
    # Start with base extraction
    merged = dict(base_extraction)

    # Get list of passing field names to freeze (base field names only)
    frozen_fields = {f.get("field") for f in passing_fields}

    # Update re-extracted fields (not frozen)
    # Process in groups: base field, _evidence, _page together
    processed_fields = set()

    for key, new_value in reextracted.items():
        # Determine the base field name
        if key.endswith("_evidence"):
            base_field = key[:-9]  # Remove "_evidence"
        elif key.endswith("_page"):
            base_field = key[:-5]  # Remove "_page"
        else:
            base_field = key

        # Skip if this base field is frozen
        if base_field in frozen_fields:
            continue

        # Update the value
        merged[key] = new_value
        processed_fields.add(base_field)

    return merged


def get_field_hints_for_display(
    failing_fields: list[dict],
    low_confidence_fields: list[dict],
    layout_schema: dict
) -> list[dict]:
    """
    Format field hints for display/logging purposes.

    Returns a list of dicts with field details for human review.
    """
    hints = []

    for field in failing_fields + low_confidence_fields:
        field_name = field.get("field", "")
        layout_info = layout_schema.get(field_name, {})

        hints.append({
            "field": field_name,
            "previous_value": field.get("extracted"),
            "expected_value": field.get("expected"),
            "rejection_reason": field.get("reason"),
            "match_type": field.get("match_type"),
            "section": layout_info.get("Header", "Unknown"),
            "context": layout_info.get("context", ""),
            "is_table_field": layout_info.get("isInTable", False),
            "bucket": "failing" if field in failing_fields else "low_confidence",
        })

    return hints
