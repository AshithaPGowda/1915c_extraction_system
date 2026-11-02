"""
Pipeline step implementations.
Each function runs one stage of the extraction pipeline.

Flow:
0. Checkbox resolver pre-extracts boolean fields from PDF metadata
1. Initial extraction (Attempt 1)
2. Critic Attempt 1 → AcroForm-based verification
3. Re-extraction agent for soft_flag + missing_child fields
4. Merge with frozen verified fields
5. Critic Attempt 2 on subset only
6. Critic Attempt 3 if still unresolved
7. Human queue for remaining failures
"""

import os
import json
import time
import random

from google import genai
from google.genai import types

# Load .env file only if it exists (prefer system env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


# Constants
MAX_ATTEMPTS = 3  # Now 3 attempts with AcroForm-based flow
PASS_THRESHOLD = 0.85
CONNECTION_RETRY_MAX = 3  # Max retries for connection-level errors
CONNECTION_RETRY_BASE_DELAY = 5.0  # Base delay in seconds for exponential backoff


def _is_connection_error(error: Exception) -> bool:
    """
    Check if an error is a connection-level error that should be retried.

    These are NOT HTTP status code errors (which the SDK handles),
    but low-level TCP/HTTP disconnection errors.
    """
    error_str = str(error).lower()

    # Check for common connection error patterns
    connection_patterns = [
        "disconnected",
        "connection reset",
        "connection refused",
        "connection aborted",
        "remote end closed connection",
        "server disconnected",
        "remoteprotocolerror",
        "connectionerror",
        "timeout",
        "timed out",
    ]

    for pattern in connection_patterns:
        if pattern in error_str:
            return True

    # Check for httpx-specific errors
    try:
        import httpx
        if isinstance(error, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout)):
            return True
    except ImportError:
        pass

    # Check for generic connection errors
    if isinstance(error, (ConnectionError, ConnectionResetError, ConnectionAbortedError)):
        return True

    return False


def _retry_on_connection_error(func, *args, max_retries=CONNECTION_RETRY_MAX, **kwargs):
    """
    Retry wrapper for connection-level errors with exponential backoff.

    Only retries on connection errors (TCP/HTTP disconnects), not on
    HTTP status code errors which the SDK handles internally.

    Args:
        func: The function to call
        *args: Positional arguments to pass to func
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of func(*args, **kwargs)

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not _is_connection_error(e):
                # Not a connection error - don't retry
                raise

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = CONNECTION_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"[RETRY] Connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"[RETRY] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                print(f"[RETRY] All {max_retries + 1} attempts exhausted. Last error: {e}")
                raise

    # Should not reach here, but just in case
    raise last_exception


def _get_client():
    """Get Gemini client. Checks GOOGLE_GENAI_USE_VERTEXAI env var."""
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE"

    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION", "us-central1"),
        )
    else:
        # Use Google AI API with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)


def _load_overview_prompt() -> str:
    """Load the overview agent prompt. Lazy — called at runtime."""
    from ..sub_agents.overview_prompt import OVERVIEW_AGENT_PROMPT
    return OVERVIEW_AGENT_PROMPT


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


def _call_overview_model(client, pdf_bytes: bytes, base_prompt: str) -> str:
    """
    Call the Gemini model for initial overview extraction.

    Args:
        client: Gemini client
        pdf_bytes: PDF file bytes
        base_prompt: The base overview extraction prompt

    Returns:
        Raw model response text

    Note:
        Uses _retry_on_connection_error wrapper to handle TCP/HTTP disconnects
        that the SDK's built-in retry logic doesn't catch.
    """
    def _make_request():
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
        return response.text.strip()

    return _retry_on_connection_error(_make_request)


def _apply_acroform_corrections(extraction: dict, critique: dict) -> tuple[dict, list[dict]]:
    """
    Apply corrected_value from AcroForm-based critic to the extraction.

    When AcroForm verification shows the LLM extraction was wrong,
    the corrected bucket contains the authoritative value from AcroForm.

    Args:
        extraction: The extraction dict to update
        critique: The critique dict containing corrected fields

    Returns:
        Tuple of (updated extraction dict, list of applied corrections)
    """
    updated = extraction.copy()
    corrections_applied = []

    for entry in critique.get("corrected", []):
        field_name = entry.get("field")
        corrected_value = entry.get("corrected_value")

        if field_name and corrected_value is not None:
            original_value = updated.get(field_name)

            # Apply correction
            updated[field_name] = corrected_value

            corrections_applied.append({
                "field": field_name,
                "original_value": original_value,
                "corrected_value": corrected_value,
                "correction_source": "acroform",
                "acroform_value": entry.get("acroform_value"),
                "inverted_field": entry.get("inverted_field", False),
            })

    return updated, corrections_applied


def _find_ancestor_page(
    field_name: str,
    extracted_data: dict,
    acroform_tree,
) -> int | None:
    """
    Find page number from nearest ancestor that has a _page value.

    Walks up the ancestor chain of the field's AcroForm path,
    looking for sibling fields that were extracted with page numbers.

    Args:
        field_name: Schema field name
        extracted_data: Extraction dict with _page keys
        acroform_tree: AcroFormTree instance

    Returns:
        Page number (1-indexed) or None if not found
    """
    from ..parsers import get_acroform_path, SCHEMA_TO_ACROFORM

    acroform_path = get_acroform_path(field_name)
    if not acroform_path:
        return None

    # Get parent path
    parts = acroform_path.split(":")
    if len(parts) < 2:
        return None

    parent_path = ":".join(parts[:-1])

    # Find all schema fields that share this parent
    for schema_field, af_path in SCHEMA_TO_ACROFORM.items():
        if af_path.startswith(parent_path + ":") or af_path == parent_path:
            # Check if this sibling field has a page number
            page = extracted_data.get(f"{schema_field}_page")
            if page is not None:
                try:
                    # Convert to int (LLM may return strings like "5" instead of 5)
                    return int(page)
                except (ValueError, TypeError) as e:
                    import logging
                    logging.warning(
                        f"_find_ancestor_page: Could not convert page '{page}' "
                        f"(type: {type(page).__name__}) for field '{schema_field}': {e}"
                    )
                    continue  # Try next sibling

    # Walk up further if no sibling found
    if len(parts) >= 3:
        grandparent_path = ":".join(parts[:-2])
        for schema_field, af_path in SCHEMA_TO_ACROFORM.items():
            if af_path.startswith(grandparent_path + ":"):
                page = extracted_data.get(f"{schema_field}_page")
                if page is not None:
                    try:
                        return int(page)
                    except (ValueError, TypeError) as e:
                        import logging
                        logging.warning(
                            f"_find_ancestor_page: Could not convert page '{page}' "
                            f"(type: {type(page).__name__}) for field '{schema_field}': {e}"
                        )
                        continue  # Try next sibling

    return None


def _build_reextraction_context(
    critique: dict,
    extracted_data: dict,
    acroform_tree,
    waiver_key: str | None = None,
) -> list[dict]:
    """
    Build rich context for re-extraction attempt.

    For each field needing re-extraction (soft_flag + missing_child),
    include AcroForm context, page window, and failure reason.
    """
    from ..parsers import SCHEMA_TO_ACROFORM, get_acroform_path
    from ..sub_agents.critic_agent import get_page_window
    from ..shared_libraries.file_utils import log_event

    fields_to_reextract = []

    # Process soft_flag fields
    for field in critique.get("soft_flag", []):
        try:
            field_name = field.get("field")
            acroform_path = get_acroform_path(field_name)

            context = {
                "field": field_name,  # Use "field" key to match reextraction_agent expectations
                "field_name": field_name,  # Keep for backwards compatibility
                "failure_reason": field.get("conflict_type") or field.get("reason", "soft_flag"),
                "acroform_value": field.get("acroform_value"),
                "llm_value": field.get("llm_value") or extracted_data.get(field_name),
                "pages": get_page_window(extracted_data, field_name),
                "needs_image_ocr": field.get("needs_image_ocr", False),  # Pass through for image OCR routing
            }

            # Add AcroForm subtree for context
            if acroform_path and acroform_tree.has_acroform:
                parts = acroform_path.split(":")
                if len(parts) >= 2:
                    parent_path = ":".join(parts[:-1])
                    try:
                        context["acroform_subtree"] = acroform_tree.get_subtree(parent_path)
                        context["ancestor_path"] = acroform_tree.get_ancestor_chain(acroform_path)
                        context["parent_path"] = parent_path
                    except Exception:
                        pass

            fields_to_reextract.append(context)

        except Exception as e:
            import logging
            logging.error(f"Error processing soft_flag field '{field}': {e}")
            if waiver_key:
                log_event(waiver_key, {
                    "agent": "pipeline",
                    "action": "build_context_error",
                    "field_type": "soft_flag",
                    "field": str(field),
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
            continue  # Skip this field but continue with others

    # Process missing_child fields
    for field in critique.get("missing_child", []):
        try:
            field_name = field.get("field")
            acroform_path = field.get("acroform_path") or get_acroform_path(field_name)

            # Try to find page from ancestor fields
            ancestor_page = _find_ancestor_page(field_name, extracted_data, acroform_tree)

            if ancestor_page is not None:
                # Build page window around ancestor's page (ancestor_page is guaranteed to be int)
                pages = [max(1, ancestor_page - 1), ancestor_page, ancestor_page + 1]
            else:
                pages = []  # Will add note in prompt about unknown location

            # Determine parent path for missing_child guidance
            parent_path = None
            if acroform_path:
                parts = acroform_path.split(":")
                if len(parts) >= 2:
                    parent_path = ":".join(parts[:-1])

            context = {
                "field": field_name,  # Use "field" key to match reextraction_agent expectations
                "field_name": field_name,  # Keep for backwards compatibility
                "failure_reason": "missing_child",
                "acroform_value": None,  # Not extracted yet
                "llm_value": None,
                "pages": pages,
                "parent_path": parent_path,
            }

            if acroform_path and acroform_tree.has_acroform:
                parts = acroform_path.split(":")
                if len(parts) >= 2:
                    try:
                        context["acroform_subtree"] = acroform_tree.get_subtree(parent_path)
                    except Exception:
                        pass

            fields_to_reextract.append(context)

        except Exception as e:
            import logging
            logging.error(f"Error processing missing_child field '{field}': {e}")
            if waiver_key:
                log_event(waiver_key, {
                    "agent": "pipeline",
                    "action": "build_context_error",
                    "field_type": "missing_child",
                    "field": str(field),
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
            continue  # Skip this field but continue with others

    return fields_to_reextract


def _format_critique_breakdown(critique: dict) -> dict:
    """Format critique buckets into a summary breakdown."""
    return {
        "verified_acroform": len(critique.get("verified_acroform", [])),
        "verified_body": len(critique.get("verified_body", [])),
        "corrected": len(critique.get("corrected", [])),
        "soft_flag": len(critique.get("soft_flag", [])),
        "hallucinated": len(critique.get("hallucinated", [])),
        "missing_child": len(critique.get("missing_child", [])),
        "human_queue": len(critique.get("human_queue", [])),
    }


async def run_overview_extraction(pdf_path: str, waiver_key: str) -> dict:
    """
    Run the overview extraction pipeline with AcroForm-based validation.

    Pipeline:
    1. Initial extraction (Attempt 1) with full PDF
    2. Critic Attempt 1 → AcroForm verification buckets
    3. Re-extraction agent for soft_flag + missing_child fields
    4. Merge with frozen verified fields from Attempt 1
    5. Critic Attempt 2/3 if still unresolved
    6. Human queue for remaining failures

    Returns result dict with status, accuracy, and output paths.
    """
    from ..shared_libraries.file_utils import write_json_output, log_event, load_layout
    from ..tools.overview_tools import save_overview_output
    from ..sub_agents.critic_agent import run_critic, get_page_window
    from ..sub_agents.reextraction_agent import (
        run_reextraction,
        merge_extractions,
        get_field_hints_for_display,
    )
    from ..sub_agents.normalisation_agent import run_normalisation
    from ..sub_agents.checkbox_resolver import resolve_all_checkboxes, get_resolver_summary
    from ..storage import save_extraction, save_corrections
    from ..parsers import AcroFormTree, get_metadata

    client = _get_client()
    base_prompt = _load_overview_prompt()
    pdf_bytes = _load_pdf_bytes(pdf_path)
    layout_schema = load_layout("layout_waiver_overview")

    # Initialize AcroForm tree for the pipeline
    acroform_tree = AcroFormTree(pdf_path)
    pdf_metadata = get_metadata(pdf_path)

    # =========================================================================
    # CHECKBOX RESOLVER: Pre-extract boolean fields from PDF metadata
    # =========================================================================
    checkbox_cache = resolve_all_checkboxes(
        pdf_path=pdf_path,
        layout=layout_schema,
        page_hints=None,  # No page hints yet - first pass
        waiver_key=waiver_key,
    )

    resolver_summary = get_resolver_summary(checkbox_cache)
    log_event(waiver_key, {
        "agent": "checkbox_resolver",
        "action": "pre_extraction_complete",
        "checkbox_fields_total": len(checkbox_cache),
        "resolved_acroform": resolver_summary["acroform"],
        "resolved_spatial": resolver_summary["spatial"],
        "no_resolution": resolver_summary["none"],
    })

    # =========================================================================
    # ATTEMPT 1: Initial extraction
    # =========================================================================
    log_event(waiver_key, {
        "agent": "waiver_overview_agent",
        "action": "extraction_start",
        "pdf_path": pdf_path,
        "attempt": 1,
        "is_cms_form": pdf_metadata.get("is_cms_form", False),
        "has_acroform": acroform_tree.has_acroform,
    })

    try:
        raw_text = _call_overview_model(client, pdf_bytes, base_prompt)
    except Exception as e:
        log_event(waiver_key, {
            "agent": "waiver_overview_agent",
            "action": "model_error",
            "attempt": 1,
            "error": str(e),
        })
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "error": str(e),
        }

    # Debug: write raw response (unique per waiver)
    from ..shared_libraries.paths import OUTPUTS_DIR
    debug_dir = str(OUTPUTS_DIR)
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, f"debug_{waiver_key}_attempt1_raw.txt"), "w") as f:
        f.write(raw_text)

    # Parse JSON response
    try:
        extracted_attempt1 = _parse_json_response(raw_text)
    except json.JSONDecodeError as e:
        log_event(waiver_key, {
            "agent": "waiver_overview_agent",
            "action": "json_parse_error",
            "attempt": 1,
            "error": str(e),
            "raw_response_preview": raw_text[:500],
        })
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "error": f"JSON parse error: {e}",
        }

    log_event(waiver_key, {
        "agent": "waiver_overview_agent",
        "action": "extraction_complete",
        "attempt": 1,
        "fields_extracted": len([v for v in extracted_attempt1.values() if v is not None]),
    })

    # =========================================================================
    # CRITIC ATTEMPT 1: AcroForm-based verification
    # =========================================================================
    try:
        critique1 = run_critic(
            pdf_path=pdf_path,
            extracted_data=extracted_attempt1,
            attempt=1,
            waiver_key=waiver_key,
        )
    except Exception as e:
        log_event(waiver_key, {
            "agent": "critic_agent",
            "action": "error",
            "attempt": 1,
            "error": str(e),
        })
        critique1 = {
            "verified_acroform": [],
            "verified_body": [],
            "corrected": [],
            "soft_flag": [],
            "hallucinated": [],
            "missing_child": [],
            "human_queue": [],
            "accuracy": 0.0,
            "decision": "attempt2",
            "error": str(e),
        }

    breakdown1 = _format_critique_breakdown(critique1)

    log_event(waiver_key, {
        "agent": "pipeline",
        "action": "critique1_complete",
        "breakdown": breakdown1,
        "accuracy": critique1.get("accuracy", 0),
        "decision": critique1.get("decision", "unknown"),
    })

    # Apply AcroForm corrections from Attempt 1
    corrected_extraction, corrections_a1 = _apply_acroform_corrections(
        extracted_attempt1, critique1
    )
    if corrections_a1:
        log_event(waiver_key, {
            "agent": "pipeline",
            "action": "acroform_corrections_applied",
            "attempt": 1,
            "corrections": corrections_a1,
        })

    # Initialize for tracking
    all_critiques = [critique1]
    final_extracted = corrected_extraction
    reextracted = {}
    corrections_all = corrections_a1.copy()

    # Check if we passed on first attempt
    if critique1.get("decision") == "pass":
        log_event(waiver_key, {
            "agent": "pipeline",
            "action": "passed_attempt1",
            "accuracy": critique1["accuracy"],
        })
    else:
        # =====================================================================
        # ATTEMPT 2: Re-extraction for soft_flag + missing_child fields
        # =====================================================================
        needs_reextraction = (
            critique1.get("soft_flag", []) +
            critique1.get("missing_child", [])
        )

        if needs_reextraction:
            # Build rich context for re-extraction
            reextraction_context = _build_reextraction_context(
                critique1, corrected_extraction, acroform_tree, waiver_key
            )

            log_event(waiver_key, {
                "agent": "pipeline",
                "action": "reextraction_fields",
                "attempt": 2,
                "fields_count": len(reextraction_context),
                "fields": [f["field_name"] for f in reextraction_context],
            })

            # Convert to format expected by reextraction agent
            # (for now, use the existing interface)
            failing_fields = [
                {"field": f["field_name"], "reason": f["failure_reason"]}
                for f in reextraction_context
                if f["failure_reason"] != "missing_child"
            ]
            low_confidence_fields = [
                {"field": f["field_name"], "reason": f["failure_reason"]}
                for f in reextraction_context
                if f["failure_reason"] == "missing_child"
            ]

            # Run re-extraction with rich context
            reextracted = run_reextraction(
                waiver_key=waiver_key,
                pdf_path=pdf_path,
                failing_fields=failing_fields,
                low_confidence_fields=low_confidence_fields,
                attempt=2,
                checkbox_cache=checkbox_cache,
                fields_context=reextraction_context,
            )

            # Merge with verified fields from Attempt 1
            verified_fields = (
                critique1.get("verified_acroform", []) +
                critique1.get("verified_body", []) +
                critique1.get("corrected", [])  # Corrected fields are also "verified"
            )
            merged_extraction = merge_extractions(
                base_extraction=corrected_extraction,
                passing_fields=verified_fields,
                reextracted=reextracted,
            )

            log_event(waiver_key, {
                "agent": "pipeline",
                "action": "merge_complete",
                "attempt": 2,
                "frozen_fields_count": len(verified_fields),
                "reextracted_fields_count": len(reextracted),
            })

            # Critic Attempt 2 on merged extraction
            try:
                critique2 = run_critic(
                    pdf_path=pdf_path,
                    extracted_data=merged_extraction,
                    attempt=2,
                    waiver_key=waiver_key,
                )
            except Exception as e:
                log_event(waiver_key, {
                    "agent": "critic_agent",
                    "action": "error",
                    "attempt": 2,
                    "error": str(e),
                })
                critique2 = {
                    "verified_acroform": [],
                    "verified_body": [],
                    "corrected": [],
                    "soft_flag": needs_reextraction,  # Still failing
                    "hallucinated": [],
                    "missing_child": [],
                    "human_queue": [],
                    "accuracy": 0.0,
                    "decision": "attempt3",
                    "error": str(e),
                }

            all_critiques.append(critique2)
            breakdown2 = _format_critique_breakdown(critique2)

            log_event(waiver_key, {
                "agent": "pipeline",
                "action": "critique2_complete",
                "breakdown": breakdown2,
                "accuracy": critique2.get("accuracy", 0),
                "decision": critique2.get("decision", "unknown"),
            })

            # Apply corrections from Attempt 2
            final_extracted, corrections_a2 = _apply_acroform_corrections(
                merged_extraction, critique2
            )
            corrections_all.extend(corrections_a2)

            # =====================================================================
            # ATTEMPT 3: If still unresolved
            # =====================================================================
            if critique2.get("decision") == "attempt3":
                still_needs_work = (
                    critique2.get("soft_flag", []) +
                    critique2.get("missing_child", [])
                )

                if still_needs_work:
                    log_event(waiver_key, {
                        "agent": "pipeline",
                        "action": "attempt3_start",
                        "fields_count": len(still_needs_work),
                    })

                    # Build context for attempt 3
                    reextraction_context3 = _build_reextraction_context(
                        critique2, final_extracted, acroform_tree, waiver_key
                    )

                    failing_fields3 = [
                        {"field": f["field_name"], "reason": f["failure_reason"]}
                        for f in reextraction_context3
                    ]

                    reextracted3 = run_reextraction(
                        waiver_key=waiver_key,
                        pdf_path=pdf_path,
                        failing_fields=failing_fields3,
                        low_confidence_fields=[],
                        attempt=3,
                        checkbox_cache=checkbox_cache,
                        fields_context=reextraction_context3,
                    )

                    # Merge again
                    verified_fields2 = (
                        critique2.get("verified_acroform", []) +
                        critique2.get("verified_body", []) +
                        critique2.get("corrected", [])
                    )
                    merged_extraction3 = merge_extractions(
                        base_extraction=final_extracted,
                        passing_fields=verified_fields2,
                        reextracted=reextracted3,
                    )

                    # Critic Attempt 3 (final)
                    try:
                        critique3 = run_critic(
                            pdf_path=pdf_path,
                            extracted_data=merged_extraction3,
                            attempt=3,
                            waiver_key=waiver_key,
                        )
                    except Exception as e:
                        critique3 = {
                            "verified_acroform": [],
                            "verified_body": [],
                            "corrected": [],
                            "soft_flag": [],
                            "hallucinated": [],
                            "missing_child": [],
                            "human_queue": still_needs_work,
                            "accuracy": 0.0,
                            "decision": "human",
                            "error": str(e),
                        }

                    all_critiques.append(critique3)
                    breakdown3 = _format_critique_breakdown(critique3)

                    log_event(waiver_key, {
                        "agent": "pipeline",
                        "action": "critique3_complete",
                        "breakdown": breakdown3,
                        "accuracy": critique3.get("accuracy", 0),
                        "decision": critique3.get("decision", "unknown"),
                        "human_queue_count": len(critique3.get("human_queue", [])),
                    })

                    # Apply final corrections
                    final_extracted, corrections_a3 = _apply_acroform_corrections(
                        merged_extraction3, critique3
                    )
                    corrections_all.extend(corrections_a3)
        else:
            # Nothing to re-extract
            log_event(waiver_key, {
                "agent": "pipeline",
                "action": "no_reextraction_needed",
                "reason": "no_soft_flag_or_missing_child",
            })

    # =========================================================================
    # NORMALIZATION & OUTPUT
    # =========================================================================
    # Get the latest critique
    latest_critique = all_critiques[-1]

    # Pass critique data to normalisation agent for evidence chain
    normalised, evidence_path = run_normalisation(
        waiver_key=waiver_key,
        final_extracted=final_extracted,
        attempt1_extracted=extracted_attempt1,
        attempt1_critique=critique1,
        attempt2_extracted=reextracted if reextracted else None,
        attempt2_critique=all_critiques[1] if len(all_critiques) > 1 else None,
    )
    output_path = save_overview_output(waiver_key, normalised)

    # Calculate final accuracy from latest critique
    final_accuracy = latest_critique.get("accuracy", 0.0)
    final_decision = latest_critique.get("decision", "unknown")

    # Determine final status
    if final_decision == "pass":
        final_status = "success"
    elif final_decision == "human":
        final_status = "human_review_required"
    elif final_accuracy >= PASS_THRESHOLD:
        final_status = "success"
    else:
        final_status = "validation_failed"

    # Build result
    norm_meta = normalised.get("_normalisation_meta", {})

    result = {
        "status": final_status,
        "waiver_key": waiver_key,
        "pdf_path": pdf_path,
        "output_path": output_path,
        "evidence_path": evidence_path,
        "fields_extracted": len([k for k, v in normalised.items() if v is not None and not k.startswith("_")]),
        "attempt1_accuracy": critique1.get("accuracy", 0),
        "final_accuracy": final_accuracy,
        "attempts_used": len(all_critiques),
        "final_decision": final_decision,
        "acroform_corrections_count": len(corrections_all),
        "norm_errors": norm_meta.get("norm_errors_count", 0),
        "unknown_fields": norm_meta.get("unknown_fields_count", 0),
        "checkbox_resolver_summary": resolver_summary,
        "is_cms_form": pdf_metadata.get("is_cms_form", False),
        "has_acroform": acroform_tree.has_acroform,
    }

    # Add breakdown for each attempt
    for i, critique in enumerate(all_critiques, 1):
        result[f"critique{i}_breakdown"] = _format_critique_breakdown(critique)

    if corrections_all:
        result["acroform_corrections"] = corrections_all

    # Collect human queue fields
    human_queue = latest_critique.get("human_queue", [])
    if human_queue:
        result["human_queue_count"] = len(human_queue)
        result["human_queue_fields"] = [f.get("field") for f in human_queue]
        log_event(waiver_key, {
            "agent": "pipeline",
            "action": "human_queue_populated",
            "count": len(human_queue),
            "fields": result["human_queue_fields"],
        })

    # Collect hallucinated fields
    hallucinated = []
    for critique in all_critiques:
        hallucinated.extend(critique.get("hallucinated", []))
    if hallucinated:
        result["hallucinated_count"] = len(hallucinated)
        result["hallucinated_fields"] = [f.get("field") for f in hallucinated]

    # Save to storage layer (sqlite first, then chroma separately)
    sqlite_rows = 0
    chroma_docs = 0

    # SQLite storage (primary)
    try:
        sqlite_rows = save_extraction(waiver_key, "overview", normalised, result)
    except Exception as sql_err:
        log_event(waiver_key, {
            "agent": "storage",
            "action": "sqlite_error",
            "error": str(sql_err),
        })

    # ChromaDB storage (optional, failures are non-fatal)
    # Use BaseException to catch Rust panics from pyo3_runtime.PanicException
    try:
        chroma_docs = save_corrections(waiver_key, all_critiques, "overview")
    except BaseException as chroma_err:
        log_event(waiver_key, {
            "agent": "storage",
            "action": "chroma_error",
            "error": str(chroma_err),
        })
        chroma_docs = 0

    if sqlite_rows > 0 or chroma_docs > 0:
        log_event(waiver_key, {
            "agent": "storage",
            "action": "storage_complete",
            "sqlite_rows": sqlite_rows,
            "chroma_documents": chroma_docs,
        })

    result["storage"] = {
        "sqlite_rows": sqlite_rows,
        "chroma_documents": chroma_docs,
    }

    # Final outcome log
    log_event(waiver_key, {
        "agent": "pipeline",
        "action": "extraction_complete",
        "final_status": final_status,
        "final_accuracy": final_accuracy,
        "attempts_used": len(all_critiques),
        "human_queue_count": len(human_queue),
        "acroform_corrections_count": len(corrections_all),
        "output_path": output_path,
    })

    return result
