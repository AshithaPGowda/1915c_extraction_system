"""
Checkbox Resolver Module.

Three-layer resolution strategy for boolean/checkbox fields:
1. AcroForm metadata (pypdf) - highest confidence
2. Spatial analysis (pdfplumber) - medium confidence
3. No result - flags for image OCR in Attempt 2

This module runs BEFORE extraction to pre-resolve checkbox values,
providing ground truth hints to the critic agent.
"""

import os
import re
from typing import Any
from pathlib import Path

from ..shared_libraries.file_utils import log_event
from ..parsers import SCHEMA_TO_ACROFORM


# Module-level cache: (pdf_path, field_name) -> resolution result
_checkbox_cache: dict[tuple[str, str], dict] = {}

# Build reverse mapping: AcroForm path (lowercase) -> schema field name
ACROFORM_TO_SCHEMA: dict[str, str] = {
    path.lower(): field for field, path in SCHEMA_TO_ACROFORM.items()
}

# Checkbox character patterns
CHECKED_PATTERNS = {'●', '■', '☒', '✓', '✗', '☑', 'X', 'x', '✔', '◉', '▣'}
UNCHECKED_PATTERNS = {'○', '□', '☐', '☑', '◯', '▢', ' '}


def _is_checkbox_field(field_name: str, field_info: dict) -> bool:
    """
    Determine if a field is a boolean/checkbox field.

    Checks:
    - Field info has type "boolean"
    - Field name contains (0=No or (0 = No patterns
    """
    field_lower = field_name.lower()
    context = (field_info.get("context", "") or "").lower()

    # Check for binary field indicators
    if "binary field" in context:
        return True

    # Check for coding patterns in field name
    if "(0 = no" in field_lower or "(0=no" in field_lower:
        return True
    if "(0 = no;" in field_lower or "(0=no;" in field_lower:
        return True
    if "(1 = yes" in field_lower or "(1=yes" in field_lower:
        return True
    if "(0=yes" in field_lower or "(0 = yes" in field_lower:
        return True  # Miller Trust inverted

    return False


def _get_checkbox_fields(layout: dict) -> set[str]:
    """Get all checkbox field names from the layout schema."""
    checkbox_fields = set()
    for field_name, field_info in layout.items():
        if _is_checkbox_field(field_name, field_info):
            checkbox_fields.add(field_name)
    return checkbox_fields


def _normalize_for_matching(name: str) -> str:
    """Strip all non-alphanumeric chars and lowercase for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _normalize_acroform_value(raw_value: Any) -> int | None:
    """
    Normalize AcroForm checkbox value to 0 or 1.

    /Yes, /1, /On -> 1
    /Off, /0, /No -> 0
    Anything else -> None
    """
    if raw_value is None:
        return None

    val = str(raw_value).strip().lower().lstrip("/")

    if val in ("yes", "1", "on", "true"):
        return 1
    elif val in ("off", "0", "no", "false", ""):
        return 0
    else:
        # Try to parse as number
        try:
            return 1 if int(val) else 0
        except (ValueError, TypeError):
            return None


def _get_radio_button_selection(field_obj) -> int | None:
    """
    Check radio button group selection by examining Kids' /AS values.

    For radio buttons, the parent /V value can be stale. The actual
    selection state is in the Kids' /AS (Appearance State) values.

    Returns:
        1 if any kid is selected (AS != /Off)
        0 if all kids have AS = /Off
        None if no Kids or can't determine
    """
    if "/Kids" not in field_obj:
        return None

    try:
        kids = field_obj["/Kids"]
        if not kids:
            return None

        for kid in kids:
            kid_obj = kid.get_object() if hasattr(kid, "get_object") else kid
            as_value = kid_obj.get("/AS", None)

            if as_value is not None:
                as_str = str(as_value).strip().lower().lstrip("/")
                # If any kid is NOT /Off, it's selected
                if as_str not in ("off", ""):
                    return 1

        # All kids are /Off - nothing selected
        return 0
    except Exception:
        return None


# NOTE: Field mappings are now centralized in parsers/field_mapping.py
# ACROFORM_TO_SCHEMA is built from SCHEMA_TO_ACROFORM as a reverse lookup


def _match_acroform_to_schema(acroform_name: str, schema_fields: set[str]) -> str | None:
    """
    Try to match an AcroForm field name to a schema field name.

    Uses:
    1. Direct lookup in centralized ACROFORM_TO_SCHEMA mapping
    2. Substring matching on normalized names as fallback
    """
    acro_lower = acroform_name.lower()
    acro_norm = _normalize_for_matching(acroform_name)

    # Strategy 1: Direct lookup in centralized mapping
    if acro_lower in ACROFORM_TO_SCHEMA:
        schema_field = ACROFORM_TO_SCHEMA[acro_lower]
        if schema_field in schema_fields:
            return schema_field

    # Strategy 2: Check if any mapping path contains this name
    for acro_path, schema_field in ACROFORM_TO_SCHEMA.items():
        if acro_lower in acro_path or acro_path in acro_lower:
            if schema_field in schema_fields:
                return schema_field

    # Strategy 3: Substring matching on normalized names (fallback)
    for schema_field in schema_fields:
        schema_norm = _normalize_for_matching(schema_field)

        # Check if either contains the other as substring
        if acro_norm in schema_norm or schema_norm in acro_norm:
            return schema_field

        # Also check partial matches for common patterns
        if len(acro_norm) >= 4 and len(schema_norm) >= 4:
            if acro_norm[:4] in schema_norm or schema_norm[:4] in acro_norm:
                return schema_field

    return None


# =============================================================================
# LAYER 1: AcroForm Metadata (pypdf)
# =============================================================================

def _resolve_via_acroform(pdf_path: str, checkbox_fields: set[str]) -> dict[str, dict]:
    """
    Extract checkbox values from AcroForm metadata.

    Returns dict mapping schema field name -> {value, method, confidence}
    """
    results = {}

    try:
        from pypdf import PdfReader
    except ImportError:
        return results

    try:
        reader = PdfReader(pdf_path)

        # Check if PDF has AcroForm
        try:
            root = reader.trailer["/Root"]
            if hasattr(root, "get_object"):
                root = root.get_object()
            has_acroform = "/AcroForm" in root if hasattr(root, "__contains__") else False
        except Exception:
            has_acroform = False

        if not has_acroform:
            return results

        # Collect all button fields from the PDF with field objects
        # Store: field_name -> (raw_value, field_obj)
        acroform_buttons = {}

        # Method 1: Get fields from AcroForm/Fields directly (preserves Kids)
        try:
            acroform = root["/AcroForm"]
            if hasattr(acroform, "get_object"):
                acroform = acroform.get_object()

            if "/Fields" in acroform:
                for field in acroform["/Fields"]:
                    try:
                        field_obj = field.get_object() if hasattr(field, "get_object") else field
                        field_type = field_obj.get("/FT")

                        if field_type == "/Btn":
                            field_name = field_obj.get("/T", "")
                            if hasattr(field_name, "get_object"):
                                field_name = field_name.get_object()
                            field_name = str(field_name) if field_name else ""

                            if field_name:
                                # Check if this is a radio button group with Kids
                                radio_value = _get_radio_button_selection(field_obj)

                                if radio_value is not None:
                                    # Use Kids' AS values (more reliable for radio buttons)
                                    acroform_buttons[field_name] = (radio_value, "radio_kids")
                                else:
                                    # Fall back to /V or /AS value
                                    value = field_obj.get("/V") or field_obj.get("/AS") or ""
                                    if hasattr(value, "name"):
                                        value = value.name
                                    acroform_buttons[field_name] = (str(value), "direct")
                    except Exception:
                        continue
        except Exception:
            pass

        # Method 2: Scan page annotations for widget buttons (fallback)
        for page_num, page in enumerate(reader.pages):
            try:
                if "/Annots" in page:
                    annots = page["/Annots"]
                    if hasattr(annots, "get_object"):
                        annots = annots.get_object()
                    if annots:
                        for annot in annots:
                            try:
                                annot_obj = annot.get_object() if hasattr(annot, "get_object") else annot

                                subtype = annot_obj.get("/Subtype")
                                if subtype == "/Widget":
                                    field_type = annot_obj.get("/FT")
                                    if field_type == "/Btn":
                                        field_name = annot_obj.get("/T", "")
                                        if hasattr(field_name, "get_object"):
                                            field_name = field_name.get_object()
                                        field_name = str(field_name) if field_name else ""

                                        if field_name and field_name not in acroform_buttons:
                                            # For annotations, check Kids first
                                            radio_value = _get_radio_button_selection(annot_obj)
                                            if radio_value is not None:
                                                acroform_buttons[field_name] = (radio_value, "radio_kids")
                                            else:
                                                value = annot_obj.get("/V") or annot_obj.get("/AS") or ""
                                                if hasattr(value, "name"):
                                                    value = value.name
                                                acroform_buttons[field_name] = (str(value), "direct")
                            except Exception:
                                continue
            except Exception:
                continue

        # Match AcroForm fields to schema fields
        for acro_name, (raw_value, source) in acroform_buttons.items():
            schema_field = _match_acroform_to_schema(acro_name, checkbox_fields)
            if schema_field:
                # For radio_kids source, value is already 0 or 1
                if source == "radio_kids":
                    normalized_value = raw_value
                    raw_display = f"kids_as={raw_value}"
                else:
                    normalized_value = _normalize_acroform_value(raw_value)
                    raw_display = raw_value

                if normalized_value is not None:
                    results[schema_field] = {
                        "value": normalized_value,
                        "method": "acroform",
                        "confidence": "high",
                        "raw_acroform_name": acro_name,
                        "raw_acroform_value": raw_display,
                        "acroform_source": source,
                    }

    except Exception as e:
        # Log error but don't fail
        pass

    return results


# =============================================================================
# LAYER 2: Spatial Analysis (pdfplumber)
# =============================================================================

def _resolve_via_spatial(
    pdf_path: str,
    checkbox_fields: set[str],
    already_resolved: set[str],
    page_hints: dict[str, int] | None = None
) -> dict[str, dict]:
    """
    Use pdfplumber to find checkbox characters near field labels.

    Returns dict mapping schema field name -> {value, method, confidence}
    """
    results = {}

    try:
        import pdfplumber
    except ImportError:
        return results

    # Fields to resolve (not already resolved by AcroForm)
    fields_to_resolve = checkbox_fields - already_resolved
    if not fields_to_resolve:
        return results

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for field_name in fields_to_resolve:
                # Determine which pages to search
                if page_hints and field_name in page_hints:
                    # Convert to int (LLM may return strings like "5" instead of 5)
                    pages_to_search = [int(page_hints[field_name]) - 1]  # 0-indexed
                else:
                    pages_to_search = range(len(pdf.pages))

                # Search key: first 30 chars of field name
                search_key = _normalize_for_matching(field_name[:30])

                found = False
                for page_idx in pages_to_search:
                    if page_idx < 0 or page_idx >= len(pdf.pages):
                        continue

                    page = pdf.pages[page_idx]

                    try:
                        words = page.extract_words(
                            x_tolerance=3,
                            y_tolerance=3,
                            keep_blank_chars=True
                        )

                        # Find word cluster that matches field label
                        for word in words:
                            word_text = word.get("text", "")
                            word_norm = _normalize_for_matching(word_text)

                            # Check if this word matches part of our search key
                            if len(word_norm) >= 3 and (word_norm in search_key or search_key[:5] in word_norm):
                                # Found potential label - look for checkbox character
                                label_x0 = word.get("x0", 0)
                                label_y0 = word.get("top", 0)
                                label_y1 = word.get("bottom", 0)

                                # Look at characters within 50px to the LEFT
                                chars = page.chars
                                checkbox_char = None

                                for char in chars:
                                    char_x = char.get("x0", 0)
                                    char_y = (char.get("top", 0) + char.get("bottom", 0)) / 2
                                    label_y_mid = (label_y0 + label_y1) / 2

                                    # Within 50px to the left and at same y-level (within 10px)
                                    if (label_x0 - 50 <= char_x < label_x0 and
                                        abs(char_y - label_y_mid) < 10):
                                        char_text = char.get("text", "")

                                        if char_text in CHECKED_PATTERNS:
                                            checkbox_char = 1
                                            break
                                        elif char_text in UNCHECKED_PATTERNS:
                                            checkbox_char = 0

                                # Also check the label's own text for inline checkbox
                                if checkbox_char is None:
                                    for ch in word_text:
                                        if ch in CHECKED_PATTERNS:
                                            checkbox_char = 1
                                            break
                                        elif ch in UNCHECKED_PATTERNS:
                                            checkbox_char = 0

                                if checkbox_char is not None:
                                    results[field_name] = {
                                        "value": checkbox_char,
                                        "method": "spatial",
                                        "confidence": "medium",
                                        "page": page_idx + 1,
                                        "label_found": word_text[:30],
                                    }
                                    found = True
                                    break
                                else:
                                    # Label found but no clear checkbox
                                    if field_name not in results:
                                        results[field_name] = {
                                            "value": None,
                                            "method": "spatial_label_found",
                                            "confidence": "low",
                                            "page": page_idx + 1,
                                            "label_found": word_text[:30],
                                        }

                        if found:
                            break

                    except Exception:
                        continue

                    if found:
                        break

    except Exception as e:
        pass

    return results


# =============================================================================
# MAIN RESOLVER FUNCTION
# =============================================================================

def resolve_all_checkboxes(
    pdf_path: str,
    layout: dict,
    page_hints: dict[str, int] | None = None,
    waiver_key: str | None = None
) -> dict[str, dict]:
    """
    Resolve all checkbox fields using three-layer strategy.

    Args:
        pdf_path: Path to the PDF file
        layout: Layout schema dict
        page_hints: Optional dict mapping field name -> page number (from Attempt 1 evidence)
        waiver_key: Optional waiver key for logging

    Returns:
        Dict mapping field_name -> {
            "value": 0 | 1 | None,
            "method": "acroform" | "spatial" | "spatial_label_found" | "none",
            "confidence": "high" | "medium" | "low" | "none"
        }
    """
    results = {}

    # Get all checkbox fields from layout
    checkbox_fields = _get_checkbox_fields(layout)

    if not checkbox_fields:
        return results

    # Check cache first
    cache_hits = 0
    fields_to_resolve = set()

    for field_name in checkbox_fields:
        cache_key = (pdf_path, field_name)
        if cache_key in _checkbox_cache:
            results[field_name] = _checkbox_cache[cache_key]
            cache_hits += 1
        else:
            fields_to_resolve.add(field_name)

    if not fields_to_resolve:
        # All from cache
        if waiver_key:
            log_event(waiver_key, {
                "agent": "checkbox_resolver",
                "action": "resolve_all",
                "checkbox_fields": len(checkbox_fields),
                "cache_hits": cache_hits,
                "cache_miss": 0,
            })
        return results

    # Layer 1: AcroForm metadata
    acroform_results = _resolve_via_acroform(pdf_path, fields_to_resolve)
    results.update(acroform_results)

    # Cache AcroForm results
    for field_name, result in acroform_results.items():
        _checkbox_cache[(pdf_path, field_name)] = result

    already_resolved = set(acroform_results.keys())

    # Layer 2: Spatial analysis (for fields not resolved by AcroForm)
    spatial_results = _resolve_via_spatial(
        pdf_path,
        fields_to_resolve,
        already_resolved,
        page_hints
    )

    for field_name, result in spatial_results.items():
        if field_name not in results:
            results[field_name] = result
            _checkbox_cache[(pdf_path, field_name)] = result

    # Layer 3: No result (for remaining fields)
    for field_name in fields_to_resolve:
        if field_name not in results:
            no_result = {
                "value": None,
                "method": "none",
                "confidence": "none",
            }
            results[field_name] = no_result
            _checkbox_cache[(pdf_path, field_name)] = no_result

    # Log summary
    if waiver_key:
        method_counts = {"acroform": 0, "spatial": 0, "spatial_label_found": 0, "none": 0}
        for result in results.values():
            method = result.get("method", "none")
            if method in method_counts:
                method_counts[method] += 1
            else:
                method_counts["none"] += 1

        log_event(waiver_key, {
            "agent": "checkbox_resolver",
            "action": "resolve_all",
            "checkbox_fields": len(checkbox_fields),
            "resolved_acroform": method_counts["acroform"],
            "resolved_spatial": method_counts["spatial"],
            "spatial_label_found": method_counts["spatial_label_found"],
            "no_resolution": method_counts["none"],
            "cache_hits": cache_hits,
        })

    return results


def get_resolver_summary(checkbox_cache: dict) -> dict:
    """
    Get a summary of resolver method counts.

    Returns:
        Dict with counts: {"acroform": N, "spatial": N, "none": N}
    """
    summary = {"acroform": 0, "spatial": 0, "none": 0}

    for result in checkbox_cache.values():
        method = result.get("method", "none")
        if method == "acroform":
            summary["acroform"] += 1
        elif method in ("spatial", "spatial_label_found"):
            summary["spatial"] += 1
        else:
            summary["none"] += 1

    return summary


def clear_cache():
    """Clear the module-level cache."""
    global _checkbox_cache
    _checkbox_cache = {}
