"""
Normalisation Agent for Overview Extraction.

Runs after critic passes to clean and standardize extracted values
before storage. Also strips evidence keys from main output and writes
a separate evidence.json file for the full audit trail.

Field Resolution Strategy (three layers):
1. Fuzzy string match - normalize field names and compare
2. LLM semantic resolution - ask Gemini to classify unknown fields
3. Flag as genuinely unknown - if neither layer resolves it
"""

import os
import re
import json
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

from ..shared_libraries.model_config import get_model_name, get_location

from ..shared_libraries.file_utils import load_layout, log_event, get_output_dir


# Fields to pass through without normalisation
PASSTHROUGH_FIELDS = {
    "_discovered_services",
    "_normalisation_meta",
}

# Module-level cache for LLM field resolutions
# Keyed by normalized field name to avoid repeated API calls
_llm_resolution_cache: dict[str, dict] = {}


# =============================================================================
# LAYER 1: Fuzzy String Matching
# =============================================================================

def _normalize_field_name(field_name: str) -> str:
    """
    Normalize a field name for fuzzy matching.

    - Strips leading/trailing whitespace
    - Collapses internal multiple whitespace to single space
    - Normalizes spacing around =, ;, , (removes spaces around them)
    - Lowercases the result

    Args:
        field_name: The field name to normalize

    Returns:
        Normalized field name for comparison
    """
    name = field_name.strip()
    # Collapse multiple whitespace to single space
    name = re.sub(r'\s+', ' ', name)
    # Normalize spacing around punctuation: remove spaces around = ; ,
    name = re.sub(r'\s*=\s*', '=', name)
    name = re.sub(r'\s*;\s*', ';', name)
    name = re.sub(r'\s*,\s*', ',', name)
    return name.lower()


def _fuzzy_match_field(field_name: str, layout: dict) -> str | None:
    """
    Try to find a fuzzy match for a field name in the layout schema.

    Args:
        field_name: The extracted field name
        layout: The layout schema dict

    Returns:
        The canonical layout field name if matched, None otherwise
    """
    normalized_extracted = _normalize_field_name(field_name)

    for layout_field in layout.keys():
        if _normalize_field_name(layout_field) == normalized_extracted:
            return layout_field

    return None


# =============================================================================
# LAYER 2: LLM Semantic Resolution
# =============================================================================

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


LLM_FIELD_RESOLUTION_PROMPT = """You are a field name resolution assistant for a document extraction system.

Given an unknown field name that was extracted from a PDF but doesn't match the schema, classify it into one of three categories:

1. **schema_match** - The field maps to an existing schema field (return the canonical_name)
   - Look for semantic equivalence, typos, spacing differences, punctuation variations
   - Example: "Waiper Population" → "Waiver Population" (typo)
   - Example: "1.F. Nursing Facility (0 = No, 1= Yes)" → "1.F. Nursing Facility (0=No, 1=Yes)" (spacing)

2. **metadata_artifact** - It's an internal/metadata key that should be stripped silently
   - Fields ending with _evidence, _confidence, _norm_error
   - Fields like "critic_reason", "extracted_fields", "low_confidence_fields", "waiver_key"
   - Any field that looks like pipeline metadata rather than document content

3. **genuinely_unknown** - No schema equivalent exists
   - The field represents data that isn't captured in the current schema
   - It's a valid document field but not defined in our extraction schema

## Unknown Field:
Name: {field_name}
Value (truncated): {value_preview}

## Schema Fields (all {schema_count} fields):
{schema_fields}

## Your Task:
Analyze the unknown field and return a JSON object with:
- classification: "schema_match" | "metadata_artifact" | "genuinely_unknown"
- canonical_name: (only if schema_match) the exact schema field name it maps to
- reason: brief explanation of your decision

Return ONLY valid JSON, no markdown fences."""


def _resolve_unknown_field_via_llm(
    field_name: str,
    value: Any,
    layout: dict,
    waiver_key: str | None = None
) -> dict:
    """
    Use LLM to classify an unknown field.

    Args:
        field_name: The unknown field name
        value: The field's value (for context)
        layout: The layout schema dict
        waiver_key: Optional waiver key for logging

    Returns:
        Dict with keys: classification, canonical_name (if schema_match), reason
    """
    # Check cache first
    cache_key = _normalize_field_name(field_name)
    if cache_key in _llm_resolution_cache:
        cached = _llm_resolution_cache[cache_key]
        if waiver_key:
            log_event(waiver_key, {
                "agent": "normalisation_agent",
                "action": "field_resolution",
                "resolution_type": "cache_hit",
                "field_name": field_name,
                "classification": cached.get("classification"),
                "canonical_name": cached.get("canonical_name"),
            })
        return cached

    # Prepare value preview (truncate to 100 chars)
    if value is None:
        value_preview = "null"
    else:
        value_str = str(value)
        value_preview = value_str[:100] + "..." if len(value_str) > 100 else value_str

    # Prepare schema fields list
    schema_fields = "\n".join(f"- {f}" for f in sorted(layout.keys()))

    # Build prompt
    prompt = LLM_FIELD_RESOLUTION_PROMPT.format(
        field_name=field_name,
        value_preview=value_preview,
        schema_count=len(layout),
        schema_fields=schema_fields,
    )

    # Call LLM
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=get_model_name(),
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,  # Increased from 1024 to handle large normalizations
            ),
        )

        raw_text = response.text.strip()

        # Parse JSON response
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[-1].strip() == "```":
                raw_text = "\n".join(lines[1:-1])
            else:
                raw_text = "\n".join(lines[1:])

        result = json.loads(raw_text)

        # Validate result structure
        if "classification" not in result:
            result = {
                "classification": "genuinely_unknown",
                "reason": f"LLM response missing classification: {raw_text[:100]}"
            }

        # Validate classification value
        valid_classifications = {"schema_match", "metadata_artifact", "genuinely_unknown"}
        if result["classification"] not in valid_classifications:
            result = {
                "classification": "genuinely_unknown",
                "reason": f"Invalid classification '{result['classification']}'"
            }

        # If schema_match, validate canonical_name exists in layout
        if result["classification"] == "schema_match":
            canonical = result.get("canonical_name")
            if not canonical or canonical not in layout:
                # Try fuzzy match on the canonical name
                if canonical:
                    fuzzy_canonical = _fuzzy_match_field(canonical, layout)
                    if fuzzy_canonical:
                        result["canonical_name"] = fuzzy_canonical
                    else:
                        result = {
                            "classification": "genuinely_unknown",
                            "reason": f"LLM suggested '{canonical}' but it's not in schema"
                        }
                else:
                    result = {
                        "classification": "genuinely_unknown",
                        "reason": "LLM said schema_match but no canonical_name provided"
                    }

    except json.JSONDecodeError as e:
        result = {
            "classification": "genuinely_unknown",
            "reason": f"LLM response not valid JSON: {str(e)}"
        }
    except Exception as e:
        result = {
            "classification": "genuinely_unknown",
            "reason": f"LLM call failed: {str(e)}"
        }

    # Cache the result
    _llm_resolution_cache[cache_key] = result

    # Log the resolution
    if waiver_key:
        log_event(waiver_key, {
            "agent": "normalisation_agent",
            "action": "field_resolution",
            "resolution_type": "llm_call",
            "field_name": field_name,
            "classification": result.get("classification"),
            "canonical_name": result.get("canonical_name"),
            "reason": result.get("reason"),
        })

    return result


def _get_field_type(field_name: str, layout: dict) -> str:
    """
    Determine the type of a field from layout schema.

    Returns one of: 'date', 'boolean', 'integer', 'float', 'text'
    """
    field_info = layout.get(field_name, {})
    context = field_info.get("context", "").lower()
    field_lower = field_name.lower()

    # Date fields
    if "date field" in context or "date" in field_lower and "effective" in field_lower:
        return "date"
    if "expiration date" in field_lower:
        return "date"

    # Boolean fields (0/1 binary)
    if "binary field" in context:
        return "boolean"
    if "(0 = no, 1= yes)" in field_lower or "(0=no, 1=yes)" in field_lower:
        return "boolean"
    if "(0 = no; 1 = yes)" in field_lower or "(0 = no; 1= yes)" in field_lower:
        return "boolean"

    # Numeric fields - participants, spending, years
    if "unduplicated" in field_lower and "participant" in field_lower:
        return "integer"
    if "total spending" in field_lower:
        return "float"
    if "year 1" in field_lower or "year 2" in field_lower:
        if "participant" in field_lower or "spending" in field_lower:
            return "float"

    # Age fields (can be integer or "No Max")
    if "(minimum age)" in field_lower or "(maximum age)" in field_lower:
        return "age"

    # Coded fields (1=x, 2=y, 3=z) - keep as integer
    if re.search(r'\(1\s*=.*2\s*=', field_lower):
        return "integer"

    # Default to text
    return "text"


def _normalise_date(value: Any) -> tuple[Any, str | None]:
    """
    Normalise a date value to ISO 8601 YYYY-MM-DD format.

    Returns:
        Tuple of (normalised_value, error_message or None)
    """
    if value is None:
        return None, None

    val_str = str(value).strip()
    if not val_str:
        return None, None

    # Already ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
        return val_str, None

    # With timestamp suffix (from pandas/Excel)
    if 'T' in val_str:
        return val_str.split('T')[0], None
    if ' 00:00:00' in val_str:
        return val_str.split(' ')[0], None

    # Common date formats to try
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%B %d, %Y',      # January 15, 2021
        '%b %d, %Y',      # Jan 15, 2021
        '%d %B %Y',       # 15 January 2021
        '%d %b %Y',       # 15 Jan 2021
        '%m-%d-%Y',
        '%d-%m-%Y',
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(val_str, fmt)
            return dt.strftime('%Y-%m-%d'), None
        except ValueError:
            continue

    # Unparseable - return original with error
    return value, f"unparseable date: {val_str}"


def _normalise_boolean(value: Any) -> Any:
    """
    Normalise boolean/binary values to integers 0 or 1.

    Note: We use 0/1 integers, not Python True/False, to match
    the schema convention for binary fields.
    """
    if value is None:
        return None

    # Already an integer
    if isinstance(value, int):
        return 1 if value else 0

    # Float that's a whole number
    if isinstance(value, float):
        if value == int(value):
            return 1 if int(value) else 0

    # String values
    if isinstance(value, str):
        val_lower = value.strip().lower()
        if val_lower in ('yes', 'true', '1', 'y'):
            return 1
        if val_lower in ('no', 'false', '0', 'n'):
            return 0

    # Boolean type
    if isinstance(value, bool):
        return 1 if value else 0

    # Can't normalise - return as-is
    return value


def _normalise_integer(value: Any) -> Any:
    """Normalise to integer, handling string numbers and currency."""
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value == int(value):
            return int(value)
        return value  # Keep as float if not whole

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return None

        # Strip currency symbols and commas
        cleaned = re.sub(r'[$,\s]', '', val_str)

        try:
            # Try integer first
            return int(cleaned)
        except ValueError:
            try:
                # Try float, then check if whole
                f = float(cleaned)
                if f == int(f):
                    return int(f)
                return f
            except ValueError:
                return value  # Return original if can't parse

    return value


def _normalise_float(value: Any) -> Any:
    """Normalise to float, handling string numbers and currency."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return None

        # Strip currency symbols and commas
        cleaned = re.sub(r'[$,\s]', '', val_str)

        try:
            return float(cleaned)
        except ValueError:
            return value  # Return original if can't parse

    return value


def _normalise_age(value: Any) -> Any:
    """
    Normalise age fields. Preserves "No Max" as-is.
    """
    if value is None:
        return None

    if isinstance(value, str):
        val_str = value.strip()

        # Preserve "No Max" variants
        if val_str.lower() in ('no max', 'no maximum', 'no limit', 'none', 'n/a', 'x'):
            return "No Max"

        # Try to parse as integer
        try:
            return int(val_str)
        except ValueError:
            try:
                f = float(val_str)
                if f == int(f):
                    return int(f)
                return f
            except ValueError:
                return val_str

    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            return int(value)
        return value

    return value


def _normalise_text(value: Any) -> Any:
    """
    Normalise text fields: strip whitespace, collapse internal newlines.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    # Strip leading/trailing whitespace
    text = value.strip()

    # Collapse multiple whitespace/newlines to single space
    text = re.sub(r'\s+', ' ', text)

    return text


def _is_evidence_key(field_name: str) -> bool:
    """Check if a field name is an evidence/metadata key that should be stripped."""
    return (
        field_name.endswith("_evidence") or
        field_name.endswith("_confidence") or
        field_name.endswith("_norm_error") or
        field_name.endswith("_page") or  # Page metadata from extraction
        field_name == "critic_reason"
    )


def _extract_evidence_from_extraction(extracted: dict) -> dict:
    """
    Extract evidence fields from the extraction dict.

    Returns a dict mapping base field name -> evidence string
    """
    evidence = {}
    for key, value in extracted.items():
        if key.endswith("_evidence"):
            base_field = key[:-9]  # Remove "_evidence" suffix
            evidence[base_field] = value
    return evidence


def _build_evidence_chain(
    waiver_key: str,
    attempt1_extracted: dict,
    attempt1_critique: dict,
    attempt2_extracted: dict | None,
    attempt2_critique: dict | None,
    final_extracted: dict
) -> dict:
    """
    Build the complete evidence chain for all fields.

    Args:
        waiver_key: The waiver key
        attempt1_extracted: Extraction from attempt 1 (with evidence)
        attempt1_critique: Critique from attempt 1
        attempt2_extracted: Extraction from attempt 2 if performed (with evidence)
        attempt2_critique: Critique from attempt 2 if performed
        final_extracted: Final merged extraction

    Returns:
        Evidence chain dict with per-field audit trail
    """
    evidence_chain = {
        "waiver_key": waiver_key,
        "generated_at": datetime.utcnow().isoformat(),
        "fields": {}
    }

    # Get evidence from attempt 1
    attempt1_evidence = _extract_evidence_from_extraction(attempt1_extracted)

    # Get evidence from attempt 2 if exists
    attempt2_evidence = {}
    if attempt2_extracted:
        attempt2_evidence = _extract_evidence_from_extraction(attempt2_extracted)

    # Build critique lookup maps
    def build_critique_map(critique: dict) -> dict:
        """Build field -> critique entry map from all buckets."""
        result = {}
        for bucket in ["passing", "failing", "low_confidence"]:
            for entry in critique.get(bucket, []):
                field = entry.get("field")
                if field:
                    result[field] = entry
        return result

    critique1_map = build_critique_map(attempt1_critique) if attempt1_critique else {}
    critique2_map = build_critique_map(attempt2_critique) if attempt2_critique else {}

    # Process all fields from final extraction
    for field_name, final_value in final_extracted.items():
        # Skip internal and evidence keys
        if field_name.startswith("_") or _is_evidence_key(field_name):
            continue

        field_entry = {
            "final_value": final_value,
        }

        # Attempt 1 data
        attempt1_value = attempt1_extracted.get(field_name)
        attempt1_ev = attempt1_evidence.get(field_name)
        critique1_entry = critique1_map.get(field_name, {})

        field_entry["attempt_1"] = {
            "value": attempt1_value,
            "evidence": attempt1_ev,
            "decision": None,
            "critic_reason": None,
        }

        # Find which bucket this field was in for attempt 1
        for bucket in ["passing", "failing", "low_confidence"]:
            for entry in attempt1_critique.get(bucket, []):
                if entry.get("field") == field_name:
                    field_entry["attempt_1"]["decision"] = bucket
                    field_entry["attempt_1"]["critic_reason"] = entry.get("critic_reason") or entry.get("reason")
                    # Include correction info if present
                    if entry.get("corrected_value") is not None:
                        field_entry["attempt_1"]["corrected_value"] = entry.get("corrected_value")
                        field_entry["attempt_1"]["correction_source"] = entry.get("correction_source", "critic_reasoning")
                    break

        # Attempt 2 data if exists
        if attempt2_extracted and field_name in attempt2_extracted:
            attempt2_value = attempt2_extracted.get(field_name)
            attempt2_ev = attempt2_evidence.get(field_name)

            field_entry["attempt_2"] = {
                "value": attempt2_value,
                "verbatim_source": attempt2_ev,
                "decision": None,
                "critic_reason": None,
            }

            # Find which bucket this field was in for attempt 2
            if attempt2_critique:
                for bucket in ["passing", "failing", "low_confidence"]:
                    for entry in attempt2_critique.get(bucket, []):
                        if entry.get("field") == field_name:
                            field_entry["attempt_2"]["decision"] = bucket
                            field_entry["attempt_2"]["critic_reason"] = entry.get("critic_reason") or entry.get("reason")
                            # Include correction info if present
                            if entry.get("corrected_value") is not None:
                                field_entry["attempt_2"]["corrected_value"] = entry.get("corrected_value")
                                field_entry["attempt_2"]["correction_source"] = entry.get("correction_source", "critic_reasoning")
                            break

        # Determine resolution status
        attempt1_decision = field_entry["attempt_1"].get("decision")
        attempt2_decision = field_entry.get("attempt_2", {}).get("decision")

        if attempt1_decision == "passing":
            field_entry["resolution"] = "pass_attempt_1"
        elif attempt2_decision == "passing":
            field_entry["resolution"] = "pass_attempt_2"
        elif attempt2_decision == "failing":
            field_entry["resolution"] = "human_review"
        elif attempt2_decision == "low_confidence":
            field_entry["resolution"] = "human_review"
        elif attempt1_decision == "failing" and attempt2_extracted is None:
            field_entry["resolution"] = "human_review"
        else:
            field_entry["resolution"] = "semantic_pass"

        evidence_chain["fields"][field_name] = field_entry

    return evidence_chain


def normalise_overview(
    extracted: dict,
    layout: dict | None = None,
    waiver_key: str | None = None
) -> dict:
    """
    Normalise extracted overview data according to type rules.
    Strips evidence keys from the main output.

    Uses a three-layer field resolution strategy:
    1. Direct match - field name exists in layout
    2. Fuzzy match - normalized field names match
    3. LLM resolution - semantic classification of unknown fields

    Args:
        extracted: Raw extracted dict from the overview agent (may include evidence keys)
        layout: Layout schema dict, or None to load default
        waiver_key: Optional waiver key for logging field resolutions

    Returns:
        Normalised dict with _normalisation_meta added, evidence keys stripped
    """
    if layout is None:
        layout = load_layout("layout_waiver_overview")

    normalised = {}
    norm_errors = []
    unknown_fields = []
    stripped_evidence_keys = []
    llm_resolved_fields = []  # Track fields resolved by LLM
    llm_stripped_fields = []  # Track metadata artifacts stripped by LLM

    for field_name, value in extracted.items():
        # Strip evidence and confidence keys - they go to evidence.json
        if _is_evidence_key(field_name):
            stripped_evidence_keys.append(field_name)
            continue

        # Pass through special fields unchanged
        if field_name in PASSTHROUGH_FIELDS or field_name.startswith("_"):
            normalised[field_name] = value
            continue

        # =================================================================
        # THREE-LAYER FIELD RESOLUTION
        # =================================================================

        field_name_canonical = None

        # Layer 0: Direct match (field exists in layout exactly)
        # No logging for direct matches - this is the expected case
        if field_name in layout:
            field_name_canonical = field_name

        # Layer 1: Fuzzy string match
        if field_name_canonical is None:
            fuzzy_match = _fuzzy_match_field(field_name, layout)
            if fuzzy_match:
                field_name_canonical = fuzzy_match
                if waiver_key:
                    log_event(waiver_key, {
                        "agent": "normalisation_agent",
                        "action": "field_resolution",
                        "resolution_type": "fuzzy_match",
                        "field_name": field_name,
                        "canonical_name": field_name_canonical,
                    })

        # Layer 2: LLM semantic resolution
        if field_name_canonical is None:
            llm_result = _resolve_unknown_field_via_llm(
                field_name, value, layout, waiver_key
            )

            classification = llm_result.get("classification")

            if classification == "schema_match":
                # LLM found a schema match
                field_name_canonical = llm_result.get("canonical_name")
                llm_resolved_fields.append({
                    "original": field_name,
                    "canonical": field_name_canonical,
                    "reason": llm_result.get("reason"),
                })

            elif classification == "metadata_artifact":
                # LLM says this is metadata - strip it silently
                llm_stripped_fields.append({
                    "field": field_name,
                    "reason": llm_result.get("reason"),
                })
                if waiver_key:
                    log_event(waiver_key, {
                        "agent": "normalisation_agent",
                        "action": "field_resolution",
                        "resolution_type": "llm_stripped",
                        "field_name": field_name,
                        "reason": llm_result.get("reason"),
                    })
                continue  # Skip this field entirely

            else:  # genuinely_unknown
                # Layer 3: Flag as genuinely unknown
                unknown_fields.append(field_name)
                normalised[field_name] = value
                if waiver_key:
                    log_event(waiver_key, {
                        "agent": "normalisation_agent",
                        "action": "field_resolution",
                        "resolution_type": "genuinely_unknown",
                        "field_name": field_name,
                        "reason": llm_result.get("reason"),
                    })
                continue

        # =================================================================
        # NORMALISATION (using canonical field name)
        # =================================================================

        # Determine field type and normalise
        field_type = _get_field_type(field_name_canonical, layout)

        if field_type == "date":
            norm_value, error = _normalise_date(value)
            normalised[field_name_canonical] = norm_value  # FIX: use canonical name
            if error:
                norm_errors.append({"field": field_name_canonical, "error": error})

        elif field_type == "boolean":
            normalised[field_name_canonical] = _normalise_boolean(value)

        elif field_type == "integer":
            normalised[field_name_canonical] = _normalise_integer(value)

        elif field_type == "float":
            normalised[field_name_canonical] = _normalise_float(value)

        elif field_type == "age":
            normalised[field_name_canonical] = _normalise_age(value)

        else:  # text
            normalised[field_name_canonical] = _normalise_text(value)

    # Add normalisation metadata
    normalised["_normalisation_meta"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "norm_errors_count": len(norm_errors),
        "unknown_fields_count": len(unknown_fields),
        "stripped_evidence_keys": len(stripped_evidence_keys),
        "llm_resolved_count": len(llm_resolved_fields),
        "llm_stripped_count": len(llm_stripped_fields),
        "norm_errors": norm_errors if norm_errors else None,
        "unknown_fields": unknown_fields if unknown_fields else None,
        "llm_resolved_fields": llm_resolved_fields if llm_resolved_fields else None,
        "llm_stripped_fields": llm_stripped_fields if llm_stripped_fields else None,
    }

    return normalised


def write_evidence_file(waiver_key: str, evidence_chain: dict) -> str:
    """
    Write the evidence chain to a separate JSON file.

    Args:
        waiver_key: The waiver key
        evidence_chain: The evidence chain dict

    Returns:
        Path to the written file
    """
    output_dir = get_output_dir(waiver_key)
    evidence_path = os.path.join(output_dir, f"{waiver_key}_evidence.json")

    with open(evidence_path, "w") as f:
        json.dump(evidence_chain, f, indent=2, default=str)

    return evidence_path


def run_normalisation(
    waiver_key: str,
    final_extracted: dict,
    attempt1_extracted: dict | None = None,
    attempt1_critique: dict | None = None,
    attempt2_extracted: dict | None = None,
    attempt2_critique: dict | None = None,
) -> tuple[dict, str | None]:
    """
    Run normalisation and build evidence chain.

    Args:
        waiver_key: Normalized waiver key
        final_extracted: Final merged extraction (to normalise)
        attempt1_extracted: Raw extraction from attempt 1 (for evidence chain)
        attempt1_critique: Critique from attempt 1 (for evidence chain)
        attempt2_extracted: Raw extraction from attempt 2 if exists (for evidence chain)
        attempt2_critique: Critique from attempt 2 if exists (for evidence chain)

    Returns:
        Tuple of (normalised dict, evidence_file_path or None)
    """
    layout = load_layout("layout_waiver_overview")
    normalised = normalise_overview(final_extracted, layout, waiver_key)

    meta = normalised.get("_normalisation_meta", {})

    # Build and write evidence chain if we have the data
    evidence_path = None
    if attempt1_extracted and attempt1_critique:
        evidence_chain = _build_evidence_chain(
            waiver_key,
            attempt1_extracted,
            attempt1_critique,
            attempt2_extracted,
            attempt2_critique,
            final_extracted,
        )
        evidence_path = write_evidence_file(waiver_key, evidence_chain)

    log_event(waiver_key, {
        "agent": "normalisation_agent",
        "action": "normalisation_complete",
        "norm_errors_count": meta.get("norm_errors_count", 0),
        "unknown_fields_count": meta.get("unknown_fields_count", 0),
        "stripped_evidence_keys": meta.get("stripped_evidence_keys", 0),
        "llm_resolved_count": meta.get("llm_resolved_count", 0),
        "llm_stripped_count": meta.get("llm_stripped_count", 0),
        "evidence_file": evidence_path,
        "unknown_fields": meta.get("unknown_fields"),
        "llm_resolved_fields": meta.get("llm_resolved_fields"),
        "llm_stripped_fields": meta.get("llm_stripped_fields"),
    })

    return normalised, evidence_path
