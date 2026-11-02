"""
Critic Agent for Overview Extraction Validation.

AcroForm-based verification oracle — no ground truth required.

Three-step verification:
1. AcroForm verification (deterministic) for mapped checkbox fields
2. Body text verification (LLM) for unmapped fields
3. Nested completeness check for active AcroForm sections

Output buckets:
- verified_acroform: AcroForm confirmed, high confidence
- verified_body: Evidence confirmed by LLM, good confidence
- corrected: AcroForm overrode LLM extraction, logged for audit
- soft_flag: Conflict needs re-extraction attempt
- hallucinated: Inactive parent chain, hard reject
- missing_child: Incomplete extraction, needs re-extraction
- human_queue: Failed all attempts, needs human review
"""

import os
import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

from ..shared_libraries.model_config import get_model_name, get_location

from ..parsers import (
    AcroFormTree,
    get_metadata,
    SCHEMA_TO_ACROFORM,
    INVERTED_FIELDS,
    BODY_TEXT_ONLY_FIELDS,
    get_acroform_path,
    is_inverted_field,
    is_body_text_only,
    has_acroform_mapping,
)
from ..shared_libraries.file_utils import load_layout, log_event


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

from ..shared_libraries.paths import HUMAN_OVERRIDES_PATH

# Fields to skip in verification
SKIP_FIELDS = {
    "Date of Completed Waiver Review",
    "_discovered_services",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS FOR BODY TEXT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

BODY_TEXT_CRITIC_SYSTEM = """You are verifying extracted field values against their evidence text.

Your task is to determine if the evidence text actually supports the extracted value.

For each field:
1. Read the evidence text carefully
2. Determine if it clearly supports the extracted value
3. Consider the field type and expected format
4. Flag any suspicious or unsupported extractions

Be strict: if the evidence is ambiguous or doesn't clearly support the value, mark it as unsupported."""


BODY_TEXT_CRITIC_USER = """Verify these extracted fields against their evidence.

## Layout Schema (for field context):
```json
{layout_schema}
```

## Fields to Verify:
```json
{fields_to_verify}
```

For each field, analyze the evidence and return JSON:

```json
{{
  "fields": [
    {{
      "field": "exact field name",
      "extracted": <the extracted value>,
      "evidence": "the evidence text",
      "page": <page number or null>,
      "evidence_supports": true | false,
      "confidence": "high" | "medium" | "low",
      "reason": "explanation of why evidence does/doesn't support the value"
    }}
  ]
}}
```

Rules:
- evidence_supports=true only if evidence CLEARLY shows the extracted value
- For checkbox fields (0/1), look for checked/unchecked indicators in evidence
- For text fields, evidence should contain or closely match the extracted value
- If evidence is missing or empty, evidence_supports=false
- Be conservative: when in doubt, mark as unsupported

Return ONLY valid JSON, no markdown fences."""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
        api_key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)


def _parse_json_response(raw_text: str) -> dict | None:
    """Parse JSON from model response, handling markdown fences."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])

    return json.loads(text)


def _normalize_value(value: Any) -> int | None:
    """Normalize a value to 0 or 1 for comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("1", "yes", "true", "on"):
            return 1
        elif val in ("0", "no", "false", "off", ""):
            return 0
    return None


def _flip_value(value: int | None) -> int | None:
    """Flip 0↔1 for inverted fields."""
    if value is None:
        return None
    return 1 - value


def _write_human_override(
    pdf_path: str,
    waiver_number: str,
    field_name: str,
    acroform_value: Any,
    llm_value: Any,
    conflict_type: str,
    attempt_reached: int,
    inverted_field: bool = False,
):
    """Write a record to human_overrides.jsonl for audit/review."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "pdf_path": str(pdf_path),
        "waiver_number": waiver_number or "",
        "field_name": field_name,
        "acroform_value": acroform_value,
        "llm_value": llm_value,
        "conflict_type": conflict_type,
        "attempt_reached": attempt_reached,
        "inverted_field": inverted_field,
        "human_decision": None,
        "final_value": None,
    }

    # Ensure directory exists
    HUMAN_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use file locking for parallel safety
    with open(HUMAN_OVERRIDES_PATH, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _call_body_text_critic(
    client,
    fields_to_verify: dict,
    layout_schema: dict,
) -> dict:
    """Call LLM to verify body text fields against their evidence."""
    user_prompt = BODY_TEXT_CRITIC_USER.format(
        layout_schema=json.dumps(layout_schema, indent=2),
        fields_to_verify=json.dumps(fields_to_verify, indent=2),
    )

    response = client.models.generate_content(
        model=get_model_name(),
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=BODY_TEXT_CRITIC_SYSTEM + "\n\n" + user_prompt)],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=65536,
        ),
    )

    return _parse_json_response(response.text)


# ═══════════════════════════════════════════════════════════════════════════════
# CRITIC AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CriticAgent:
    """
    AcroForm-based validation critic for extracted overview data.

    Uses AcroForm metadata as the verification oracle instead of ground truth.
    Falls back to LLM evidence verification for body-text-only fields.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize critic with PDF path.

        Args:
            pdf_path: Path to the PDF file being verified
        """
        self.pdf_path = Path(pdf_path)
        self.acroform_tree = AcroFormTree(pdf_path)
        self.metadata = get_metadata(pdf_path)
        self.field_mapping = SCHEMA_TO_ACROFORM
        self.inverted_fields = INVERTED_FIELDS
        self.body_text_only = BODY_TEXT_ONLY_FIELDS
        self.is_cms_form = self.metadata.get("is_cms_form", False)
        self.waiver_number = self.metadata.get("waiver_number", "")
        self.layout_schema = load_layout("layout_waiver_overview")

    def _verify_acroform_fields(
        self,
        extracted_data: dict,
        attempt: int,
    ) -> dict[str, dict]:
        """
        Verify fields that have AcroForm mappings.

        Returns dict mapping field_name to verification result:
        {
            "bucket": "verified_acroform" | "corrected" | "soft_flag" | "hallucinated",
            "acroform_value": <value from AcroForm>,
            "llm_value": <value from LLM extraction>,
            "conflict_type": <type if conflict>,
            "inverted_field": <true if field has inverted coding>,
        }
        """
        results = {}

        for field_name, extracted_value in extracted_data.items():
            # Skip metadata fields
            if field_name.endswith(("_evidence", "_page", "_confidence")):
                continue
            if field_name in SKIP_FIELDS or field_name.startswith("_"):
                continue

            # Check if field has AcroForm mapping
            acroform_path = get_acroform_path(field_name)
            if not acroform_path:
                continue  # Will be handled by body text verification

            # Get AcroForm value
            acroform_value = self.acroform_tree.get_value(acroform_path)

            # Normalize values for comparison
            acroform_norm = _normalize_value(acroform_value)
            llm_norm = _normalize_value(extracted_value)

            # Handle inverted fields: flip LLM value before comparing
            is_inverted = is_inverted_field(field_name)
            compare_value = _flip_value(llm_norm) if is_inverted else llm_norm

            # Check ancestor chain for hallucination detection
            if not self.acroform_tree.is_ancestor_chain_active(acroform_path):
                # Parent is inactive or field doesn't exist in AcroForm
                # Route to image OCR for visual verification instead of just flagging
                results[field_name] = {
                    "bucket": "soft_flag",  # Changed from "hallucinated" to enable reextraction
                    "acroform_value": acroform_value,
                    "llm_value": extracted_value,
                    "conflict_type": "inactive_parent",
                    "acroform_path": acroform_path,
                    "ancestor_chain": self.acroform_tree.get_ancestor_chain(acroform_path),
                    "inverted_field": is_inverted,
                    "needs_image_ocr": True,  # Route to image OCR path for visual verification
                    "reason": "Ancestor chain inactive or field missing in AcroForm - need visual verification",
                }
                continue

            # Check for parent-child inconsistency in AcroForm
            # If parent is OFF but this child field is ON, AcroForm data is unreliable
            # Send to image OCR for visual verification
            parent_child_inconsistent = False
            if acroform_path and ":" in acroform_path:
                parts = acroform_path.split(":")
                if len(parts) >= 2:
                    # Check common parent patterns (e.g., locHospType -> locHosp)
                    parent_patterns = [
                        (acroform_path.replace("Type", ""), acroform_path.endswith("Type")),
                        (":".join(parts[:-1]), True),  # Direct parent segment
                    ]
                    for parent_path, is_valid_pattern in parent_patterns:
                        if is_valid_pattern and parent_path != acroform_path:
                            parent_value = self.acroform_tree.get_value(parent_path)
                            if parent_value is False and acroform_value is True:
                                # Parent OFF but child ON — inconsistent AcroForm
                                parent_child_inconsistent = True
                                break

            if parent_child_inconsistent:
                # Don't trust AcroForm correction — send to image OCR
                results[field_name] = {
                    "bucket": "soft_flag",
                    "acroform_value": acroform_value,
                    "llm_value": extracted_value,
                    "conflict_type": "parent_child_inconsistent",
                    "acroform_path": acroform_path,
                    "inverted_field": is_inverted,
                    "needs_image_ocr": True,  # Route to image OCR path
                    "reason": "Parent AcroForm field is OFF but child is ON - need visual verification",
                }
                continue

            # Compare values
            if acroform_norm is None:
                # AcroForm has no value for this field
                if llm_norm is not None:
                    # LLM extracted something but AcroForm is empty — soft flag
                    results[field_name] = {
                        "bucket": "soft_flag",
                        "acroform_value": acroform_value,
                        "llm_value": extracted_value,
                        "conflict_type": "acroform_empty",
                        "acroform_path": acroform_path,
                        "inverted_field": is_inverted,
                    }
                # else: both None, nothing to verify
                continue

            if compare_value == acroform_norm:
                # Match — verified
                results[field_name] = {
                    "bucket": "verified_acroform",
                    "acroform_value": acroform_value,
                    "llm_value": extracted_value,
                    "acroform_path": acroform_path,
                    "inverted_field": is_inverted,
                }
            else:
                # Mismatch — AcroForm wins, this is a correction
                # Determine the correct output value (flip back if inverted)
                corrected_value = _flip_value(acroform_norm) if is_inverted else acroform_norm

                results[field_name] = {
                    "bucket": "corrected",
                    "acroform_value": acroform_value,
                    "llm_value": extracted_value,
                    "corrected_value": corrected_value,
                    "conflict_type": "wrong_value",
                    "acroform_path": acroform_path,
                    "inverted_field": is_inverted,
                }

                # Write to human_overrides.jsonl for audit
                _write_human_override(
                    pdf_path=str(self.pdf_path),
                    waiver_number=self.waiver_number,
                    field_name=field_name,
                    acroform_value=acroform_value,
                    llm_value=extracted_value,
                    conflict_type="wrong_value",
                    attempt_reached=attempt,
                    inverted_field=is_inverted,
                )

        return results

    def _verify_body_text_fields(
        self,
        extracted_data: dict,
        acroform_results: dict,
    ) -> dict[str, dict]:
        """
        Verify fields that don't have AcroForm mappings using LLM.

        These are typically body text fields where we verify the evidence
        actually supports the extracted value.
        """
        # Collect fields without AcroForm mapping (not already verified)
        fields_to_verify = {}

        for field_name, value in extracted_data.items():
            # Skip metadata fields
            if field_name.endswith(("_evidence", "_page", "_confidence")):
                continue
            if field_name in SKIP_FIELDS or field_name.startswith("_"):
                continue

            # Skip if already handled by AcroForm verification
            if field_name in acroform_results:
                continue

            # Skip if has AcroForm mapping (shouldn't happen, but defensive)
            if has_acroform_mapping(field_name):
                continue

            # Get evidence and page
            evidence = extracted_data.get(f"{field_name}_evidence")
            page = extracted_data.get(f"{field_name}_page")

            fields_to_verify[field_name] = {
                "value": value,
                "evidence": evidence,
                "page": page,
            }

        if not fields_to_verify:
            return {}

        # Call LLM to verify evidence
        try:
            client = _get_client()
            llm_result = _call_body_text_critic(
                client, fields_to_verify, self.layout_schema
            )
        except Exception as e:
            # On LLM failure, mark all as soft_flag
            results = {}
            for field_name in fields_to_verify:
                results[field_name] = {
                    "bucket": "soft_flag",
                    "reason": f"LLM verification failed: {e}",
                }
            return results

        # Parse LLM results
        results = {}
        for field_result in llm_result.get("fields", []):
            field_name = field_result.get("field")
            if not field_name:
                continue

            if field_result.get("evidence_supports"):
                results[field_name] = {
                    "bucket": "verified_body",
                    "confidence": field_result.get("confidence", "medium"),
                    "reason": field_result.get("reason"),
                }
            else:
                results[field_name] = {
                    "bucket": "soft_flag",
                    "confidence": field_result.get("confidence", "low"),
                    "reason": field_result.get("reason"),
                }

        return results

    def _check_nested_completeness(
        self,
        extracted_data: dict,
        acroform_results: dict,
    ) -> list[dict]:
        """
        Check if all active AcroForm descendants were extracted.

        Returns list of missing child fields.
        """
        missing = []

        # Get all extracted field names (excluding metadata)
        extracted_fields = {
            k for k in extracted_data.keys()
            if not k.endswith(("_evidence", "_page", "_confidence"))
            and not k.startswith("_")
        }

        # For each verified AcroForm field, check its siblings
        for field_name, result in acroform_results.items():
            if result["bucket"] != "verified_acroform":
                continue

            acroform_path = result.get("acroform_path")
            if not acroform_path:
                continue

            # Get parent path
            parts = acroform_path.split(":")
            if len(parts) < 2:
                continue
            parent_path = ":".join(parts[:-1])

            # Get all active descendants of parent
            try:
                active_descendants = self.acroform_tree.get_active_descendants(parent_path)
            except Exception:
                continue

            # Check which active descendants are mapped to schema fields
            for descendant_path in active_descendants:
                # Find schema field for this descendant
                for schema_field, af_path in self.field_mapping.items():
                    if af_path == descendant_path and schema_field not in extracted_fields:
                        missing.append({
                            "field": schema_field,
                            "acroform_path": descendant_path,
                            "parent_path": parent_path,
                            "conflict_type": "missing_child",
                        })

        return missing

    def critique(
        self,
        extracted_data: dict,
        attempt: int = 1,
    ) -> dict:
        """
        Verify extracted data using AcroForm as oracle.

        Args:
            extracted_data: The extracted overview dict (with _evidence, _page keys)
            attempt: Current attempt number (1, 2, or 3)

        Returns:
            Critique dict with buckets and decision
        """
        # Handle non-CMS forms
        if not self.is_cms_form:
            return {
                "verified_acroform": [],
                "verified_body": [],
                "corrected": [],
                "soft_flag": [],
                "hallucinated": [],
                "missing_child": [],
                "human_queue": [],
                "accuracy": 0.0,
                "decision": "human",
                "note": "Non-CMS form — requires human review",
            }

        # Handle PDFs without AcroForm
        if not self.acroform_tree.has_acroform:
            return {
                "verified_acroform": [],
                "verified_body": [],
                "corrected": [],
                "soft_flag": [],
                "hallucinated": [],
                "missing_child": [],
                "human_queue": [],
                "accuracy": 0.0,
                "decision": "human",
                "note": "PDF has no AcroForm — requires human review",
            }

        # Step 1: AcroForm verification (deterministic)
        acroform_results = self._verify_acroform_fields(extracted_data, attempt)

        # Step 2: Body text verification (LLM)
        body_results = self._verify_body_text_fields(extracted_data, acroform_results)

        # Step 3: Nested completeness check
        missing_children = self._check_nested_completeness(extracted_data, acroform_results)

        # Build output buckets
        verified_acroform = []
        verified_body = []
        corrected = []
        soft_flag = []
        hallucinated = []

        for field_name, result in acroform_results.items():
            entry = {"field": field_name, **result}
            bucket = result["bucket"]
            if bucket == "verified_acroform":
                verified_acroform.append(entry)
            elif bucket == "corrected":
                corrected.append(entry)
            elif bucket == "soft_flag":
                soft_flag.append(entry)
            elif bucket == "hallucinated":
                hallucinated.append(entry)

        for field_name, result in body_results.items():
            entry = {"field": field_name, **result}
            bucket = result["bucket"]
            if bucket == "verified_body":
                verified_body.append(entry)
            elif bucket == "soft_flag":
                soft_flag.append(entry)

        missing_child = [{"field": m["field"], **m} for m in missing_children]

        # Determine if fields should go to human queue (after attempt 3)
        human_queue = []
        if attempt >= 3:
            # Move remaining soft_flag and missing_child to human_queue
            for item in soft_flag + missing_child:
                human_queue.append(item)
                # Write to human_overrides.jsonl
                _write_human_override(
                    pdf_path=str(self.pdf_path),
                    waiver_number=self.waiver_number,
                    field_name=item["field"],
                    acroform_value=item.get("acroform_value"),
                    llm_value=item.get("llm_value") or extracted_data.get(item["field"]),
                    conflict_type=item.get("conflict_type", "unresolved"),
                    attempt_reached=attempt,
                    inverted_field=item.get("inverted_field", False),
                )
            soft_flag = []
            missing_child = []

        # Calculate accuracy
        total_verified = len(verified_acroform) + len(verified_body) + len(corrected)
        total_fields = (
            total_verified
            + len(soft_flag)
            + len(hallucinated)
            + len(missing_child)
            + len(human_queue)
        )
        accuracy = total_verified / total_fields if total_fields > 0 else 1.0

        # Determine decision
        has_issues = (
            len(soft_flag) > 0
            or len(missing_child) > 0
            or len(hallucinated) > 0
        )

        if not has_issues and len(human_queue) == 0:
            decision = "pass"
        elif attempt == 1:
            decision = "attempt2"
        elif attempt == 2:
            decision = "attempt3"
        else:
            decision = "human"

        return {
            "verified_acroform": verified_acroform,
            "verified_body": verified_body,
            "corrected": corrected,
            "soft_flag": soft_flag,
            "hallucinated": hallucinated,
            "missing_child": missing_child,
            "human_queue": human_queue,
            "accuracy": accuracy,
            "decision": decision,
            "total_fields": total_fields,
            "attempt": attempt,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def run_critic(
    pdf_path: str,
    extracted_data: dict,
    attempt: int,
    waiver_key: str | None = None,
) -> dict:
    """
    Run the critic agent and log results.

    Args:
        pdf_path: Path to the PDF file
        extracted_data: Extracted overview data (with evidence keys)
        attempt: Current attempt number (1, 2, or 3)
        waiver_key: Optional waiver key for logging

    Returns:
        Critique dict with verification buckets and decision
    """
    try:
        critic = CriticAgent(pdf_path)
        critique = critic.critique(extracted_data, attempt=attempt)

        # Log the critique
        if waiver_key:
            log_event(waiver_key, {
                "agent": "critic_agent",
                "action": "critique_complete",
                "attempt": attempt,
                "accuracy": critique["accuracy"],
                "decision": critique["decision"],
                "verified_acroform_count": len(critique["verified_acroform"]),
                "verified_body_count": len(critique["verified_body"]),
                "corrected_count": len(critique["corrected"]),
                "soft_flag_count": len(critique["soft_flag"]),
                "hallucinated_count": len(critique["hallucinated"]),
                "missing_child_count": len(critique["missing_child"]),
                "human_queue_count": len(critique["human_queue"]),
                "verified_acroform_fields": [f["field"] for f in critique["verified_acroform"]],
                "corrected_fields": [
                    {"field": f["field"], "corrected_value": f.get("corrected_value")}
                    for f in critique["corrected"]
                ],
                "soft_flag_fields": [f["field"] for f in critique["soft_flag"]],
                "hallucinated_fields": [f["field"] for f in critique["hallucinated"]],
            })

        return critique

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        # Log the error with full details
        if waiver_key:
            log_event(waiver_key, {
                "agent": "critic_agent",
                "action": "critique_error",
                "attempt": attempt,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_traceback,
            })

        # Return a safe error response that allows pipeline to continue
        return {
            "verified_acroform": [],
            "verified_body": [],
            "corrected": [],
            "soft_flag": [],
            "hallucinated": [],
            "missing_child": [],
            "human_queue": [],
            "accuracy": 0.0,
            "decision": "human",
            "error": str(e),
            "error_type": type(e).__name__,
            "error_traceback": error_traceback,
        }


def get_page_window(extracted_data: dict, field_name: str) -> list[int]:
    """
    Get [page-1, page, page+1] for targeted re-extraction.

    Args:
        extracted_data: Extracted data dict with _page keys
        field_name: Field to get page window for

    Returns:
        List of page numbers [prev, current, next] or empty if no page info
    """
    page = extracted_data.get(f"{field_name}_page")
    if page is None:
        return []

    try:
        # Convert to int (LLM may return strings like "5" instead of 5)
        page = int(page)
        return [max(1, page - 1), page, page + 1]
    except (ValueError, TypeError) as e:
        # Log warning but don't crash - return empty list
        import logging
        logging.warning(
            f"get_page_window: Could not convert page value '{page}' "
            f"(type: {type(page).__name__}) to int for field '{field_name}': {e}"
        )
        return []
