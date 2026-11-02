"""
Custom service-specific values for the services ADK app.

The user names a value they want that is not one of the standard service
columns (for example "provider qualifications for Respite", or "the unit of
service for Day Habilitation Level 2"). One model call finds it in the PDF and
returns the value with its page and verbatim evidence; the user approves,
changes, retries (with a one-off hint) or rejects it. Only an approved value is
saved: to data/outputs/<KEY>/custom_service_extractions.json and as one row in
data/outputs/custom_service_extractions.csv (key: waiver + service + label).
It never touches the standard services CSV.
"""

import datetime
import json

from waiver_level_extraction_agent.adk_agents.tools.state_pipeline_tools import (
    _adk_client, _retry_transient, _resolve_pdf, _state,
)

try:
    from google.adk.tools import ToolContext
except ImportError:  # pragma: no cover
    ToolContext = None

_PROMPT = """You are extracting ONE specific value about a service from a 1915(c) HCBS Medicaid waiver application PDF.

## What to extract
- Service: {service}
- Name of the value: {label}
- Meaning: {description}
- What it usually contains: {usual_contents}
- Value type: {value_type}
- Where to look (hint only - search the whole document if it is not there): {where_to_look}
- Allowed values: {allowed_values}
{hint_block}
## Rules
1. Read the whole document. The location hint may be wrong or missing.
2. If a Service is named above, extract the value FOR THAT SERVICE ONLY. If the service has several variants or child services and the value differs between them, give the parent service's value in "value" and list the others in "other_occurrences" with the variant name in "value" text. If no service is named, the value is about the waiver's services in general.
3. Return the value exactly as the document states it. Do not paraphrase or summarise text values. For yes_no return 1 for yes/checked and 0 for no/unchecked. For number return only the number (no units or commas). For date use YYYY-MM-DD when possible. For list return a JSON array of strings.
4. If the value is not in the document, return found=false and value=null. Never guess.
5. "evidence" must be the VERBATIM text you used (for checkboxes include CHECKED / UNCHECKED and the label).
6. "page" is the 1-indexed PDF page of the evidence.
7. If the same item appears in several places with different values, give the best answer in "value" and list the others in "other_occurrences" with their pages.

Return ONLY a JSON object:
{{"found": true/false, "value": ..., "page": <int or null>, "evidence": "<verbatim>",
  "confidence": "high|medium|low", "notes": "<anything the reviewer should know>",
  "other_occurrences": [{{"value": ..., "page": <int>, "evidence": "<verbatim>"}}]}}
"""


def _pending_key(service_name: str, label: str) -> str:
    return f"{service_name or ''}||{label}"


def extract_service_custom_value(
    waiver_key: str,
    label: str,
    description: str,
    service_name: str = "",
    value_type: str = "text",
    usual_contents: str = "",
    where_to_look: str = "",
    allowed_values: str = "",
    pdf_path: str = "",
    hint: str = "",
    tool_context: "ToolContext" = None,
) -> dict:
    """Extract ONE user-defined value about a service that is not a standard
    services column. ONLY call this after the user has confirmed the definition
    (name, meaning, what it usually contains, type, which service). One model
    call; nothing is saved - the user must approve the result first (see
    save_service_custom_value).

    Args:
        waiver_key: The waiver identifier
        label: Short name for the value, e.g. "Provider qualifications"
        description: What the value means, in the user's words
        service_name: The service it is about (a parent or child service name
            exactly as shown in the review table), or "" for a value about the
            waiver's services in general
        value_type: "text", "number", "yes_no" (1/0), "date" or "list"
        usual_contents: What it normally looks like / example values
        where_to_look: Appendix/section/page hint, or "" if unknown
        allowed_values: Fixed list of allowed answers, or "" for free
        pdf_path: Optional PDF in any folder (default: found by waiver_key,
            or the PDF already loaded in this session)
        hint: One-off guidance for a retry (e.g. "look in Appendix C-1/C-3").
            Used for this call only; it is NOT part of the saved definition, so
            never put it in description / where_to_look.
    """
    from google.genai import types
    from waiver_level_extraction_agent.core.pipeline import _load_pdf_bytes, _parse_json_response, _retry_on_connection_error
    from waiver_level_extraction_agent.shared_libraries.model_config import get_model_name
    from waiver_level_extraction_agent.shared_libraries.file_utils import log_event

    st = _state(tool_context)
    pdf = pdf_path or st.get("pdf_path") or ""
    found = _resolve_pdf(waiver_key, pdf)
    if found.get("status") != "found":
        return {"status": "error", "error": found.get("error", "PDF not found")}
    pdf = found["pdf_path"]
    if not label or not description:
        return {"status": "error", "error": "label and description are required - confirm them with the user first"}

    prompt = _PROMPT.format(
        service=service_name or "(none - about the waiver's services in general)",
        label=label, description=description, value_type=value_type,
        usual_contents=usual_contents or "(not specified)",
        where_to_look=where_to_look or "(unknown)",
        allowed_values=allowed_values or "(free)",
        hint_block=f"- Extra guidance for this attempt: {hint}\n" if hint else "",
    )
    model = get_model_name()
    log_event(waiver_key, {"agent": "service_custom_value", "action": "extraction_start",
                           "label": label, "service": service_name, "model": model, "hint": hint})
    try:
        client = _adk_client(model)
        pdf_bytes = _load_pdf_bytes(pdf)

        def _request():
            r = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_bytes)),
                    types.Part(text=prompt),
                ])],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16384),
            )
            return r.text.strip()

        result = _parse_json_response(_retry_transient(lambda: _retry_on_connection_error(_request)))
    except Exception as e:  # noqa: BLE001
        log_event(waiver_key, {"agent": "service_custom_value", "action": "model_error",
                               "label": label, "service": service_name, "error": str(e)})
        return {"status": "error", "error": str(e)}

    result = {**result, "label": label, "service_name": service_name, "description": description,
              "value_type": value_type, "usual_contents": usual_contents,
              "where_to_look": where_to_look, "model": model, "pdf_path": pdf}
    log_event(waiver_key, {"agent": "service_custom_value", "action": "extraction_complete",
                           "label": label, "service": service_name, "found": result.get("found"),
                           "confidence": result.get("confidence"), "page": result.get("page")})
    if tool_context is not None:
        pending = dict(st.get("custom_service_pending") or {})
        pending[_pending_key(service_name, label)] = result
        st["custom_service_pending"] = pending
        st["pdf_path"] = pdf
    return {"status": "success", "waiver_key": waiver_key, **result}


def save_service_custom_value(
    waiver_key: str,
    label: str,
    service_name: str = "",
    value: str = "",
    output_csv: str = "",
    dry_run: bool = False,
    tool_context: "ToolContext" = None,
) -> dict:
    """Save a user-approved custom service value: to the waiver's
    custom_service_extractions.json and as one row (waiver + service + label) in
    the long-format CSV (default data/outputs/custom_service_extractions.csv).
    Call only after the user approved the value shown by
    extract_service_custom_value.

    Args:
        waiver_key: The waiver identifier
        label: The label used in extract_service_custom_value
        service_name: The same service_name used in extract_service_custom_value ("" if none)
        value: Optional corrected value the user wants saved instead of the
            extracted one (leave "" to save the extracted value as is)
        output_csv: Optional different CSV (file name -> data/outputs/, or a path)
        dry_run: Report what would be written without writing
    """
    from waiver_level_extraction_agent.shared_libraries.paths import OUTPUTS_DIR
    from waiver_level_extraction_agent.shared_libraries.file_utils import log_event
    from waiver_level_extraction_agent.storage.csv_merge import merge_custom_service_value

    st = _state(tool_context)
    res = (st.get("custom_service_pending") or {}).get(_pending_key(service_name, label))
    if not res:
        return {"status": "error", "error": f"No extracted value for '{label}' (service '{service_name}') in this "
                                            "session - run extract_service_custom_value first"}
    final = value if value != "" else res.get("value")
    if isinstance(final, (list, dict)):
        final = json.dumps(final, ensure_ascii=False)
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "waiver_key": waiver_key, "service_name": service_name, "label": label, "value": final,
        "page": res.get("page"), "evidence": res.get("evidence"), "confidence": res.get("confidence"),
        "description": res.get("description"), "value_type": res.get("value_type"),
        "model": res.get("model"), "extracted_at": now, "approved_at": now,
    }
    csv_result = merge_custom_service_value(row, output_csv=output_csv, dry_run=dry_run)
    if dry_run or csv_result.get("status") == "error":
        return {"status": csv_result["status"], "row": row, "csv": csv_result}

    jp = OUTPUTS_DIR / waiver_key / "custom_service_extractions.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(jp.read_text()) if jp.exists() else {}
    data[_pending_key(service_name, label)] = {
        **row, "found": res.get("found"), "notes": res.get("notes"),
        "other_occurrences": res.get("other_occurrences"),
        "usual_contents": res.get("usual_contents"), "where_to_look": res.get("where_to_look")}
    jp.write_text(json.dumps(data, indent=2, default=str))
    log_event(waiver_key, {"agent": "service_custom_value", "action": "saved",
                           "label": label, "service": service_name})
    return {"status": "success", "saved_value": final, "json_path": str(jp), "csv": csv_result}
