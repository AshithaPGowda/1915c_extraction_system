"""
Re-extraction agent prompt template.

Focused prompt for re-extracting only failing and low-confidence fields.
Receives: full PDF inline, list of fields with rejection reasons and location hints.
Returns: JSON with only the re-extracted fields, each with verbatim evidence and page numbers.

Enhanced with AcroForm context for improved accuracy.
"""

from typing import Any


REEXTRACTION_PROMPT_TEMPLATE = """You are a precision document extraction specialist.

Your previous extraction attempt had errors. You must now re-extract ONLY the specific fields listed below.

## TARGET FIELDS FOR RE-EXTRACTION

{field_details}

## CRITICAL EXTRACTION RULES

### Checkbox Fields
Read checkboxes VISUALLY from the PDF:
- ☒ or [X] or filled box = checked = 1 (Yes/True)
- ☐ or [ ] or empty box = unchecked = 0 (No/False)
NEVER infer checkbox values from surrounding text alone.

### Radio Button Groups — READ ALL OPTIONS
For radio button groups (○/◉ or O/● symbols):
- ALWAYS read ALL options in the group and their filled/unfilled state
- NEVER stop at the first option you encounter
- The selected option is the one with a FILLED circle (◉ or ●)
- An empty circle (○ or O) means that option is NOT selected
- Example: "○ No  ● Yes" → the answer is Yes (filled circle)

### Nested Checkbox Groups — READ ALL SUB-ITEMS
When a radio button or checkbox reveals sub-checkboxes underneath:
- Pattern: parent selected → "check each that applies" → list of sub-checkboxes
- You MUST read EVERY sub-checkbox state individually
- Never stop after finding the first checked sub-item
- Each sub-checkbox is a separate field
- Example: ◉ Applicable → ☒ option A, ☐ option B, ☒ option C
  → Extract ALL THREE states separately

### Miller Trust State (B-4a) — HIGH ATTENTION CHECKBOX
⚠️ THIS IS A CHECKBOX FIELD REQUIRING CAREFUL VISUAL INSPECTION.

This field has INVERTED coding (opposite of typical 0=No, 1=Yes):
- 0 = Yes (the state IS a Miller Trust state)
- 1 = No (the state is NOT a Miller Trust state)

You must VISUALLY locate the Miller Trust checkbox in Appendix B-4a:
- If "Yes" is CHECKED (☒ or ●): return 0
- If "No" is CHECKED (☒ or ●): return 1
- Look for labels: "Miller Trust", "Qualified Income Trust", "QIT"

DO NOT infer from context. You MUST see the actual checkbox state.

### Type of Waiver (1.D) — NOT Type of Request (1.C)
1.D Type of Waiver is DIFFERENT from 1.C Type of Request:
- Look specifically for section labeled "D. Type of Waiver"
- Options: Regular Waiver=1, Model Waiver=3
- Do NOT confuse with "C. Type of Request" (new/renewal/amendment)

### Coded Fields
For any field with format "(0=No, 1=Yes)" or similar:
- Return the INTEGER code, not a string
- Read the actual checkbox or selection in the document

### AcroForm Verification
Some fields below include **AcroForm authoritative values** from the PDF form metadata.
- AcroForm values are AUTHORITATIVE for checkbox fields — the form metadata knows the true state
- If your visual extraction disagrees with AcroForm, re-read the checkbox more carefully
- For body-text-only fields (no AcroForm), rely purely on evidence text

## EVIDENCE AND PAGE REQUIREMENT — MANDATORY FORMAT

For EVERY field you return, you MUST ALSO return:
1. A parallel `_evidence` key with the verbatim text/checkbox state
2. A parallel `_page` key with the 1-indexed page number where you found this field

**For checkbox and radio button fields, use this EXACT evidence format:**
"[CHECKED ☒ / UNCHECKED ☐] — label text — interpreted as value X"

Examples:
- "UNCHECKED ☐ — As a Medicaid state plan service under §1915(g)(1) — interpreted as 0"
- "CHECKED ☒ — Not applicable — interpreted as 1"
- "CHECKED ● — Yes (Miller Trust state) — interpreted as 0 (inverted coding)"
- "UNCHECKED ○ — No — not selected"

This format makes the checkbox/radio state UNAMBIGUOUS.

Example output format:
```json
{{
  "B-4a. Miller Trust State. (0=Yes, 1=No)": 0,
  "B-4a. Miller Trust State. (0=Yes, 1=No)_evidence": "CHECKED ☒ — Yes (state uses Miller Trusts) — interpreted as 0 per inverted coding",
  "B-4a. Miller Trust State. (0=Yes, 1=No)_page": 15,
  "1.D. Type of Waiver (1=regular waiver, 2= amendment; 3 Model Waiver)": 1,
  "1.D. Type of Waiver (1=regular waiver, 2= amendment; 3 Model Waiver)_evidence": "Section D. Type of Waiver: CHECKED ☒ — Regular Waiver — interpreted as 1; UNCHECKED ☐ — Model Waiver",
  "1.D. Type of Waiver (1=regular waiver, 2= amendment; 3 Model Waiver)_page": 1,
  "1.G. Concurrent Operation with Other Programs: Not applicable (1 = Yes, 0=not applicable)": 0,
  "1.G. Concurrent Operation with Other Programs: Not applicable (1 = Yes, 0=not applicable)_evidence": "UNCHECKED ☐ — Not applicable — interpreted as 0",
  "1.G. Concurrent Operation with Other Programs: Not applicable (1 = Yes, 0=not applicable)_page": 3
}}
```

## INSTRUCTIONS

1. For EACH field listed above:
   - Locate the field in the PDF using the section/location hints provided
   - If a page window is provided, START your search on those pages
   - Read the value EXACTLY as it appears in the document
   - For radio buttons: read ALL options before determining which is filled
   - For nested checkboxes: read ALL sub-items
   - Apply the correct coding (pay attention to inverted coding like Miller Trust)
   - If AcroForm value is provided, ensure your extraction reconciles with it
   - Format the evidence with explicit CHECKED/UNCHECKED state
   - Record the page number where you found the field

2. Return ONLY the fields listed above, each with its evidence AND page key:
   ```json
   {{
     "field_name_1": <extracted_value>,
     "field_name_1_evidence": "<CHECKED/UNCHECKED format>",
     "field_name_1_page": <page_number>,
     "field_name_2": <extracted_value>,
     "field_name_2_evidence": "<CHECKED/UNCHECKED format>",
     "field_name_2_page": <page_number>,
     ...
   }}
   ```

3. Additional extraction rules:
   - For binary/checkbox fields: Return 0 or 1 as integers, not strings
   - For coded fields: Return the numeric code, not the text description
   - For dates: Return in YYYY-MM-DD format
   - For text fields: Extract EXACTLY as written, no paraphrasing
   - For "No Max" / unlimited age fields: Return "No Max" or null
   - If a field truly cannot be found, return null (but still provide evidence of what you searched)

4. DO NOT:
   - Include any fields not listed above
   - Add explanatory text or markdown around the JSON
   - Paraphrase or summarize text fields
   - Guess values — if unsure, return null with evidence of why
   - Omit the evidence key or page key for any field
   - Stop reading radio buttons after the first option
   - Skip sub-checkboxes in nested groups

Return ONLY valid JSON with the re-extracted field values, their evidence, AND their page numbers."""


def _format_acroform_subtree(subtree: dict, indent: int = 0) -> str:
    """
    Format an AcroForm subtree dict as a readable indented tree.

    Args:
        subtree: The subtree dict from AcroFormTree.get_subtree()
        indent: Current indentation level

    Returns:
        Formatted string showing field hierarchy
    """
    if not subtree:
        return ""

    lines = []
    prefix = "  " * indent

    # Handle different subtree formats
    children = subtree.get("children", {})
    if not children and isinstance(subtree, dict):
        # Might be a flat dict of field_name -> value
        children = {k: v for k, v in subtree.items() if k != "value" and k != "children"}

    for i, (name, child) in enumerate(children.items()):
        is_last = i == len(children) - 1
        connector = "└─" if is_last else "├─"

        # Get value
        if isinstance(child, dict):
            value = child.get("value")
        else:
            value = child

        # Format the line
        value_str = str(value) if value is not None else "None"
        lines.append(f"{prefix}{connector} {name} = {value_str}")

        # Recurse into nested children
        if isinstance(child, dict) and child.get("children"):
            nested = _format_acroform_subtree(child, indent + 1)
            if nested:
                lines.append(nested)

    return "\n".join(lines)


def format_field_for_reextraction(
    field_name: str,
    previous_value: Any,
    rejection_reason: str,
    layout_info: dict | None,
    acroform_value: bool | str | None = None,
    acroform_subtree: dict | None = None,
    page_window: list[int] | None = None,
    conflict_type: str | None = None,
    parent_path: str | None = None,
) -> str:
    """
    Format a single field's re-extraction instructions.

    Args:
        field_name: The field name
        previous_value: The value from the previous extraction attempt
        rejection_reason: The critic's rejection reason
        layout_info: Layout schema info for this field (section, context)
        acroform_value: Authoritative value from AcroForm metadata (optional)
        acroform_subtree: Form structure context showing siblings (optional)
        page_window: List of page numbers to search [prev, current, next] (optional)
        conflict_type: Type of conflict (wrong_value, soft_flag, missing_child, hallucinated)
        parent_path: Parent section path for missing_child fields (optional)

    Returns:
        Formatted string with field details
    """
    section = "Unknown Section"
    context = "Extract the value as it appears in the document."
    is_in_table = False

    if layout_info:
        section = layout_info.get("Header", "Unknown Section")
        context = layout_info.get("context", context)
        is_in_table = layout_info.get("isInTable", False)

    location_hint = f"Look in: {section}"
    if is_in_table:
        location_hint += " (appears in a table or checkbox grid)"

    # Add special hints for known problematic fields (existing logic)
    special_hint = ""
    if "Miller Trust" in field_name:
        special_hint = "\n- **⚠️ HIGH ATTENTION CHECKBOX**: This is a CHECKBOX field with INVERTED coding: 0=Yes (IS a Miller Trust state), 1=No (is NOT). You MUST visually locate and read the checkbox state in Appendix B-4a. Do NOT infer from text."
    elif "Type of Waiver" in field_name and "1.D" in field_name:
        special_hint = "\n- **SPECIAL**: Look for section 'D. Type of Waiver' specifically. Do NOT use 'C. Type of Request'. Checkboxes: ☒ Regular Waiver=1, ☒ Model Waiver=3"
    elif "Not applicable" in field_name or "Concurrent Operation" in field_name:
        special_hint = "\n- **SPECIAL**: Read ALL radio/checkbox options. If 'Not applicable' is UNCHECKED, return 0. If CHECKED, return 1."
    elif layout_info and ("check each that applies" in context.lower() or "check all" in context.lower()):
        special_hint = "\n- **SPECIAL**: This has nested checkboxes. Read EVERY sub-checkbox state, not just the first one."

    # Build base field block
    lines = [
        f"### {field_name}",
        f"- **Previous value**: {previous_value}",
        f"- **Rejection reason**: {rejection_reason}",
        f"- **Location**: {location_hint}",
        f"- **Extraction guidance**: {context}{special_hint}",
    ]

    # Add AcroForm verification section if we have AcroForm info
    if acroform_value is not None or conflict_type:
        lines.append("")
        lines.append("#### AcroForm Verification")

        if acroform_value is not None:
            lines.append(f"- **⚠️ AcroForm authoritative value**: {acroform_value}")
            lines.append("- **This is what the PDF form metadata says. Your extraction must reconcile with this.**")
        else:
            lines.append("- **⚠️ AcroForm authoritative value**: None (no form metadata for this field)")

        # Add conflict-specific guidance
        if conflict_type:
            lines.append(f"- **Conflict type**: {conflict_type}")

            if conflict_type == "wrong_value":
                lines.append("- **Guidance**: AcroForm and your previous extraction disagree. AcroForm value is authoritative for checkboxes. Re-read the checkbox carefully.")

            elif conflict_type == "soft_flag":
                if acroform_value is None:
                    # Body-text-only field
                    lines.append("- **Guidance**: This field has no AcroForm backing. Verify purely from body text evidence. Your previous extraction may be correct — re-read the evidence on the page window carefully.")
                else:
                    # Genuine conflict with AcroForm
                    lines.append("- **Guidance**: AcroForm shows /Off but evidence suggests a value. Check if this field has body text that overrides the checkbox state. Evidence takes priority here.")

            elif conflict_type == "missing_child":
                lines.append("- **Guidance**: This field was not extracted but its parent checkbox is active. It must have a value. Look for nested sub-items under the parent section.")

            elif conflict_type == "hallucinated":
                lines.append("- **Guidance**: This field's parent checkbox is inactive (/Off). The field should be empty/null. Do not extract a value unless body text explicitly overrides the checkbox state.")

    # Add form structure context if provided
    if acroform_subtree:
        lines.append("")
        lines.append("#### Form Structure Context")
        lines.append("```")
        tree_str = _format_acroform_subtree(acroform_subtree)
        if tree_str:
            lines.append(tree_str)
        else:
            lines.append("(subtree structure not available)")
        lines.append("```")

    # Add page window
    if page_window:
        lines.append(f"- **Page window**: {page_window} — start your search on these pages")
    elif conflict_type == "missing_child" and parent_path:
        lines.append(f"- **Page window**: Field location unknown — search parent section: {parent_path}")

    # Always end with required output format
    lines.append('- **Required output**: Value AND evidence in format "CHECKED/UNCHECKED ☒/☐ — label — interpreted as X" AND page number')
    lines.append("")

    return "\n".join(lines)


def build_reextraction_prompt(
    failing_fields: list[dict],
    low_confidence_fields: list[dict],
    layout_schema: dict,
    fields_context: list[dict] | None = None,
) -> str:
    """
    Build the complete re-extraction prompt.

    Args:
        failing_fields: List of failing field dicts from critic
        low_confidence_fields: List of low-confidence field dicts from critic
        layout_schema: Layout schema dict
        fields_context: Optional rich context from pipeline._build_reextraction_context()
            Each dict contains: field_name, failure_reason, acroform_value,
            llm_value, acroform_subtree, ancestor_path, pages, parent_path

    Returns:
        Complete prompt string for re-extraction
    """
    all_fields = failing_fields + low_confidence_fields

    if not all_fields:
        return ""

    # Build lookup from fields_context if provided
    context_lookup = {}
    if fields_context:
        for ctx in fields_context:
            field_name = ctx.get("field_name")
            if field_name:
                context_lookup[field_name] = ctx

    field_details_parts = []
    for field in all_fields:
        field_name = field.get("field", "")
        previous_value = field.get("extracted")
        rejection_reason = field.get("reason") or field.get("critic_reason", "Value did not match expected")

        # Get layout info for this field
        layout_info = layout_schema.get(field_name)

        # Get rich context if available
        ctx = context_lookup.get(field_name, {})
        acroform_value = ctx.get("acroform_value")
        acroform_subtree = ctx.get("acroform_subtree")
        page_window = ctx.get("pages") or []
        conflict_type = ctx.get("failure_reason") or ctx.get("conflict_type")
        parent_path = ctx.get("parent_path")

        # If previous_value not in field dict, try context
        if previous_value is None:
            previous_value = ctx.get("llm_value")

        field_detail = format_field_for_reextraction(
            field_name=field_name,
            previous_value=previous_value,
            rejection_reason=rejection_reason,
            layout_info=layout_info,
            acroform_value=acroform_value,
            acroform_subtree=acroform_subtree,
            page_window=page_window,
            conflict_type=conflict_type,
            parent_path=parent_path,
        )
        field_details_parts.append(field_detail)

    field_details = "\n".join(field_details_parts)

    return REEXTRACTION_PROMPT_TEMPLATE.format(field_details=field_details)
