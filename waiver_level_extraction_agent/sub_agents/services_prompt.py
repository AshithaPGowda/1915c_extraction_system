"""
Prompt templates for the Services Extraction Agent.

Extracts detailed service-level data from 1915(c) HCBS waiver PDFs,
producing one row per service per waiver with parent/child hierarchy.

Target schema is the "Waiver Services - PD_Append C&E" sheet (49 columns):
Appendix C attributes, Appendix E self-direction, and Appendix J cost
factors (capitation, unit, users, avg units/user, avg cost/unit, total cost)
for years 1-5. Component Cost is computed downstream, never extracted.
"""

import json
import os


def _load_layout() -> str:
    """Load the services layout schema (49-column sheet-2 format)."""
    layout_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "layouts", "layout_services_pass2_rich.json"
    )
    with open(layout_path) as f:
        layout = json.load(f)
    return json.dumps(layout, indent=2)


SERVICES_DISCOVERY_PROMPT = """
You are a specialist extraction agent for 1915(c) Home and Community-Based Services (HCBS)
Medicaid waiver documents. Your task is to discover ALL services listed in this waiver.

## Task: Service Discovery

Scan the entire PDF and identify ALL services mentioned in:
1. Appendix C (C-1/C-3) - Primary service definitions
2. Appendix J - Cost tables which may show service breakdowns/sub-services

For each service found, note:
- The exact service name as written in the document
- Whether it appears to be a main service (Appendix C) or a sub-service (Appendix J cost breakdown)
- The page number(s) where it appears

## Output Format

Return a JSON object with:
```json
{
  "services": [
    {
      "name": "Exact Service Name",
      "is_main_service": true,
      "pages": [15, 16, 142],
      "notes": "Optional notes about the service"
    }
  ],
  "appendix_c_page_range": [15, 45],
  "appendix_e_page_range": [46, 80],
  "appendix_j_page_range": [140, 160]
}
```

## Rules

1. Include ALL services, even if they seem similar
2. Preserve exact spelling and capitalization from the document
3. Note page ranges for each appendix to help with targeted extraction
4. Do NOT extract any field values yet - just discover services
5. List main services in the ORDER they appear in Appendix C-1/C-3

Return ONLY valid JSON, no markdown fences or explanatory text.
"""


# Shared description of the per-year Appendix J fields, used by both the
# single-service and batch prompts.
_YEAR_FIELDS_DOC = """
For EACH year (1-5) present in the Appendix J-2 cost tables, extract:
- `year_N_capitation`: 1 if the J-2d capitation checkbox is checked for this
  service, 0 if unchecked. null if Appendix J has no capitation column.
- `year_N_unit`: the unit of service EXACTLY as written (e.g. "15 min",
  "per hour", "Monthly", "As needed"). null if not shown.
- `year_N_users`: number of unduplicated users for this row.
- `year_N_avg_units_per_user`: average units per user (may be decimal).
- `year_N_avg_cost_per_unit`: average cost per unit (may be decimal).
- `year_N_total_cost`: the TOTAL COST shown in the table for this service-year.

Do NOT compute or extract Component Cost - it is derived downstream.
If the waiver has fewer than 5 years, use null for missing years.
"""


def build_service_extraction_prompt(
    service_name: str,
    state: str,
    waiver_title: str,
    file_name: str,
    appendix_hints: dict | None = None,
    guidance: str | None = None,
) -> str:
    """
    Build a focused extraction prompt for a single service.

    Args:
        service_name: The exact service name to extract
        state: The state name (from overview extraction)
        waiver_title: The waiver title (from overview extraction)
        file_name: The PDF filename
        appendix_hints: Optional dict with page hints for appendices
        guidance: Optional free-text steer for this one extraction (e.g. "the
            cost table is on page 210-215, not the usual Appendix J location"
            or "last attempt hallucinated a self-direction authority - this
            service is provider-managed only, verify against C-1 before
            filling that field"). Used for manually retrying/correcting one
            named service (CLI --only-service, or the ADK orchestrator's
            targeted single-service mode) - not used in the normal
            all-services discovery loop.

    Returns:
        The extraction prompt string
    """
    layout = _load_layout()

    page_hints = ""
    if appendix_hints:
        if appendix_hints.get("appendix_c_pages"):
            page_hints += f"\n- Appendix C (service definitions): pages {appendix_hints['appendix_c_pages']}"
        if appendix_hints.get("appendix_e_pages"):
            page_hints += f"\n- Appendix E (self-direction): pages {appendix_hints['appendix_e_pages']}"
        if appendix_hints.get("appendix_j_pages"):
            page_hints += f"\n- Appendix J (cost tables): pages {appendix_hints['appendix_j_pages']}"

    guidance_block = f"\n\n## Additional Guidance For This Attempt\n{guidance}" if guidance else ""

    return f"""
You are extracting detailed information for a SINGLE service from a 1915(c) HCBS waiver PDF.

## Service to Extract: "{service_name}"

## Waiver Context
- State: {state}
- Waiver Title: {waiver_title}
- File Name: {file_name}

## Page Hints{page_hints if page_hints else " (none available)"}{guidance_block}

## Field Schema (target sheet columns)
{layout}

## CRITICAL: Parent/Child Service Hierarchy

Services have a TWO-LEVEL HIERARCHY:

### Parent Service Row
- The main service from Appendix C-1/C-3 with all Appendix C attributes
  (Service Type, Delivery Method, Provider specs, HCBS Taxonomy)
- Appendix E attributes (Self-direction Authority, Info/Assistance) if applicable
- **Total Cost** per year from Appendix J (aggregated across all sub-services)
- When the service has children, the parent's per-year users/unit/avg fields
  are usually blank (they live on the child rows)

### Child Service Rows (sub-services)
- Service VARIATIONS found in Appendix J cost tables under the main service
  (e.g. "Day Habilitation" -> "Day Habilitation Level 1", "Day Habilitation Level 2")
- Child rows have: sub-service name, and per-year unit / # users /
  avg units per user / avg cost per unit from the J-2 table
- Child `year_N_total_cost` is "N/A" (costs aggregate at parent level)
- Child rows do NOT have: Service Type, Delivery Method, Provider specs,
  Self-direction, Capitation

## Extraction Instructions

1. **Find the parent service**: Locate "{service_name}" in Appendix C-1/C-3
   (check both the "Service:" line and "Alternate Service Title" line)
2. **Extract Appendix C attributes** for the parent:
   - `hcbs_taxonomy`: the HCBS Taxonomy category label for this service as
     shown in the C-1/C-3 specification (e.g. "Home-Based Services",
     "Day Services", "Nursing Services"). null if no taxonomy section.
   - `service_type` (1 = Statutory Service; 2 = Extended State Plan Service;
     3 = Supports for Participant Direction; 4 = Other Service)
   - `service_delivery_method` (1 = Participant-directed; 2 = Provider managed; 3 = Both)
   - `provider_eligibility`: who may provide the service
     (1 = Legally Responsible Person; 2 = Relative; 3 = Legal Guardian).
     Concatenate the applicable digits, e.g. "2", "23", "123". null if none checked.
   - `provider_specifications` (1 = agency; 2 = individual; 3 = both)
3. **Check Appendix E** for self-direction:
   - `self_direction_authority`: E-1g (1 = Employer Authority; 2 = Budget
     Authority; 3 = Both). null if this service is not participant-directed.
   - `information_assistance`: E-1j (1 if this service covers information &
     assistance in support of participant direction; null otherwise)
4. **Extract Appendix J cost factors** for "{service_name}":
{_YEAR_FIELDS_DOC}
5. **Identify children**: sub-rows/breakdowns under the main service in the
   J-2 tables (different levels, delivery modes, living arrangements).
   If the J-2 table has exactly one row for this service, there are NO children.

## Output Format

Return a JSON object:
```json
{{
  "parent": {{
    "row_type": "parent",
    "service_name": "{service_name}",
    "state": "{state}",
    "waiver_title": "{waiver_title}",
    "file_name": "{file_name}",
    "hcbs_taxonomy": "<taxonomy category or null>",
    "service_type": <1-4 or null>,
    "service_delivery_method": <1-3 or null>,
    "provider_eligibility": "<string like '2' or '123' or null>",
    "provider_specifications": <1-3 or null>,
    "self_direction_authority": <1-3 or null>,
    "information_assistance": <1 or null>,
    "year_1_capitation": <0, 1, or null>,
    "year_1_unit": "<unit string or null>",
    "year_1_users": <number or null>,
    "year_1_avg_units_per_user": <number or null>,
    "year_1_avg_cost_per_unit": <number or null>,
    "year_1_total_cost": <number or null>,
    ... same six fields for years 2-5 ...
    "evidence": {{
      "service_type": "verbatim text from PDF showing service type selection",
      "service_delivery_method": "verbatim text...",
      "hcbs_taxonomy": "verbatim taxonomy text...",
      "appendix_c_page": <page number>,
      "appendix_e_page": <page number or null>,
      "appendix_j_page": <page number>
    }}
  }},
  "children": [
    {{
      "row_type": "child",
      "service_name": "Sub-service Name",
      "parent_service": "{service_name}",
      "hcbs_taxonomy": "<same taxonomy as parent or null>",
      "year_1_unit": "<unit string or null>",
      "year_1_users": <number or null>,
      "year_1_avg_units_per_user": <number or null>,
      "year_1_avg_cost_per_unit": <number or null>,
      "year_1_total_cost": "N/A",
      ... same five fields for years 2-5 ...
      "evidence": {{
        "source": "verbatim text from Appendix J showing this sub-service",
        "appendix_j_page": <page number>
      }}
    }}
  ]
}}
```

## Rules

1. If the service has NO sub-breakdowns, return an empty "children" array
2. All costs and counts must be plain numbers (no currency symbols or commas)
3. Child rows MUST have `year_X_total_cost: "N/A"` - costs aggregate at parent
4. If a field cannot be found, use null (not empty string)
5. Evidence must be EXACT verbatim quotes from the PDF, with 1-indexed page numbers
6. Unit strings are copied EXACTLY as written in the J-2 table

Return ONLY valid JSON, no markdown fences or explanatory text.
"""


def build_batch_extraction_prompt(
    services: list[str],
    state: str,
    waiver_title: str,
    file_name: str,
) -> str:
    """
    Build a prompt to extract ALL services at once (for smaller waivers).

    This is more efficient for waivers with few services but may lose
    accuracy for complex waivers with many services.

    Args:
        services: List of service names to extract
        state: The state name
        waiver_title: The waiver title
        file_name: The PDF filename

    Returns:
        The extraction prompt string
    """
    layout = _load_layout()
    service_list = "\n".join(f"  - {s}" for s in services)

    return f"""
You are extracting detailed information for ALL services from a 1915(c) HCBS waiver PDF.

## Services to Extract
{service_list}

## Waiver Context
- State: {state}
- Waiver Title: {waiver_title}
- File Name: {file_name}

## Field Schema (target sheet columns)
{layout}

## CRITICAL: Parent/Child Service Hierarchy

Services have a TWO-LEVEL HIERARCHY.

### Parent Service Row
- Main service from Appendix C with all attributes (including HCBS Taxonomy)
- Appendix E self-direction attributes if applicable
- Total Cost per year from Appendix J (aggregated)
- Per-year users/unit/avg fields usually blank when children carry them

### Child Service Rows
- Sub-service variations from Appendix J cost breakdowns
- Have per-year unit / # users / avg units per user / avg cost per unit
- Cost shows "N/A" (aggregated at parent)

## Per-Year Appendix J Fields
{_YEAR_FIELDS_DOC}

## Output Format

Return a JSON object with all services:
```json
{{
  "waiver_key": "<normalized waiver key>",
  "state": "{state}",
  "waiver_title": "{waiver_title}",
  "file_name": "{file_name}",
  "services": [
    {{
      "parent": {{
        "row_type": "parent",
        "service_name": "Service Name",
        "hcbs_taxonomy": "<taxonomy category or null>",
        "service_type": <1-4>,
        "service_delivery_method": <1-3>,
        "provider_eligibility": "<string>",
        "provider_specifications": <1-3>,
        "self_direction_authority": <1-3 or null>,
        "information_assistance": <1 or null>,
        "year_1_capitation": <0, 1, or null>,
        "year_1_unit": "<unit or null>",
        "year_1_users": <number or null>,
        "year_1_avg_units_per_user": <number or null>,
        "year_1_avg_cost_per_unit": <number or null>,
        "year_1_total_cost": <number or null>,
        ... years 2-5 ...
      }},
      "children": [
        {{
          "row_type": "child",
          "service_name": "Sub-service Name",
          "parent_service": "Parent Service Name",
          "hcbs_taxonomy": "<taxonomy or null>",
          "year_1_unit": "<unit or null>",
          "year_1_users": <number>,
          "year_1_avg_units_per_user": <number or null>,
          "year_1_avg_cost_per_unit": <number or null>,
          "year_1_total_cost": "N/A",
          ... years 2-5 ...
        }}
      ]
    }}
  ]
}}
```

## Rules

1. Extract EVERY service in the list - do not skip any
2. Preserve exact service names from the document
3. If a service has no sub-breakdowns, use empty "children" array
4. All costs numeric (no $ or commas); unit strings verbatim
5. Child costs MUST be "N/A" - aggregated at parent
6. Use null for fields that cannot be found

Return ONLY valid JSON, no markdown fences.
"""
