"""
Schema Field → AcroForm Path Mapping.

Central location for all checkbox field mappings between:
- Schema field names (from layout_waiver_overview.json)
- AcroForm paths (colon-separated, e.g., "svloc:locHosp")

This mapping enables deterministic verification of extracted values
against AcroForm metadata without requiring ground truth.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA TO ACROFORM MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
# Maps schema field names (exact match) to AcroForm paths.
# AcroForm paths are colon-separated hierarchical identifiers.

SCHEMA_TO_ACROFORM: dict[str, str] = {
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1.F — Level of Care
    # ═══════════════════════════════════════════════════════════════
    "1.F. Hospital Level of Care (0 = No, 1= Yes)": "svloc:locHosp",
    "1.F. Hospital as defined in 42 CFR §440.10 Hospital Level of Care Subcategories (0 = No, 1= Yes)": "svloc:locHospType",
    "1.F. Nursing Facility Level of Care (0 = No, 1= Yes)": "svloc:locNurFac",
    "1.F. Nursing Facility as defined in 42 CFR §440.40 and 42 CFR §440.155  (0 = No, 1= Yes)": "svloc:locNurFacType",
    "1.F. ICF/IID Level of Care (0 = No, 1= Yes)": "svloc:locICFMR",
    "1.F. Institution for Mental Disease for persons with mental illnesses aged 65 and older as provided in 42 CFR §440.140 (0 = No, 1= Yes)": "svloc:locIMD",

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1.G — Concurrent Operation
    # ═══════════════════════════════════════════════════════════════
    "1.G. Concurrent Operation with Other Programs: Not applicable (1 = Yes, 0=not applicable)": "svconcurrentOp:concOp",
    "1.G. Concurrent Operation with Other Programs: Services furnished under the provisions of §1915(a)(1)(a) of the Act and described in Appendix I  (0 = No, 1= Yes)": "svconcurrentOp:conc1915a",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act  (0 = No, 1= Yes)": "svconcurrentOp:conc1915b",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(1)  (0 = No, 1= Yes)": "svconcurrentOp:conc1915b1",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(2)  (0 = No, 1= Yes)": "svconcurrentOp:conc1915b2",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(3)  (0 = No, 1= Yes)": "svconcurrentOp:conc1915b3",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(4)  (0 = No, 1= Yes)": "svconcurrentOp:conc1915b4",
    "1.G. Concurrent Operation with Other Programs: A program operated under §1932(a) of the Act.": "svconcurrentOp:conc1932a",
    "1.G. Concurrent Operation with Other Programs: A program authorized under §1915(i) of the Act.": "svconcurrentOp:conc1915i",
    "1.G. Concurrent Operation with Other Programs: A program authorized under §1915(j) of the Act": "svconcurrentOp:conc1915j",
    "1.G. A program authorized under §1115 of the Act": "svconcurrentOp:conc1115",

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1.H — Dual Eligibility
    # ═══════════════════════════════════════════════════════════════
    "1.H. Dual Eligiblity for Medicaid and Medicare (0 = No, 1 =Yes)": "svconcurrentOp:concMedicaidMedicare",

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4B/4C — Income Resources & Statewideness
    # ═══════════════════════════════════════════════════════════════
    "4B. Income and Resources for the Medically Needy (NA if not applicable, 0=No, 1=Yes)": "svwaiverReq:incRes1902a",
    "4C. Statewideness (0=No, 1=Yes)": "svwaiverReq:statewide",
    "4C. Statewideness is requested in order to make participant-direction of services (0=No, 1=Yes, leave blank if answer to the previous question is 0)": "svwaiverReq:swidePDLim",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX B-1 — Waiver Population
    # ═══════════════════════════════════════════════════════════════
    "B-1a Waiver Population: aged (0 = No; 1 = Yes) ": "svapdxB1_1:tgagAgedInc",
    "B-1a Waiver Population: disabled, physical (0 = No; 1 = Yes) ": "svapdxB1_1:tgagDisPhyInc",
    "B-1a Waiver Population: disabled, other (0 = No; 1 = Yes) ": "svapdxB1_1:tgagDisOthInc",
    "B-1a Waiver Population: Brain Injury (0 = No; 1 = Yes) ": "svapdxB1_1:tgagBraInjInc",
    "B-1a Waiver Population: HIV/AIDS (0 = No; 1 = Yes) ": "svapdxB1_1:tgagHivAidsInc",
    "B-1a Waiver Population: Medically Fragile (0 = No; 1 = Yes) ": "svapdxB1_1:tgagMedFraInc",
    "B-1a Waiver Population: Technology Dependent (0 = No; 1 = Yes) ": "svapdxB1_1:tgagTecDepInc",
    "B-1a Waiver Population: Autism (0 = No; 1 = Yes) ": "svapdxB1_1:tgddAutismInc",
    "B-1a Waiver Population: Developmental Disability (0 = No; 1 = Yes) ": "svapdxB1_1:tgddDevDisInc",
    "B-1a  Waiver Population: Intellectual Disability (0 = No; 1 = Yes) ": "svapdxB1_1:tgddMenRetInc",
    "B-1a Waiver Population: Mental Illness (0 = No; 1 = Yes) ": "svapdxB1_1:tgmiMIInc",
    "B-1a Waiver Population: Serious Emotional Disturbance (0 = No; 1 = Yes) ": "svapdxB1_1:tgmiEmoDisInc",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX B-3/B-4 — Eligibility & Miller Trust
    # ═══════════════════════════════════════════════════════════════
    "B-3c: Reserved Waiver Capacity (0 = No, 1= yes)": "svapdxB3_2:elgQtyCapRsvd",
    "B-4a. Miller Trust State. (0=Yes, 1=No)": "svapdxB4_1:millTrStYN",
    "B-4b. Special home and community-based waiver group under 42 CFR §435.217 (0 = No; 1= Yes)": "svapdxB4_1:elgGrpSpecHomCom",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX C — Case Management / Provider Qualifications
    # ═══════════════════════════════════════════════════════════════
    "C-1b. Provision of Case Management Services to Waiver Participants (0 = not applicable; 1= applicable)": "svapdxC1_2:svcCsMgmt",
    # Sub-checkboxes of C-1b (only reachable/meaningful when svcCsMgmt=1 above) -
    # previously unmapped, which is why they landed blank instead of a real
    # 0/1 value: the AcroForm has all five as siblings under svapdxC1_2.
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a waiver service defined in Appendix C-3.)": "svapdxC1_2:svcCsMgmtSvc",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a Medicaid state plan service under §1915(i) of the Act (HCBS as a State Plan Option).": "svapdxC1_2:svcCsMgmtHcbs",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a Medicaid state plan service under §1915(g)(1) of the Act (Targeted Case Management).": "svapdxC1_2:svcCsMgmtTrg",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As an administrative activity.": "svapdxC1_2:svcCsMgmtAdmin",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a primary care case management system service under a concurrent managed care authority": "svapdxC1_2:svcCsMgmtCmca",
    "C-2d. Provision of Personal Care or Similar Services by Legally Responsible Individuals. (0 = No payment to legally responsible individuals for furnishing personal care or similar services. 1=Yes)": "svapdxC2_3:pcLegRspIndv",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX D — Service Plan
    # ═══════════════════════════════════════════════════════════════
    "D-1.a. Responsibility for Service Plan Development, Other": "svapdxD1_2:plnDevrOthSvcAllwd",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX E — Participant Direction
    # ═══════════════════════════════════════════════════════════════
    "3.E. Participant-Direction of Services (0=No, 1=Yes)": "svcomponents:particDirSvc",
    "Participant direction opportunities (0 = No, 1=Yes)": "svapdxE0_1:particDirSvc",
    "Independence Plus designation (0 = not requsted; 1= requested) ": "svapdxE0_1:dosReqIndPlusDsgn",

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX G — Health and Welfare Safeguards
    # ═══════════════════════════════════════════════════════════════
    "G-2a. Safeguards: Restraints (0=No, 1=Yes)": "svapdxG2_1:sfgRstrntAlwd",
    "G-2b. Safeguards: Restrictive Interventions (0=No, 1=Yes)": "svapdxG2_2:sfgResIntAlwd",
    "G-2c. Safeguards: Seclusion (0=No, 1=Yes)": "svapdxG2_3:sfgSeclAlwd",
    "G-3. Safeguards: Medication Administration (0=No, 1=Yes)": "svapdxG3_1:sfgMedsApp",
}


# ═══════════════════════════════════════════════════════════════════════════════
# INVERTED FIELDS
# ═══════════════════════════════════════════════════════════════════════════════
# Fields where the coding is inverted: 0=Yes, 1=No (opposite of standard)
# When comparing LLM extraction to AcroForm, flip the extracted value first.

INVERTED_FIELDS: set[str] = {
    "B-4a. Miller Trust State. (0=Yes, 1=No)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BODY TEXT ONLY FIELDS
# ═══════════════════════════════════════════════════════════════════════════════
# Checkbox fields that do NOT have AcroForm backing in CMS PDFs.
# These must be verified via LLM evidence check only.

BODY_TEXT_ONLY_FIELDS: set[str] = {
    # FMS Employment authority fields
    "Scope of FMS Employment authority: Assist participant in verifying support worker citizenship status (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Collect and process timesheets of support workers (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Process payroll, withholding, filing and payment of applicable federal, state and local employment-related taxes and insurance (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Other (0 = No, 1= Yes)",
    # FMS Budget authority fields
    "Scope of FMS Budget authority: Maintain a separate account for each participant's participant-directed budget (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Track and report participant funds, disbursements and the balance of participant funds (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Process and pay invoices for goods and services approved in the service plan (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Provide participant with periodic reports of expenditures and the status of the participant-directed budget (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Other (0 = No, 1= Yes)",
    # E-1j/E-1k fields
    "E-1j Information and Assistance in Support of Participant Direction: Case Management Activity (0=No, 1=Yes)",
    "E-1j Information and Assistance in Support of Participant Direction: Waiver Service Coverage (0=No, 1=Yes)",
    "E-1j Information and Assistance in Support of Participant Direction: Administrative Activity (0=No, 1=Yes)",
    "E-1k Independent Advocacy (0=No, 1=Yes)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_acroform_path(field_name: str) -> str | None:
    """Get the AcroForm path for a schema field name."""
    return SCHEMA_TO_ACROFORM.get(field_name)


def is_inverted_field(field_name: str) -> bool:
    """Check if a field has inverted coding (0=Yes, 1=No)."""
    return field_name in INVERTED_FIELDS


def is_body_text_only(field_name: str) -> bool:
    """Check if a field is body text only (no AcroForm backing)."""
    return field_name in BODY_TEXT_ONLY_FIELDS


def has_acroform_mapping(field_name: str) -> bool:
    """Check if a field has an AcroForm mapping."""
    return field_name in SCHEMA_TO_ACROFORM


def get_all_acroform_fields() -> list[str]:
    """Get all schema field names that have AcroForm mappings."""
    return list(SCHEMA_TO_ACROFORM.keys())


def get_all_body_text_fields() -> list[str]:
    """Get all schema field names that are body text only."""
    return list(BODY_TEXT_ONLY_FIELDS)
