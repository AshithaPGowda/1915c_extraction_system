#!/usr/bin/env python3
"""
Export overview.json files to a global CSV.

Usage:
    python data/archive/scripts/export_to_csv.py                     # Export all processed waivers
    python data/archive/scripts/export_to_csv.py --waiver NH0397R0500  # Export specific waiver
    python data/archive/scripts/export_to_csv.py --force             # Overwrite existing entries
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from waiver_level_extraction_agent.shared_libraries.paths import (  # noqa: E402
    WAIVER_LEVEL_CSV as CSV_PATH,  # new results only - never the verified master
    OUTPUTS_DIR,
)

# Canonical field order (165 fields - excluding _evidence and _page suffixes)
# This ensures every row has the same columns in the same order
CANONICAL_FIELDS = [
    "waiver_key",  # First column is always waiver_key
    "1.A State",
    "1.B Waiver Title",
    "1.C Waiver number",
    "1.C. Requested Approval Period (years)",
    "1.D. Type of Waiver (1=regular waiver, 2= amendment; 3 Model Waiver)",
    "1.E. Approved Effective Date",
    "Actual Expiration Date (leave blank if still active)",
    "1.F. Hospital Level of Care (0 = No, 1= Yes)",
    "1.F. Hospital as defined in 42 CFR §440.10 Hospital Level of Care Subcategories (0 = No, 1= Yes)",
    "1.F. Hospital as defined in 42 CFR §440.10 Hospital Level of Care Subcategories (fill if YES to the previous question, otherwise leave blank)",
    "1.F. Hospital as defined in 42 CFR §440.10 Hospital Level of Care Subcategories",
    "1.F. Hospital: Inpatient psychiatric facility for individuals age 21 and under as provided in42 CFR §440.160 (0 = No, 1=Yes)",
    "1.F. Nursing Facility Level of Care (0 = No, 1= Yes)",
    "1.F. Nursing Facility as defined in 42 CFR §440.40 and 42 CFR §440.155  (0 = No, 1= Yes)",
    "1.F. Nursing Facility as defined in 42 CFR §440.40 and 42 CFR §440.155 Subcategories (fill if YES to the previous question, otherwise leave blank)",
    "1.F. Institution for Mental Disease for persons with mental illnesses aged 65 and older as provided in 42 CFR §440.140 (0 = No, 1= Yes)",
    "1.F. ICF/IID Level of Care (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Not applicable (1 = Yes, 0=not applicable)",
    "1.G. Concurrent Operation with Other Programs: Services furnished under the provisions of §1915(a)(1)(a) of the Act and described in Appendix I  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(1)  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(2)  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(3)  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: Waiver(s) authorized under §1915(b) of the Act: §1915(b)(4)  (0 = No, 1= Yes)",
    "1.G. Concurrent Operation with Other Programs: A program operated under §1932(a) of the Act.",
    "1.G. Concurrent Operation with Other Programs: A program authorized under §1915(i) of the Act.",
    "1.G. Concurrent Operation with Other Programs: A program authorized under §1915(j) of the Act",
    "1.G. A program authorized under §1115 of the Act",
    "1.H. Dual Eligiblity for Medicaid and Medicare (0 = No, 1 =Yes)",
    "2. Waiver Population (extract from Brief Waiver Description)",
    "3.E. Participant-Direction of Services (0=No, 1=Yes)",
    "4B. Income and Resources for the Medically Needy (NA if not applicable, 0=No, 1=Yes)",
    "4C. Statewideness (0=No, 1=Yes)",
    "4C. Statewideness is requested in order to make participant-direction of services (0=No, 1=Yes, leave blank if answer to the previous question is 0)",
    "B-1a Waiver Population: aged (0 = No; 1 = Yes) ",
    "Waiver Population: aged (minimum age). Leave age columns blank if the waiver does not serve this population ",
    "Waiver Population: aged (maximum age)",
    "B-1a Waiver Population: disabled, physical (0 = No; 1 = Yes) ",
    "Waiver Population: disabled, physical (minimum age)",
    "Waiver Population: disabled, physical (maximum age)",
    "B-1a Waiver Population: disabled, other (0 = No; 1 = Yes) ",
    "Waiver Population: disabled, other (minimum age)",
    "Waiver Population: disabled, other (maximum age)",
    "B-1a Waiver Population: Brain Injury (0 = No; 1 = Yes) ",
    "Waiver Population: Brain Injury (minimum age)",
    "Waiver Population: Brain Injury (maximum age)",
    "B-1a Waiver Population: HIV/AIDS (0 = No; 1 = Yes) ",
    "Waiver Population: HIV/AIDS (minimum age)",
    "Waiver Population: HIV/AIDS (maximum age)",
    "Waiver Population: HIV/AIDS (minimum age).1",
    "Waiver Population: HIV/AIDS (maximum age).1",
    "B-1a Waiver Population: Medically Fragile (0 = No; 1 = Yes) ",
    "B-1a Waiver Population: Technology Dependent (0 = No; 1 = Yes) ",
    "Waiver Population: Technology Dependent (minimum age)",
    "Waiver Population: Technology Dependent (maximum age)",
    "B-1a Waiver Population: Autism (0 = No; 1 = Yes) ",
    "Waiver Population: Autism (minimum age)",
    "Waiver Population: Autism (maximum age)",
    "B-1a Waiver Population: Developmental Disability (0 = No; 1 = Yes) ",
    "Waiver Population: Developmental Disability (minimum age)",
    "Waiver Population: Developmental Disability (maximum age)",
    "B-1a  Waiver Population: Intellectual Disability (0 = No; 1 = Yes) ",
    "Waiver Population: Intellectual Disability (minimum age)",
    "Waiver Population: Intellectual Disability (maximum age)",
    "B-1a Waiver Population: Mental Illness (0 = No; 1 = Yes) ",
    "Waiver Population: Mental Illness (minimum age)",
    "Waiver Population: Mental Illness (maximum age)",
    "B-1a Waiver Population: Serious Emotional Disturbance (0 = No; 1 = Yes) ",
    "Waiver Population: Serious Emotional Disturbance (minimum age)",
    "Waiver Population: Serious Emotional Disturbance (maximum age)",
    "B-1b Additional Criteria for waiver population",
    "B-2a: Individual Cost Limit (0 = No Cost Limit; 1= Cost Limit in Excess of Institutional Costs; 2= Institutional Cost Limit; 3=Cost Limit Lower Than Institutional Costs)",
    "B-3a: Unduplicated Number of Participants, Year 1",
    "B-3a: Unduplicated Number of Participants, Year 2",
    "B-3a: Unduplicated Number of Participants, Year 3",
    "B-3a: Unduplicated Number of Participants, Year 4",
    "B-3a: Unduplicated Number of Participants, Year 5",
    "B-3c: Reserved Waiver Capacity (0 = No, 1= yes)",
    "B-3c: Reserved Waiver Capacity (If Yes to the previous question, describe the purpose)",
    "B-4a. State classification (1= 1634 state; 2=SSI Criteria state; 3=209(b) state)",
    "B-4a. Miller Trust State. (0=Yes, 1=No)",
    "B-4b. Medicaid Eligibility Groups Served in the Waiver (There are many choices for this question. Please just copy and paste the selected one here)",
    "B-4b. Special home and community-based waiver group under 42 CFR §435.217 (0 = No; 1= Yes)",
    "C-1b. Provision of Case Management Services to Waiver Participants (0 = not applicable; 1= applicable)",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a waiver service defined in Appendix C-3.)",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a Medicaid state plan service under §1915(i) of the Act (HCBS as a State Plan Option).",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a Medicaid state plan service under §1915(g)(1) of the Act (Targeted Case Management).",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As an administrative activity.",
    "C-1b. Provision of Case Management Services to Waiver Participants (If Yes to care management services) As a primary care case management system service under a concurrent managed care authority",
    "C-2d. Provision of Personal Care or Similar Services by Legally Responsible Individuals. (0 = No payment to legally responsible individuals for furnishing personal care or similar services. 1=Yes)",
    "D-1.a. Responsibility for Service Plan Development, Registered nurse, licensed to practice in the state",
    "D-1.a. Responsibility for Service Plan Development, Licensed practical or vocational nurse, acting within the scope of practice under state law",
    "D-1.a. Responsibility for Service Plan Development, Licensed physician (M.D. or D.O)",
    "D-1.a. Responsibility for Service Plan Development, Case Manager (qualifications specified in Appendix C-1/C-3)",
    "D-1.a. Responsibility for Service Plan Development, Case Manager (qualifications not specified in Appendix C-1/C-3).",
    "D-1.a. Responsibility for Service Plan Development, Social Worker",
    "D-1.a. Responsibility for Service Plan Development, Other",
    "Participant direction opportunities (0 = No, 1=Yes)",
    "E-1b Participant direction opportunities? (employer authority (code as 1), budget authority (code as 2), both (code as 3))",
    "E-1c Availability of Participant Direction by Type of Living Arrangement. (1 = private residence or the home of a family member; 2= reside in other living arrangements < 4 persons; 3 = other)",
    "E-1c Availability of Participant Direction by Type of Living Arrangement (specify when E-1c = 2)",
    "E-1f Use of representative for PD (0 =representative not allowed; 1= representative allowed)",
    "E-1f Use of representative for PD (1= legal representative; 2=non-legal representative; 3 = both) Only if E-1f = 1",
    "E-1h Financial Management Services [FMS] (0 = no financial management services; 1=have financial management services)",
    "E-1h FMS by government entities (==1) or private entities (==2), or Both (==3)",
    "E-1i FMS are covered as the waiver services (==1) or as an administrative activity (==2)",
    "Scope of FMS Employment authority: Assist participant in verifying support worker citizenship status (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Collect and process timesheets of support workers (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Process payroll, withholding, filing and payment of applicable federal, state and local employment-related taxes and insurance (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Other (0 = No, 1= Yes)",
    "Scope of FMS Employment authority: Other specify (Let's SKIP this for now)",
    "Scope of FMS Budget authority: Maintain a separate account for each participant's participant-directed budget (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Track and report participant funds, disbursements and the balance of participant funds (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Process and pay invoices for goods and services approved in the service plan (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Provide participant with periodic reports of expenditures and the status of the participant-directed budget (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Other (0 = No, 1= Yes)",
    "Scope of FMS Budget authority: Other specify (Let's SKIP this for now)",
    "Scope of FMS, additional funcitions and activities: Execute and hold Medicaid provider agreements as authorized under a written agreement with the Medicaid agency",
    "Scope of FMS, additional funcitions and activities: Receive and disburse funds for the payment of participant-directed services under an agreement with the Medicaid agency or operating agency",
    "Scope of FMS, additional funcitions and activities: Provide other entities specified by the state with periodic reports of expenditures and the status of the participant-directed budget",
    "Scope of FMS, additional funcitions and activities: Other",
    "Scope of FMS, additional funcitions and activities: Other specify (Let's SKIP this for now)",
    "E-1j Information and Assistance in Support of Participant Direction: Case Management Activity (0=No, 1=Yes)",
    "E-1j Information and Assistance in Support of Participant Direction: Waiver Service Coverage (0=No, 1=Yes)",
    "E-1j Information and Assistance in Support of Participant Direction: Administrative Activity (0=No, 1=Yes)",
    "E-1k Independent Advocacy (0=No, 1=Yes)",
    "E-1n Goals for Participant Direction. Employer Authority, Year 1",
    "E-1n Goals for Participant Direction. Employer Authority, Year 2",
    "E-1n Goals for Participant Direction. Employer Authority, Year 3",
    "E-1n Goals for Participant Direction. Employer Authority, Year 4",
    "E-1n Goals for Participant Direction. Employer Authority, Year 5",
    "E-1n Goals for Participant Direction. Budget Authority only or combined with Employer Authority, Year 1",
    "E-1n Goals for Participant Direction. Budget Authority only or combined with Employer Authority, Year 2",
    "E-1n Goals for Participant Direction. Budget Authority only or combined with Employer Authority, Year 3",
    "E-1n Goals for Participant Direction. Budget Authority only or combined with Employer Authority, Year 4",
    "E-1n Goals for Participant Direction. Budget Authority only or combined with Employer Authority, Year 5",
    "E-2b Participant Exercise of Budget Flexibility (1=Modifications to the participant directed budget must be preceded by a change in the service plan; 2=The participant has the authority to modify the services included in the participant directed budget without prior approval)",
    "Unduplicated Participants Level of care, hospital, Year 1",
    "Unduplicated Participants Level of care, Nursing Facility, Year 1",
    "Unduplicated Participants Level of care, ICF/IID, Year 1",
    "Unduplicated Participants Level of care, hospital, Year 2",
    "Unduplicated Participants Level of care, Nursing Facility, Year 2",
    "Unduplicated Participants Level of care, ICF/IID, Year 2",
    "Unduplicated Participants Level of care, hospital, Year 3",
    "Unduplicated Participants Level of care, Nursing Facility, Year 3",
    "Unduplicated Participants Level of care, ICF/IID, Year 3",
    "Unduplicated Participants Level of care, hospital, Year 4",
    "Unduplicated Participants Level of care, Nursing Facility, Year 4",
    "Unduplicated Participants Level of care, ICF/IID, Year 4",
    "Unduplicated Participants Level of care, hospital, Year 5",
    "Unduplicated Participants Level of care, Nursing Facility, Year 5",
    "Unduplicated Participants Level of care, ICF/IID, Year 5",
    "Total spending, Year 1",
    "Total spending, Year 2",
    "Total spending, Year 3",
    "Total spending, Year 4",
    "Total spending, Year 5",
    "Independence Plus designation (0 = not requsted; 1= requested) ",
    "Transition 1: Transition to Waiver Services Tab to code specific services",
    "Transition 3: Transition to Waiver Services, column J",
    "Transition 5: If answer to the previous question is Yes, transition to Waiver Services tab column L",
    "Transition 7: Go to PD authority tab",
    "Transition 9: Transition to Waiver Services, column N",
]


def load_existing_waiver_keys() -> set:
    """Load existing waiver keys from CSV to avoid duplicates."""
    if not CSV_PATH.exists():
        return set()

    existing = set()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "waiver_key" in row:
                existing.add(row["waiver_key"])
    return existing


def read_overview_json(waiver_dir: Path) -> dict | None:
    """Read overview.json from a waiver output directory."""
    overview_path = waiver_dir / "overview.json"
    if not overview_path.exists():
        return None

    with open(overview_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_main_fields(data: dict) -> dict:
    """Extract main fields (excluding _evidence, _page suffixes and meta fields)."""
    extracted = data.get("extracted_fields", {})
    waiver_key = data.get("waiver_key", "")

    row = {"waiver_key": waiver_key}

    for field in CANONICAL_FIELDS[1:]:  # Skip waiver_key, already added
        value = extracted.get(field)
        # Convert to string for CSV, handle None
        if value is None:
            row[field] = ""
        elif isinstance(value, (list, dict)):
            row[field] = json.dumps(value)
        else:
            row[field] = str(value)

    return row


def export_waiver(waiver_key: str, force: bool = False) -> bool:
    """Export a single waiver to CSV. Returns True if exported."""
    existing = load_existing_waiver_keys()

    if waiver_key in existing and not force:
        print(f"  Skipping {waiver_key} (already in CSV)")
        return False

    waiver_dir = OUTPUTS_DIR / waiver_key
    data = read_overview_json(waiver_dir)

    if not data:
        print(f"  Skipping {waiver_key} (no overview.json)")
        return False

    row = extract_main_fields(data)

    # Check if CSV exists and has content (not just empty file)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    # Ensure output directory exists
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Exported {waiver_key}")
    return True


def export_all(force: bool = False) -> int:
    """Export all processed waivers to CSV. Returns count of exported."""
    existing = load_existing_waiver_keys() if not force else set()

    # Find all waiver directories
    waiver_dirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]

    print(f"Found {len(waiver_dirs)} waiver directories")
    print(f"Already in CSV: {len(existing)}")

    exported = 0
    for waiver_dir in sorted(waiver_dirs):
        waiver_key = waiver_dir.name

        if waiver_key in existing:
            print(f"  Skipping {waiver_key} (already in CSV)")
            continue

        data = read_overview_json(waiver_dir)
        if not data:
            print(f"  Skipping {waiver_key} (no overview.json)")
            continue

        row = extract_main_fields(data)

        # Check if CSV exists and has content (not just empty file)
        write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

        # Ensure output directory exists
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"  Exported {waiver_key}")
        exported += 1

    return exported


def main():
    parser = argparse.ArgumentParser(description="Export overview.json files to global CSV")
    parser.add_argument("--waiver", type=str, help="Export specific waiver key")
    parser.add_argument("--force", action="store_true", help="Overwrite existing entries")
    args = parser.parse_args()

    print(f"CSV path: {CSV_PATH}")
    print(f"Canonical fields: {len(CANONICAL_FIELDS)}")
    print()

    if args.waiver:
        success = export_waiver(args.waiver, force=args.force)
        print(f"\nExported: {1 if success else 0}")
    else:
        count = export_all(force=args.force)
        print(f"\nTotal exported: {count}")


if __name__ == "__main__":
    main()
