"""
Discovery and Evidence Analytics Tools.

Read-only analytics: PDF/waiver discovery, batch status checks, and deep
evidence/history lookups for a field or waiver. None of these touch the
extraction pipeline itself - they only read logs, stage outputs, output
files, and the CSV export, and answer analyst-style questions about
already-extracted data.

Consolidated from the former batch_processing_tools.py, csv_tools.py, and
the analytics half of pipeline_control_tools.py (which kept run_from_stage
and apply_batch_corrections - those two are pipeline-critical, not
analytics, and stayed in pipeline_control_tools.py).
"""

import json

from waiver_level_extraction_agent.adk_agents.tools.stage_storage_tools import load_stage_output, get_pipeline_status
from .constants import PROJECT_ROOT, OUTPUT_DIR, LOGS_DIR, PDF_DIR

# OUTPUTS_DIR here is the extraction pipeline's output directory
# (waiver_level_extraction_agent/data/outputs/), unrelated to this package's own name.
OUTPUTS_DIR = OUTPUT_DIR
DEFAULT_PDF_DIR = PDF_DIR


def list_pdfs_in_directory(directory: str = None, filter_pattern: str = None) -> dict:
    """List all PDF files in a directory.

    Args:
        directory: Path to directory. If None, uses default pdfs_267_latest/
        filter_pattern: Optional filter - e.g., "TX*" for Texas waivers,
                       "*R0700*" for specific revision

    Returns:
        dict with list of PDF files and their waiver keys
    """
    if directory:
        pdf_dir = Path(directory)
    else:
        pdf_dir = DEFAULT_PDF_DIR

    if not pdf_dir.exists():
        return {
            "status": "error",
            "error": f"Directory not found: {pdf_dir}",
        }

    try:
        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))

        if filter_pattern:
            import fnmatch
            pdf_files = [f for f in pdf_files if fnmatch.fnmatch(f.stem, filter_pattern)]

        pdfs = []
        for pdf_file in sorted(pdf_files):
            waiver_key = pdf_file.stem.upper()

            if "appendix" in waiver_key.lower() or "app-k" in waiver_key.lower():
                continue

            pdfs.append({
                "waiver_key": waiver_key,
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
            })

        return {
            "status": "success",
            "directory": str(pdf_dir),
            "total_pdfs": len(pdfs),
            "filter_applied": filter_pattern,
            "pdfs": pdfs,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_batch_status(waiver_keys: list[str]) -> dict:
    """Get pipeline status for a list of waivers.

    Useful to check which waivers in a batch have been processed
    and which still need work.

    Args:
        waiver_keys: List of waiver keys to check

    Returns:
        dict with status for each waiver
    """
    statuses = []
    completed = 0
    in_progress = 0
    not_started = 0

    for waiver_key in waiver_keys:
        status = get_pipeline_status(waiver_key)

        if status.get("status") == "success":
            completed_stages = status.get("completed_stages", [])

            if "normalization" in completed_stages:
                state = "completed"
                completed += 1
            elif completed_stages:
                state = "in_progress"
                in_progress += 1
            else:
                state = "not_started"
                not_started += 1

            statuses.append({
                "waiver_key": waiver_key,
                "state": state,
                "completed_stages": completed_stages,
                "next_stage": status.get("next_stage"),
            })
        else:
            statuses.append({
                "waiver_key": waiver_key,
                "state": "not_started",
                "completed_stages": [],
                "next_stage": "checkbox_resolution",
            })
            not_started += 1

    return {
        "status": "success",
        "total": len(waiver_keys),
        "completed": completed,
        "in_progress": in_progress,
        "not_started": not_started,
        "statuses": statuses,
    }


def filter_highest_versions(waiver_keys: list[str]) -> dict:
    """Filter a list of waiver keys to only include highest versions.

    Waiver keys follow pattern: SS####R####
    - SS = state code (2 letters)
    - #### = waiver number (4 digits)
    - R#### = revision (R + 4 digits, e.g., R0700)

    This returns only the highest revision for each state+waiver combination.

    Args:
        waiver_keys: List of waiver keys

    Returns:
        dict with filtered list (highest versions only)
    """
    import re

    parsed = {}
    pattern = r'^([A-Z]{2})(\d+)R(\d+)$'

    for key in waiver_keys:
        key_upper = key.upper()
        match = re.match(pattern, key_upper)
        if match:
            state = match.group(1)
            waiver_num = match.group(2)
            revision = int(match.group(3))

            base_key = f"{state}{waiver_num}"
            if base_key not in parsed or parsed[base_key]["revision"] < revision:
                parsed[base_key] = {
                    "waiver_key": key_upper,
                    "revision": revision,
                }

    highest = [info["waiver_key"] for info in parsed.values()]

    return {
        "status": "success",
        "input_count": len(waiver_keys),
        "output_count": len(highest),
        "filtered_out": len(waiver_keys) - len(highest),
        "highest_versions": sorted(highest),
    }


def get_csv_status() -> dict:
    """Get the current status of the global CSV export.

    Shows how many waivers are in the CSV vs how many have been processed.

    Returns:
        dict with CSV status
    """
    from waiver_level_extraction_agent.shared_libraries.paths import WAIVER_LEVEL_CSV
    csv_path = WAIVER_LEVEL_CSV  # new extractions; the verified master is data/verified/all_waivers.csv

    if not csv_path.exists():
        return {
            "status": "success",
            "csv_exists": False,
            "csv_count": 0,
            "message": "CSV file does not exist yet",
        }

    with open(csv_path) as f:
        import csv as csv_module
        reader = csv_module.DictReader(f)
        csv_keys = set()
        for row in reader:
            if row.get("waiver_key"):
                csv_keys.add(row["waiver_key"])

    processed_count = 0
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and (d / "overview.json").exists():
                processed_count += 1

    return {
        "status": "success",
        "csv_exists": True,
        "csv_path": str(csv_path),
        "csv_count": len(csv_keys),
        "processed_count": processed_count,
        "missing_from_csv": processed_count - len(csv_keys),
        "message": f"{len(csv_keys)} waivers in CSV, {processed_count} processed",
    }


def get_evidence_from_logs(waiver_key: str, field_name: str = None) -> dict:
    """Get evidence and errors from agent logs for a waiver.

    Parses the JSONL log file to find extraction evidence, errors,
    and the history of what happened during processing.

    Args:
        waiver_key: The waiver identifier
        field_name: Optional - filter to specific field

    Returns:
        dict with log entries, errors, and evidence
    """
    log_file = LOGS_DIR / f"{waiver_key}.jsonl"

    if not log_file.exists():
        return {
            "status": "not_found",
            "waiver_key": waiver_key,
            "message": f"No log file found at {log_file}",
        }

    try:
        entries = []
        errors = []
        field_entries = []

        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)

                    if "error" in entry or entry.get("action", "").endswith("_error"):
                        errors.append(entry)

                    if field_name:
                        entry_field = entry.get("field") or entry.get("field_name")
                        if entry_field == field_name:
                            field_entries.append(entry)

                except json.JSONDecodeError:
                    continue

        result = {
            "status": "success",
            "waiver_key": waiver_key,
            "total_entries": len(entries),
            "error_count": len(errors),
            "errors": errors[-10:] if errors else [],
        }

        if field_name:
            result["field_name"] = field_name
            result["field_entries"] = field_entries
        else:
            agents = {}
            for entry in entries:
                agent = entry.get("agent", "unknown")
                if agent not in agents:
                    agents[agent] = {"count": 0, "actions": set()}
                agents[agent]["count"] += 1
                if "action" in entry:
                    agents[agent]["actions"].add(entry["action"])

            for agent in agents:
                agents[agent]["actions"] = list(agents[agent]["actions"])

            result["by_agent"] = agents

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_evidence_from_output(waiver_key: str, field_name: str = None) -> dict:
    """Get evidence from the extraction output files.

    Reads the evidence chain and extraction data from output files.

    Args:
        waiver_key: The waiver identifier
        field_name: Optional - filter to specific field

    Returns:
        dict with evidence data
    """
    output_dir = OUTPUTS_DIR / waiver_key

    if not output_dir.exists():
        return {
            "status": "not_found",
            "waiver_key": waiver_key,
            "message": f"No output directory found at {output_dir}",
        }

    try:
        result = {
            "status": "success",
            "waiver_key": waiver_key,
        }

        overview_file = output_dir / "overview.json"
        if overview_file.exists():
            with open(overview_file) as f:
                overview = json.load(f)

            if field_name:
                fields = overview.get("extracted_fields", overview)
                value = fields.get(field_name)
                evidence = fields.get(f"{field_name}_evidence")
                page = fields.get(f"{field_name}_page")

                result["field_name"] = field_name
                result["value"] = value
                result["evidence"] = evidence
                result["page"] = page
            else:
                result["fields_count"] = len([k for k in overview.get("extracted_fields", overview).keys()
                                              if not k.endswith(("_evidence", "_page"))])
                result["accuracy"] = overview.get("_accuracy") or overview.get("accuracy")
                result["attempts_used"] = overview.get("_attempts_used") or overview.get("attempts_used")

        evidence_file = output_dir / "evidence_chain.json"
        if evidence_file.exists():
            with open(evidence_file) as f:
                evidence_chain = json.load(f)

            if field_name:
                result["evidence_chain"] = evidence_chain.get(field_name)
            else:
                result["has_evidence_chain"] = True
                result["evidence_chain_fields"] = len(evidence_chain)

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_field_extraction_history(waiver_key: str, field_name: str) -> dict:
    """Get complete extraction history for a specific field.

    Combines data from stage storage, logs, and output files to show
    how a field was extracted across all attempts.

    Args:
        waiver_key: The waiver identifier
        field_name: The field to get history for

    Returns:
        dict with complete field history across attempts
    """
    history = {
        "status": "success",
        "waiver_key": waiver_key,
        "field_name": field_name,
        "attempts": {},
    }

    for attempt_num in [1, 2, 3]:
        stage = f"extraction_attempt_{attempt_num}"
        stage_data = load_stage_output(waiver_key, stage)

        if stage_data.get("status") == "success":
            data = stage_data.get("data", {})
            history["attempts"][f"attempt_{attempt_num}"] = {
                "value": data.get(field_name),
                "evidence": data.get(f"{field_name}_evidence"),
                "page": data.get(f"{field_name}_page"),
                "timestamp": stage_data.get("timestamp"),
            }

    for attempt_num in [1, 2, 3]:
        stage = f"critic_attempt_{attempt_num}"
        stage_data = load_stage_output(waiver_key, stage)

        if stage_data.get("status") == "success":
            data = stage_data.get("data", {})

            critic_info = {}
            for bucket in ["verified_acroform", "verified_body", "corrected",
                          "soft_flag", "missing_child", "hallucinated", "human_queue"]:
                bucket_data = data.get(bucket, [])
                for item in bucket_data:
                    if item.get("field") == field_name:
                        critic_info = {
                            "bucket": bucket,
                            "details": item,
                        }
                        break

            if critic_info:
                history["attempts"][f"critic_{attempt_num}"] = critic_info

    log_data = get_evidence_from_logs(waiver_key, field_name)
    if log_data.get("status") == "success":
        history["log_entries"] = log_data.get("field_entries", [])

    output_data = get_evidence_from_output(waiver_key, field_name)
    if output_data.get("status") == "success":
        history["final_output"] = {
            "value": output_data.get("value"),
            "evidence": output_data.get("evidence"),
            "page": output_data.get("page"),
        }

    return history


def explain_field_classification(waiver_key: str, field_name: str) -> dict:
    """Explain why a specific field was classified into its bucket.

    Provides detailed reasoning about why the critic placed this field
    in a particular category (verified, corrected, soft_flag, etc.)

    Args:
        waiver_key: The waiver identifier
        field_name: The field to explain

    Returns:
        dict with classification explanation
    """
    critic_data = None
    critic_stage = None
    for attempt in [3, 2, 1]:
        stage = f"critic_attempt_{attempt}"
        stage_data = load_stage_output(waiver_key, stage)
        if stage_data.get("status") == "success":
            critic_data = stage_data.get("data", {})
            critic_stage = stage
            break

    if not critic_data:
        return {
            "status": "not_found",
            "waiver_key": waiver_key,
            "field_name": field_name,
            "message": "No critic results found. Run extraction first.",
        }

    bucket_definitions = {
        "verified_acroform": {
            "status": "PASSING",
            "meaning": "Extraction matches AcroForm metadata",
            "action": "No action needed - value is correct",
        },
        "verified_body": {
            "status": "PASSING",
            "meaning": "No AcroForm mapping exists, but value appears correct based on document context",
            "action": "No action needed - value accepted",
        },
        "corrected": {
            "status": "AUTO-CORRECTED",
            "meaning": "LLM extraction disagreed with AcroForm. AcroForm is authoritative, so value was auto-corrected",
            "action": "Value already fixed - using AcroForm value",
        },
        "soft_flag": {
            "status": "FAILING - NEEDS REVIEW",
            "meaning": "AcroForm and LLM extraction conflict, but neither is clearly correct. Visual verification needed",
            "action": "Review the PDF visually to determine correct value, then use apply_manual_correction()",
        },
        "missing_child": {
            "status": "FAILING - MISSED EXTRACTION",
            "meaning": "AcroForm metadata shows this field should have a value, but LLM didn't extract it",
            "action": "Re-extract this field or manually add the value from AcroForm",
        },
        "hallucinated": {
            "status": "FAILING - FALSE POSITIVE",
            "meaning": "LLM extracted a value, but AcroForm definitively shows the field should be empty/off",
            "action": "Value should likely be null/0. Verify and correct if needed",
        },
        "human_queue": {
            "status": "FAILING - UNRESOLVABLE",
            "meaning": "Could not be automatically resolved after all attempts. Needs human judgment",
            "action": "Manual review required - check PDF and apply correction",
        },
    }

    found_bucket = None
    field_details = None

    for bucket_name in bucket_definitions.keys():
        bucket_data = critic_data.get(bucket_name, [])
        for item in bucket_data:
            if item.get("field") == field_name:
                found_bucket = bucket_name
                field_details = item
                break
        if found_bucket:
            break

    if not found_bucket:
        return {
            "status": "not_found",
            "waiver_key": waiver_key,
            "field_name": field_name,
            "critic_stage": critic_stage,
            "message": f"Field '{field_name}' not found in any critic bucket. It may not have been extracted.",
        }

    bucket_info = bucket_definitions[found_bucket]

    explanation = {
        "status": "success",
        "waiver_key": waiver_key,
        "field_name": field_name,
        "critic_stage": critic_stage,
        "bucket": found_bucket,
        "classification_status": bucket_info["status"],
        "what_this_means": bucket_info["meaning"],
        "recommended_action": bucket_info["action"],
        "field_details": field_details,
    }

    if found_bucket == "soft_flag":
        explanation["conflict_details"] = {
            "llm_extracted": field_details.get("llm_value"),
            "acroform_says": field_details.get("acroform_value"),
            "conflict_type": field_details.get("conflict_type"),
        }
    elif found_bucket == "corrected":
        explanation["correction_details"] = {
            "llm_extracted": field_details.get("extracted_value"),
            "corrected_to": field_details.get("corrected_value"),
            "acroform_path": field_details.get("acroform_path"),
        }
    elif found_bucket == "missing_child":
        explanation["missing_details"] = {
            "acroform_path": field_details.get("acroform_path"),
            "expected_value": field_details.get("expected_value"),
        }
    elif found_bucket == "hallucinated":
        explanation["hallucination_details"] = {
            "llm_extracted": field_details.get("hallucinated_value"),
            "should_be": "null or 0 (AcroForm says field is empty/off)",
        }

    history = get_field_extraction_history(waiver_key, field_name)
    if history.get("attempts"):
        explanation["extraction_history"] = history.get("attempts")

    return explanation


def get_failing_fields_summary(waiver_key: str) -> dict:
    """Get a summary of all failing fields and why they failed.

    Analyzes the latest critic results to provide a clear summary
    of what needs attention.

    Args:
        waiver_key: The waiver identifier

    Returns:
        dict with failing fields grouped by failure type
    """
    for attempt in [3, 2, 1]:
        stage = f"critic_attempt_{attempt}"
        stage_data = load_stage_output(waiver_key, stage)
        if stage_data.get("status") == "success":
            break
    else:
        return {
            "status": "not_found",
            "waiver_key": waiver_key,
            "message": "No critic results found. Run extraction first.",
        }

    data = stage_data.get("data", {})

    summary = {
        "status": "success",
        "waiver_key": waiver_key,
        "critic_stage": stage,
        "accuracy": data.get("accuracy"),
        "decision": data.get("decision"),
        "failing_fields": {},
    }

    failure_types = {
        "soft_flag": "AcroForm conflict - needs visual verification",
        "missing_child": "Field exists in AcroForm but wasn't extracted",
        "hallucinated": "Value extracted but AcroForm says it should be empty",
        "human_queue": "Could not be resolved automatically",
    }

    for bucket, description in failure_types.items():
        fields = data.get(bucket, [])
        if fields:
            summary["failing_fields"][bucket] = {
                "description": description,
                "count": len(fields),
                "fields": fields,
            }

    total_failing = sum(len(data.get(b, [])) for b in failure_types.keys())
    summary["total_failing_fields"] = total_failing

    return summary
