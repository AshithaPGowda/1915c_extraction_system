"""
Database status and discovery tools for the orchestrator agent.

These tools give the agent access to extraction history and
institutional knowledge stored in SQLite and human_overrides.jsonl.
"""

import re
import json

from .constants import DATA_DIR, PDF_DIR


def get_processing_stats() -> dict:
    """Get overall processing statistics from the database."""
    from waiver_level_extraction_agent.storage.sqlite_store import list_extractions

    try:
        extractions = list_extractions(extraction_type="overview")

        if not extractions:
            return {
                "status": "success",
                "total_processed": 0,
                "message": "No waivers processed yet",
            }

        total = len(extractions)
        accuracies = [e["accuracy"] for e in extractions if e["accuracy"] is not None]
        statuses = {}
        for e in extractions:
            status = e.get("final_status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "status": "success",
            "total_processed": total,
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "min_accuracy": min(accuracies) if accuracies else None,
            "max_accuracy": max(accuracies) if accuracies else None,
            "status_breakdown": statuses,
            "processed_waiver_keys": [e["waiver_key"] for e in extractions],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_processed_waivers(limit: int = 100) -> dict:
    """Get list of already processed waivers from database."""
    from waiver_level_extraction_agent.storage.sqlite_store import list_extractions

    try:
        extractions = list_extractions(extraction_type="overview")

        processed = []
        for e in extractions[:limit]:
            processed.append({
                "waiver_key": e["waiver_key"],
                "accuracy": e.get("accuracy"),
                "status": e.get("final_status"),
                "attempts": e.get("attempts"),
                "timestamp": e.get("timestamp"),
            })

        return {
            "status": "success",
            "count": len(processed),
            "total_in_db": len(extractions),
            "waivers": processed,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_unprocessed_waivers(limit: int = 50) -> dict:
    """Get waivers that exist as PDFs but haven't been processed yet.

    Filters out Appendix K and approval letters (separate pipeline).
    """
    from waiver_level_extraction_agent.storage.sqlite_store import list_extractions
    from waiver_level_extraction_agent.shared_libraries.file_utils import waiver_key_from_filename

    MAIN_WAIVER_PATTERN = re.compile(r'^[A-Z]{2}\d+R\d+$')
    EXCLUDE_PATTERNS = [
        'APPENDIX-K', 'APPENDIX_K', 'APP-K',
        'APPVL', 'APPVL-LTR', '-LTR-', '_LTR_',
    ]

    try:
        all_pdfs = []
        excluded_pdfs = []

        if PDF_DIR.exists():
            for pdf_file in PDF_DIR.glob("*.pdf"):
                filename = pdf_file.stem.upper()
                waiver_key = waiver_key_from_filename(pdf_file.name)

                is_excluded = any(pat in filename for pat in EXCLUDE_PATTERNS)

                if is_excluded:
                    excluded_pdfs.append(waiver_key or pdf_file.stem)
                elif waiver_key and MAIN_WAIVER_PATTERN.match(waiver_key):
                    all_pdfs.append(waiver_key)

        extractions = list_extractions(extraction_type="overview")
        processed_keys = {e["waiver_key"] for e in extractions}

        unprocessed = [k for k in all_pdfs if k not in processed_keys]

        return {
            "status": "success",
            "total_main_waiver_pdfs": len(all_pdfs),
            "total_processed": len(processed_keys),
            "total_unprocessed": len(unprocessed),
            "excluded_appendix_k": len(excluded_pdfs),
            "unprocessed_waivers": sorted(unprocessed)[:limit],
            "note": "Appendix K and approval letters are filtered out",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_waiver_history(waiver_key: str) -> dict:
    """Get full extraction history for a specific waiver."""
    from waiver_level_extraction_agent.storage.sqlite_store import get_extraction

    try:
        extraction = get_extraction(waiver_key, "overview")

        if not extraction:
            return {
                "status": "not_found",
                "waiver_key": waiver_key,
                "message": "No extraction record found",
            }

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "extraction_type": extraction["extraction_type"],
            "accuracy": extraction["accuracy"],
            "attempts": extraction["attempts"],
            "final_status": extraction["final_status"],
            "timestamp": extraction["timestamp"],
            "norm_errors": extraction["norm_errors"],
            "unknown_fields": extraction["unknown_fields"],
            "extracted_data": extraction["extracted_data"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_human_overrides(waiver_key: str = None, limit: int = 50) -> dict:
    """Get human override/correction records."""
    try:
        overrides_path = DATA_DIR / "human_overrides.jsonl"

        if not overrides_path.exists():
            return {"status": "success", "count": 0, "overrides": []}

        overrides = []
        with open(overrides_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if waiver_key is None or record.get("waiver_number", "").replace(".", "") == waiver_key:
                        overrides.append(record)
                        if len(overrides) >= limit:
                            break

        conflict_types = {}
        for o in overrides:
            ct = o.get("conflict_type", "unknown")
            conflict_types[ct] = conflict_types.get(ct, 0) + 1

        return {
            "status": "success",
            "count": len(overrides),
            "conflict_type_summary": conflict_types,
            "overrides": overrides,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_waivers_needing_review(limit: int = 50) -> dict:
    """Get waivers flagged for human review."""
    from waiver_level_extraction_agent.storage.sqlite_store import list_extractions

    try:
        extractions = list_extractions(extraction_type="overview")

        needs_review = []
        for e in extractions:
            status = e.get("final_status", "")
            accuracy = e.get("accuracy", 0)

            if status in ("human_review_required", "validation_failed") or (accuracy and accuracy < 0.85):
                needs_review.append({
                    "waiver_key": e["waiver_key"],
                    "accuracy": accuracy,
                    "status": status,
                    "reason": "low_accuracy" if accuracy and accuracy < 0.85 else status,
                })

        return {"status": "success", "count": len(needs_review), "waivers": needs_review[:limit]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def search_extractions(field_name: str, value: str = None) -> dict:
    """Search across all extractions for a specific field value."""
    from waiver_level_extraction_agent.storage.sqlite_store import list_extractions, get_extraction

    try:
        extractions = list_extractions(extraction_type="overview")

        matches = []
        for e in extractions:
            full = get_extraction(e["waiver_key"], "overview")
            if full and full.get("extracted_data"):
                data = full["extracted_data"]
                if field_name in data:
                    field_value = data[field_name]
                    if value is None or str(field_value) == str(value):
                        matches.append({
                            "waiver_key": e["waiver_key"],
                            "field": field_name,
                            "value": field_value,
                        })

        return {
            "status": "success",
            "field": field_name,
            "search_value": value,
            "match_count": len(matches),
            "matches": matches[:50],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
