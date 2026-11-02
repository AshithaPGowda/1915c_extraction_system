"""
Analytics Agent Tools.

Read-only analytics over the extraction database and logs. None of these
touch the extraction pipeline.
"""

from .database_tools import (
    get_processing_stats,
    get_processed_waivers,
    get_unprocessed_waivers,
    get_waiver_history,
    get_human_overrides,
    get_waivers_needing_review,
    search_extractions,
)

from .error_analysis_tools import (
    get_common_extraction_errors,
    get_errors_by_waiver,
    get_error_classification,
    get_field_error_breakdown,
    get_extraction_prompt,
    get_field_schema,
    investigate_field_errors,
    get_field_evidence,
    get_evidence_for_errors,
    find_empty_fields,
    diagnose_empty_field,
)

from .discovery_and_evidence_tools import (
    list_pdfs_in_directory,
    get_batch_status,
    filter_highest_versions,
    get_csv_status,
    get_evidence_from_logs,
    get_evidence_from_output,
    get_field_extraction_history,
    explain_field_classification,
    get_failing_fields_summary,
)

__all__ = [
    "get_processing_stats",
    "get_processed_waivers",
    "get_unprocessed_waivers",
    "get_waiver_history",
    "get_human_overrides",
    "get_waivers_needing_review",
    "search_extractions",
    "get_common_extraction_errors",
    "get_errors_by_waiver",
    "get_error_classification",
    "get_field_error_breakdown",
    "get_extraction_prompt",
    "get_field_schema",
    "investigate_field_errors",
    "get_field_evidence",
    "get_evidence_for_errors",
    "find_empty_fields",
    "diagnose_empty_field",
    "list_pdfs_in_directory",
    "get_batch_status",
    "filter_highest_versions",
    "get_csv_status",
    "get_evidence_from_logs",
    "get_evidence_from_output",
    "get_field_extraction_history",
    "explain_field_classification",
    "get_failing_fields_summary",
]
