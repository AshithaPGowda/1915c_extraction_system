"""
ADK Tools Package.

Re-exports all tools from submodules for backwards compatibility.
Import from here: `from .tools import get_pdf_path, extract_all_fields, ...`

Layout:
- Flat files in this package: pipeline-critical tools (PDF/AcroForm access,
  extraction, stage checkpointing, corrections) - what the orchestrator and
  the 4 extraction sub-agents actually need to do the work.
- analytics/: read-only analytics tools (DB stats, error analysis, evidence
  lookups, discovery/batch-status) - what the separate analytics_agent uses.
  None of these touch the extraction pipeline itself.

The former processing_tools.py (process_waiver, process_batch,
reprocess_waiver) and the process_waiver_list/resume_incomplete_waivers
half of batch_processing_tools.py were deleted - they called
core.pipeline.run_overview_extraction directly, bypassing the sub-agent
hierarchy entirely. That bypass caused the orchestrator to sometimes skip
delegation and sometimes not, with a self-contradictory prompt telling it
to do both. There is now exactly one way to process a waiver in ADK mode:
delegate to the sub-agents.
"""

from .pdf_tools import (
    get_pdf_path,
    list_pdfs_in_folder,
    get_extraction_schema,
    load_pdf_bytes,
)

from .acroform_tools import (
    get_acroform_tree,
    get_acroform_value,
    verify_field_against_acroform,
    get_all_acroform_mappings,
    verify_all_fields,
)

from .checkbox_tools import (
    resolve_checkboxes,
)

from .page_image_tools import (
    get_page_images,
    get_field_page_hint,
)

from .output_tools import (
    save_extraction_output,
    log_agent_event,
)

from .correction_tools import (
    apply_manual_correction,
    get_field_value,
)

from .csv_tools import (
    export_all_to_csv,
)

from .stage_storage_tools import (
    save_stage_output,
    load_stage_output,
    get_pipeline_status,
    clear_stage,
    save_page_images_cache,
    load_page_images_cache,
    get_cached_image_pages,
    # AcroForm caching
    save_acroform_cache,
    load_acroform_cache,
    has_acroform_cache,
    extract_and_cache_acroform,
)

from .pipeline_control_tools import (
    run_from_stage,
    apply_batch_corrections,
)

from .extraction_tools import (
    extract_all_fields,
    extract_specific_fields,
)

from .state_pipeline_tools import (
    prepare_waiver,
    extract_overview_fields,
    critique_overview,
    reextract_failing_fields,
    reextract_specific_fields,
    finalize_waiver,
    resume_from_stage,
    write_overview_to_csv,
    review_extraction,
    apply_review,
    extract_custom_value,
    save_custom_value,
)

# Analytics tools used to live under here (analytics/ subpackage) but have
# moved entirely to the standalone analytics_agent/ app at the project
# root - they're not part of this package anymore.

# All exports for `from .tools import *`
__all__ = [
    # PDF tools
    "get_pdf_path",
    "list_pdfs_in_folder",
    "get_extraction_schema",
    "load_pdf_bytes",
    # AcroForm tools
    "get_acroform_tree",
    "get_acroform_value",
    "verify_field_against_acroform",
    "get_all_acroform_mappings",
    "verify_all_fields",
    # Checkbox tools
    "resolve_checkboxes",
    # Page image tools
    "get_page_images",
    "get_field_page_hint",
    # Output tools
    "save_extraction_output",
    "log_agent_event",
    # Correction tools
    "apply_manual_correction",
    "get_field_value",
    # CSV export
    "export_all_to_csv",
    # Stage storage tools
    "save_stage_output",
    "load_stage_output",
    "get_pipeline_status",
    "clear_stage",
    "save_page_images_cache",
    "load_page_images_cache",
    "get_cached_image_pages",
    # AcroForm caching
    "save_acroform_cache",
    "load_acroform_cache",
    "has_acroform_cache",
    "extract_and_cache_acroform",
    # Pipeline control tools
    "run_from_stage",
    "apply_batch_corrections",
    # Extraction tools
    "extract_all_fields",
    "extract_specific_fields",
    # State-driven pipeline tools (ADK session state)
    "prepare_waiver",
    "extract_overview_fields",
    "critique_overview",
    "reextract_failing_fields",
    "reextract_specific_fields",
    "finalize_waiver",
    "resume_from_stage",
    "write_overview_to_csv",
    "review_extraction",
    "apply_review",
    "extract_custom_value",
    "save_custom_value",
]
