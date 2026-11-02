"""
ADK Tools - Re-exports from tools package for backwards compatibility.

All tools have been refactored into smaller modules under ./tools/
Import from here or directly from the submodules.

Analytics tools (processing stats, error analysis, evidence lookups,
discovery/batch-status) moved entirely to the standalone analytics_agent/
app at the project root - they are not re-exported here anymore.
"""

# Re-export everything from the tools package
from .tools import (
    # PDF tools
    get_pdf_path,
    get_extraction_schema,
    load_pdf_bytes,
    # AcroForm tools
    get_acroform_tree,
    get_acroform_value,
    verify_field_against_acroform,
    get_all_acroform_mappings,
    verify_all_fields,
    # Checkbox tools
    resolve_checkboxes,
    # Page image tools
    get_page_images,
    get_field_page_hint,
    # Output tools
    save_extraction_output,
    log_agent_event,
    # Correction tools
    apply_manual_correction,
    get_field_value,
    # CSV tools
    export_all_to_csv,
    # Stage storage tools
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
    # Pipeline control tools
    run_from_stage,
    apply_batch_corrections,
    # Extraction tools
    extract_all_fields,
    extract_specific_fields,
)

__all__ = [
    # PDF tools
    "get_pdf_path",
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
    # CSV tools
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
]
