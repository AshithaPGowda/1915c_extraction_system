"""
Pipeline Control Tools for ADK Orchestrator.

Provides tools to:
1. Run pipeline from a specific stage
2. Apply batch corrections to multiple fields

The evidence/history lookup tools that used to live here (get_evidence_from_logs,
get_evidence_from_output, get_field_extraction_history, explain_field_classification,
get_failing_fields_summary) moved to
adk_agents/tools/analytics/discovery_and_evidence_tools.py - they're read-only
analytics, not pipeline-critical, so they belong with the analytics agent's
tools instead of the orchestrator's.
"""

from .stage_storage_tools import (
    clear_stage,
    load_stage_output,
    VALID_STAGES,
)


def run_from_stage(waiver_key: str, stage: str) -> dict:
    """Run the extraction pipeline from a specific stage.

    This clears the specified stage and all subsequent stages, then
    returns instructions for which sub-agent to invoke next.

    Args:
        waiver_key: The waiver identifier (e.g., "NV2426R0001")
        stage: Stage to restart from. Valid stages:
            - checkbox_resolution
            - extraction_attempt_1
            - critic_attempt_1
            - extraction_attempt_2
            - critic_attempt_2
            - extraction_attempt_3
            - critic_attempt_3
            - normalization

    Returns:
        dict with:
            - cleared_stages: stages that were cleared
            - next_agent: which sub-agent to call
            - prerequisites: what data is available from previous stages
            - instructions: how to proceed
    """
    if stage not in VALID_STAGES:
        return {
            "status": "error",
            "error": f"Invalid stage '{stage}'. Valid stages: {VALID_STAGES}",
        }

    # Clear the stage and all subsequent
    clear_result = clear_stage(waiver_key, stage, clear_subsequent=True)

    if clear_result.get("status") != "success":
        return clear_result

    # Determine which agent to call and what prerequisites exist
    stage_idx = VALID_STAGES.index(stage)

    # Map stages to agents
    stage_to_agent = {
        "checkbox_resolution": "overview_extraction_agent",
        "extraction_attempt_1": "overview_extraction_agent",
        "critic_attempt_1": "critic_agent",
        "extraction_attempt_2": "reextraction_agent",
        "critic_attempt_2": "critic_agent",
        "extraction_attempt_3": "reextraction_agent",
        "critic_attempt_3": "critic_agent",
        "normalization": "normalisation_agent",
    }

    # Determine prerequisites (stages before this one)
    prerequisites = VALID_STAGES[:stage_idx]
    available_data = {}

    for prereq in prerequisites:
        prereq_data = load_stage_output(waiver_key, prereq)
        if prereq_data.get("status") == "success":
            available_data[prereq] = "available"
        else:
            available_data[prereq] = "missing"

    # Build instructions based on stage
    instructions = _get_stage_instructions(stage, waiver_key)

    return {
        "status": "success",
        "waiver_key": waiver_key,
        "restart_from": stage,
        "cleared_stages": clear_result.get("cleared_stages", []),
        "next_agent": stage_to_agent.get(stage),
        "prerequisites": available_data,
        "instructions": instructions,
    }


def _get_stage_instructions(stage: str, waiver_key: str) -> str:
    """Get detailed instructions for running a specific stage."""
    instructions = {
        "checkbox_resolution": f"""
1. Use get_pdf_path("{waiver_key}") to get PDF location
2. Use resolve_checkboxes(pdf_path, "{waiver_key}") to extract checkbox values
3. Use save_stage_output("{waiver_key}", "checkbox_resolution", checkbox_data)
4. Then proceed to extraction_attempt_1
""",
        "extraction_attempt_1": f"""
1. Use load_stage_output("{waiver_key}", "checkbox_resolution") if available
2. Use get_pdf_path("{waiver_key}") to get PDF location
3. Use get_extraction_schema() to get field definitions
4. Extract all 163 fields (from the layout file) from the PDF
5. Use save_stage_output("{waiver_key}", "extraction_attempt_1", extracted_data)
6. Use save_page_images_cache("{waiver_key}", pdf_path) to cache images for later
""",
        "critic_attempt_1": f"""
1. Use load_stage_output("{waiver_key}", "extraction_attempt_1") to get extraction
2. Use get_pdf_path("{waiver_key}") to get PDF location
3. Verify each field against AcroForm using verify_field_against_acroform()
4. Categorize fields into buckets (verified, corrected, soft_flag, etc.)
5. Use save_stage_output("{waiver_key}", "critic_attempt_1", critique_results)
""",
        "extraction_attempt_2": f"""
1. Use load_stage_output("{waiver_key}", "critic_attempt_1") to get failing fields
2. Use load_stage_output("{waiver_key}", "extraction_attempt_1") to get base extraction
3. Use load_page_images_cache("{waiver_key}", [pages]) for relevant pages
4. Re-extract only the failing fields
5. Merge with passing fields from attempt 1
6. Use save_stage_output("{waiver_key}", "extraction_attempt_2", merged_data)
""",
        "critic_attempt_2": f"""
1. Use load_stage_output("{waiver_key}", "extraction_attempt_2") to get extraction
2. Verify the re-extracted fields
3. Use save_stage_output("{waiver_key}", "critic_attempt_2", critique_results)
""",
        "extraction_attempt_3": f"""
1. Use load_stage_output("{waiver_key}", "critic_attempt_2") to get still-failing fields
2. Use load_page_images_cache("{waiver_key}", [pages]) for image-based OCR
3. Focus on visual checkbox inspection for remaining failures
4. Use save_stage_output("{waiver_key}", "extraction_attempt_3", merged_data)
""",
        "critic_attempt_3": f"""
1. Use load_stage_output("{waiver_key}", "extraction_attempt_3") to get extraction
2. Final verification pass
3. Use save_stage_output("{waiver_key}", "critic_attempt_3", critique_results)
""",
        "normalization": f"""
1. Load the final extraction (attempt_3, or attempt_2, or attempt_1)
2. Load the final critique for verification metadata
3. Normalize all values to consistent formats
4. Build evidence chain
5. Use save_stage_output("{waiver_key}", "normalization", normalized_data)
6. Use save_extraction_output("{waiver_key}", normalized_data) for final output
""",
    }
    return instructions.get(stage, "No specific instructions for this stage.")


def apply_batch_corrections(waiver_key: str, corrections: list[dict]) -> dict:
    """Apply multiple field corrections at once.

    Args:
        waiver_key: The waiver identifier
        corrections: List of correction dicts, each with:
            - field_name: The field to correct
            - new_value: The corrected value
            - reason: Why this correction is being made

    Returns:
        dict with correction results
    """
    from .correction_tools import apply_manual_correction

    results = []
    successful = 0
    failed = 0

    for correction in corrections:
        field_name = correction.get("field_name")
        new_value = correction.get("new_value")
        reason = correction.get("reason", "batch_correction")

        if not field_name:
            results.append({
                "field_name": field_name,
                "status": "error",
                "error": "Missing field_name",
            })
            failed += 1
            continue

        result = apply_manual_correction(waiver_key, field_name, new_value, reason)

        results.append({
            "field_name": field_name,
            "status": result.get("status"),
            "new_value": new_value,
            "error": result.get("error"),
        })

        if result.get("status") == "success":
            successful += 1
        else:
            failed += 1

    return {
        "status": "success" if failed == 0 else "partial",
        "waiver_key": waiver_key,
        "total_corrections": len(corrections),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


