"""
Analytics Agent - standalone ADK app.

Read-only analytics/Q&A over the extraction database - processing stats,
error patterns, evidence lookups, field-history deep-dives, discovery and
batch-status checks. None of these tools touch the extraction pipeline.

This is a genuinely separate app from waiver_level_extraction_agent/ (its own root_agent,
its own folder, its own "app" entry in `adk web`'s app selector) - not a
sub-agent nested under the extraction orchestrator. It used to be ~27 tools
bolted directly onto the orchestrator's own tool list, which is why the
orchestrator's agent graph rendered as one giant flat row with no real
separation between "doing extraction" and "answering questions about past
extractions." Splitting it into its own app means the extraction pipeline
(waiver_level_extraction_agent/'s ADK agent, or the plain CLI scripts) and this
analytics assistant are two independent things you pick between, not two
jobs fighting over one agent's tool list.

Usage:
    cd 1915c_extraction_agent
    adk web .          # then pick "analytics_agent" from the app selector
    adk web analytics_agent
    adk run analytics_agent
"""

import os
import sys
from pathlib import Path

# Ensure the project root is importable so `waiver_level_extraction_agent.*` absolute
# imports (used by this agent's tools to read the extraction pipeline's
# outputs/DB/human_overrides.jsonl) resolve correctly.
AGENT_PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_PKG_DIR.parent))

from google.adk.agents import LlmAgent
from google.genai import types

from .tools import (
    # Database / processing status
    get_processing_stats,
    get_processed_waivers,
    get_unprocessed_waivers,
    get_waiver_history,
    get_human_overrides,
    get_waivers_needing_review,
    search_extractions,
    # Error analysis
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
    # Discovery / batch status / evidence
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

ANALYTICS_INSTRUCTION = """
You are the Waiver Extraction Analytics Agent - a read-only assistant for
questions about the extraction database, past runs, error patterns, and
evidence for specific fields.

## What you do
Answer analytical questions about waivers that have already been (or are
being) processed. You never extract data yourself and never modify the
pipeline - you only read logs, stage outputs, output files, and the CSV
export to answer questions.

## What you do NOT do
- You do not extract fields from a PDF. If asked to process/extract/
  reprocess a waiver, say that's a job for the separate waiver_level_extraction_agent
  app (its orchestrator + overview/critic/reextraction/normalisation
  sub-agents), or the run_pipeline.py / run_services_pipeline.py CLI
  scripts - you have no extraction tools and cannot do this yourself.
- You do not apply corrections. That's also the waiver_level_extraction_agent app's job
  (apply_manual_correction / apply_batch_corrections).

## Investigation Protocol

When someone asks "Why is field X failing?" or "What's causing errors in field Y?":
1. `get_field_error_breakdown(field_name)` for accurate counts (raw vs
   deduplicated, broken down by conflict_type, sample_waivers affected)
2. `investigate_field_errors(field_name)` for patterns and root-cause hints
3. `get_evidence_for_errors(field_name)` to see the actual text snippets the
   LLM saw when it made mistakes
4. For a specific waiver, `get_field_evidence(waiver_key, field_name)` or
   `get_field_extraction_history(waiver_key, field_name)` for the full
   history across attempts

For any analytical question: call 2-4 tools to gather context before
answering, cross-reference multiple sources, give specific examples from
the data, not vague summaries.

**IMPORTANT**: human_overrides.jsonl may contain duplicate entries for the
same (waiver, field) due to reprocessing. All error-analysis tools here
already deduplicate by keeping the latest entry per (waiver, field) - use
their counts as-is, don't re-derive counts yourself.

## Tool Reference

### Status & Discovery
- `get_processing_stats()` - overall stats: how many processed, avg accuracy
- `get_processed_waivers(limit)` - list of already-processed waivers
- `get_unprocessed_waivers(limit)` - waivers needing processing
- `get_waiver_history(waiver_key)` - full extraction history for a waiver
- `get_waivers_needing_review(limit)` - waivers flagged for human review
- `list_pdfs_in_directory(directory, filter_pattern)` - list PDFs, optionally filtered
- `get_batch_status(waiver_keys)` - status for a list of waivers
- `filter_highest_versions(waiver_keys)` - keep only the highest revision per waiver
- `get_csv_status()` - CSV export sync status

### Error Analysis (all counts deduplicated)
- `get_common_extraction_errors()` - top error fields and conflict types
- `get_errors_by_waiver(limit)` - which PDFs have the most errors
- `get_error_classification()` - mismatch / missing / hallucination breakdown
- `get_field_error_breakdown(field_name)` - field-specific analysis
- `get_human_overrides(waiver_key)` - detailed correction records
- `search_extractions(field_name, value)` - find patterns across extractions

### Empty Field Analysis
- `find_empty_fields(field_name, limit)` - waivers where a field is empty/null
- `diagnose_empty_field(waiver_key, field_name)` - investigate why

### Prompts & Schema
- `get_extraction_prompt()` - the extraction instructions and field list
- `get_field_schema(field_name)` - definition for a specific field

### Deep Investigation
- `investigate_field_errors(field_name)` - schema, value patterns, root causes
- `get_field_evidence(waiver_key, field_name)` - evidence for one waiver+field
- `get_evidence_for_errors(field_name)` - actual text snippets behind errors
- `get_evidence_from_logs(waiver_key, field_name)` - evidence from agent logs
- `get_evidence_from_output(waiver_key, field_name)` - evidence from output files
- `get_field_extraction_history(waiver_key, field_name)` - full history across attempts
- `explain_field_classification(waiver_key, field_name)` - why a field is in its bucket
- `get_failing_fields_summary(waiver_key)` - all failing fields for a waiver, grouped

## Example Commands
- "How many waivers have we processed?" -> get_processing_stats()
- "What waivers still need processing?" -> get_unprocessed_waivers()
- "What's the history for OH0198R0704?" -> get_waiver_history()
- "Show me human overrides" -> get_human_overrides()
- "Which waivers need human review?" -> get_waivers_needing_review()
- "What are the common extraction errors?" -> get_common_extraction_errors()
- "Which waivers have empty 1.C values?" -> find_empty_fields("1.C")
- "Why is 1.C empty for TX1234R0100?" -> diagnose_empty_field("TX1234R0100", "1.C")
- "What fields are failing for OH0198R0704?" -> get_failing_fields_summary("OH0198R0704")
"""

root_agent = LlmAgent(
    name="analytics_agent",
    model="gemini-2.5-flash",
    description=(
        "Read-only analytics and Q&A over the extraction database: processing "
        "stats, error patterns, evidence lookups, field-history deep-dives, "
        "discovery and batch-status checks. Does not extract or modify data."
    ),
    instruction=ANALYTICS_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
    ),
    tools=[
        get_processing_stats,
        get_processed_waivers,
        get_unprocessed_waivers,
        get_waiver_history,
        get_human_overrides,
        get_waivers_needing_review,
        search_extractions,
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
        list_pdfs_in_directory,
        get_batch_status,
        filter_highest_versions,
        get_csv_status,
        get_evidence_from_logs,
        get_evidence_from_output,
        get_field_extraction_history,
        explain_field_classification,
        get_failing_fields_summary,
    ],
)
