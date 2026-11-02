"""
Services Extraction Agent - standalone ADK app.

Extracts service-level data (one row per service, parent/child hierarchy)
from 1915(c) HCBS Medicaid waiver PDFs, via a real ADK sub-agent hierarchy
mirroring waiver_level_extraction_agent/core/services_pipeline.py exactly.

This is a genuinely separate app from waiver_level_extraction_agent/ (its
own root_agent, own folder, own "app" entry in `adk web`'s app selector) -
overview/waiver-level fields and service-level data are two different
extraction targets with two different pipelines, so they get two different
agents. Analytics (processing stats, error analysis) is a third, separate
app: analytics_agent/.

For deterministic batch/unattended processing, use the standalone CLI
script instead - it calls the same underlying pipeline function directly,
no LLM orchestration involved:
    python run_services_pipeline.py --waiver <KEY>

Usage:
    cd 1915c_extraction_agent
    adk web .          # then pick "services_extraction_agent" from the app selector
    adk web services_extraction_agent
    adk run services_extraction_agent
"""

import sys
from pathlib import Path

# Ensure the project root is importable so
# `waiver_level_extraction_agent.*` absolute imports (used by this agent's
# tools to call the real services pipeline functions) resolve correctly.
AGENT_PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_PKG_DIR.parent))

from .adk_agents import root_agent
from waiver_level_extraction_agent.storage.csv_merge import ensure_results_csvs
from waiver_level_extraction_agent.shared_libraries.config_check import check_on_import

check_on_import()  # adk: stop with a clear message; other imports: warning only
ensure_results_csvs()  # results CSVs (header only) exist from the moment the app boots
