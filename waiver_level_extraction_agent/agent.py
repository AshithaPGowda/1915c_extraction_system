"""
1915(c) HCBS Waiver Extraction Agent

ADK agent for extracting structured data from Medicaid waiver PDF documents.
Uses the true ADK sub-agent hierarchy (adk_agents/) - a single orchestrator
delegating to specialist sub-agents (overview, critic, reextraction,
normalisation) via real ADK transfer, not a flat tool-call bypass.

For deterministic batch/unattended processing, use the standalone CLI
scripts instead of this agent - they call the same underlying pipeline
functions directly, no LLM orchestration involved:
    python run_pipeline.py --waiver <KEY>
    python run_services_pipeline.py --waiver <KEY>

Usage:
    # Web UI (interactive)
    cd 1915c_extraction_agent
    adk web waiver_level_extraction_agent

    # CLI (batch)
    adk run waiver_level_extraction_agent
"""

import os
import sys
from pathlib import Path

# Ensure package is importable
AGENT_PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_PKG_DIR.parent))

from .adk_agents import root_agent
from .storage.csv_merge import ensure_results_csvs
from .shared_libraries.config_check import check_on_import

check_on_import()  # adk: stop with a clear message; other imports: warning only
ensure_results_csvs()  # results CSVs (header only) exist from the moment the app boots


def get_project_id():
    return os.getenv("PROJECT_ID")


def get_location():
    return os.getenv("LOCATION", "us-central1")
