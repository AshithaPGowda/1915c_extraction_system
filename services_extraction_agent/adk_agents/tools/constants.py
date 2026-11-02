"""
Shared constants and paths for the services extraction agent's tools.
"""

from pathlib import Path

from waiver_level_extraction_agent.shared_libraries.paths import PROJECT_ROOT, PDF_DIR
AGENT_PKG_DIR = Path(__file__).resolve().parent.parent.parent

# Services extraction reads/writes through waiver_level_extraction_agent's
# storage and shared_libraries modules (SQLite, checkpoint files, event
# logs) - nothing is duplicated here, this app is a thin ADK wrapper around
# waiver_level_extraction_agent/core/services_pipeline.py and
# sub_agents/services_*.py.
