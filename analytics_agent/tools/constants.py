"""
Shared constants and paths for the analytics agent's tools.
"""

from pathlib import Path

# Directory paths
AGENT_PKG_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_PKG_DIR.parent

# Extraction event logs live locally under this agent now (relocated from
# waiver_level_extraction_agent/data/logs/ - see shared_libraries/file_utils.py's
# get_log_path(), which every extraction run writes through).
from waiver_level_extraction_agent.shared_libraries.paths import (
    LOGS_DIR, DATA_ROOT, OUTPUTS_DIR, PDF_DIR, VERIFIED_WAIVERS_CSV,
)

# Everything else the analytics tools read (outputs, human_overrides.jsonl,
# the SQLite DB, the CSV export) still lives under waiver_level_extraction_agent/data/ -
# that's where the extraction pipeline writes it, and moving it wasn't part
# of this split.
WAIVER_EXTRACTION_DATA_DIR = DATA_ROOT
DATA_DIR = DATA_ROOT
OUTPUT_DIR = OUTPUTS_DIR
