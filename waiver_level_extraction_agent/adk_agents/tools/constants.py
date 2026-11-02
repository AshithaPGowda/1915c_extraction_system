"""
Shared constants and paths for ADK tools.
"""

from pathlib import Path

# Directory paths
from ...shared_libraries.paths import (
    PROJECT_ROOT, DATA_ROOT as DATA_DIR, OUTPUTS_DIR as OUTPUT_DIR, PDF_DIR,
)
AGENT_PKG_DIR = Path(__file__).resolve().parent.parent.parent
