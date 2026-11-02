"""
CSV export tools for bulk data export.
"""

import subprocess
import sys

from .constants import PROJECT_ROOT
from ...shared_libraries.paths import SCRIPTS_DIR, WAIVER_LEVEL_CSV


def export_all_to_csv(force: bool = False) -> dict:
    """Export all processed waivers to the global CSV file.

    This is a bulk re-export that syncs all overview.json files to CSV.
    Individual waivers are already auto-appended when processed.

    Args:
        force: If True, overwrite existing entries in CSV

    Returns:
        dict with export results
    """
    try:
        cmd = [sys.executable, str(SCRIPTS_DIR / "export_to_csv.py")]
        if force:
            cmd.append("--force")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )

        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "csv_path": str(WAIVER_LEVEL_CSV),
            "force": force,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# get_csv_status() moved to
# adk_agents/tools/analytics/discovery_and_evidence_tools.py - it's a
# read-only status check (analytics), not an export action.
