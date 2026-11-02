"""
Single source of truth for every file-system location the project uses.

Layout (top level of the repo is only: data/, waiver_level_extraction_agent/,
services_extraction_agent/, analytics_agent/ plus the run_*.py entry points):

    data/
      pdfs_267_latest/        source PDFs
      verified/               VERIFIED master CSVs - never written by any pipeline
      outputs/                new extraction outputs: per-waiver folders, waivers.db,
                              waiver_level_extraction.csv, service_level_extraction.csv
      stages/  chroma/  logs/ ADK stage checkpoints, vector store, per-waiver event logs
      human_overrides.jsonl, services_human_overrides.jsonl, known_bad_sources.txt
      archive/                old extractions, scripts, tests, docs, other agents

Environment overrides (relative values resolve against the repo root, not the
current working directory): PDF_DIR, OUTPUT_DIR, LOG_DIR, WAIVER_LEVEL_CSV,
SERVICE_LEVEL_CSV. Agent-internal config (layouts/) stays inside the agent.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


PDF_DIR = _env_path("PDF_DIR", DATA_ROOT / "pdfs_267_latest")
OUTPUTS_DIR = _env_path("OUTPUT_DIR", DATA_ROOT / "outputs")
LOGS_DIR = _env_path("LOG_DIR", DATA_ROOT / "logs")
STAGES_DIR = DATA_ROOT / "stages"
CHROMA_DIR = DATA_ROOT / "chroma"
DB_PATH = OUTPUTS_DIR / "waivers.db"

HUMAN_OVERRIDES_PATH = DATA_ROOT / "human_overrides.jsonl"
SERVICES_HUMAN_OVERRIDES_PATH = DATA_ROOT / "services_human_overrides.jsonl"
KNOWN_BAD_SOURCES_PATH = DATA_ROOT / "known_bad_sources.txt"
CANONICAL_PDFS_PATH = DATA_ROOT / "canonical_pdfs.json"

# Verified masters: read-only references for comparison.
VERIFIED_DIR = DATA_ROOT / "verified"
VERIFIED_WAIVERS_CSV = VERIFIED_DIR / "all_waivers.csv"
VERIFIED_SERVICES_CSV = VERIFIED_DIR / "all_services.csv"

# New extraction results go here by default; point these elsewhere to write a
# different CSV (env var, or the optional output_csv argument of the tools).
WAIVER_LEVEL_CSV = _env_path("WAIVER_LEVEL_CSV", OUTPUTS_DIR / "waiver_level_extraction.csv")
SERVICE_LEVEL_CSV = _env_path("SERVICE_LEVEL_CSV", OUTPUTS_DIR / "service_level_extraction.csv")
CUSTOM_VALUES_CSV = _env_path("CUSTOM_VALUES_CSV", OUTPUTS_DIR / "custom_extractions.csv")  # user-defined values
CUSTOM_SERVICE_VALUES_CSV = _env_path("CUSTOM_SERVICE_VALUES_CSV", OUTPUTS_DIR / "custom_service_extractions.csv")  # user-defined per-service values
CSV_BACKUP_DIR = DATA_ROOT / "csv_backups"

ARCHIVE_DIR = DATA_ROOT / "archive"
SCRIPTS_DIR = ARCHIVE_DIR / "scripts"
