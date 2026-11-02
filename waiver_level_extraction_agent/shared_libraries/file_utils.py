"""Shared file utilities for the waiver extraction pipeline."""

import os
import json
import re
from datetime import datetime


from .paths import OUTPUTS_DIR, LOGS_DIR

def normalize_waiver_key(raw: str) -> str:
    """
    Convert waiver number to normalized lookup key.
    'AL.0407.R04.00' -> 'AL0407R0400'
    'AL0407R0400.pdf' -> 'AL0407R0400'
    """
    # Strip file extension if present
    raw = os.path.splitext(str(raw))[0]
    # Remove dots, spaces, uppercase
    return re.sub(r'[.\s]', '', raw).upper().strip()


def waiver_key_from_filename(filename: str) -> str:
    """Extract normalized waiver key from a PDF filename."""
    basename = os.path.basename(filename)
    return normalize_waiver_key(basename)


def get_output_dir(waiver_key: str) -> str:
    """Return the output directory for a given waiver, creating it if needed."""
    path = os.path.join(str(OUTPUTS_DIR), waiver_key)
    os.makedirs(path, exist_ok=True)
    return path


def get_log_path(waiver_key: str) -> str:
    """Return the JSONL log file path for a given waiver."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.join(str(LOGS_DIR), f"{waiver_key}.jsonl")


def write_json_output(waiver_key: str, filename: str, data: dict) -> str:
    """Write a JSON output file for a waiver. Returns the file path."""
    output_dir = get_output_dir(waiver_key)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def log_event(waiver_key: str, event: dict) -> None:
    """Append a structured log event to the waiver's JSONL log file.

    Also echoes a concise one-line summary to the console so per-service
    extraction and critic/re-extraction detail is visible live when running
    manually. Set SERVICES_LOG_QUIET=1 to suppress the console echo.
    """
    event["timestamp"] = datetime.utcnow().isoformat()
    event["waiver_key"] = waiver_key
    log_path = get_log_path(waiver_key)
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")

    if os.getenv("SERVICES_LOG_QUIET"):
        return
    try:
        ts = event["timestamp"][11:19]  # HH:MM:SS
        agent = event.get("agent", "")
        action = event.get("action", "")
        parts = [f"    [{ts}]"]
        if agent:
            parts.append(agent)
        if action:
            parts.append(action)
        # Optional context fields, shown only when present.
        for key in ("service_name", "accuracy", "status", "error", "message", "reason"):
            val = event.get(key)
            if val not in (None, ""):
                parts.append(f"{key}={val}")
        print(" · ".join(parts), flush=True)
    except Exception:
        # Console echo must never break the pipeline.
        pass


def load_layout(name: str) -> dict:
    """
    Load a layout schema file from the layouts directory.

    Args:
        name: Layout name without extension (e.g., "layout_waiver_overview")

    Returns:
        Dict containing the layout schema
    """
    layouts_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "layouts"
    )
    path = os.path.join(layouts_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layout file not found: {path}")
    with open(path) as f:
        return json.load(f)
