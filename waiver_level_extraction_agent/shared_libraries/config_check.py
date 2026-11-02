"""Fail early, with a plain message, when the Google Cloud settings are wrong.

The pipeline scripts read PROJECT_ID and the adk web app reads
GOOGLE_CLOUD_PROJECT; if one is missing it is filled from the other, and if
both are set to different projects (or still hold the .env.example
placeholder) we stop instead of letting the first model call fail with a 403.
"""

import os
from pathlib import Path

PLACEHOLDER = "your-project-id"


def check_cloud_config() -> None:
    """Raise RuntimeError with a fix-it message if the project settings are unusable."""
    try:  # the .env lives in waiver_level_extraction_agent/; variables already set win
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    except ImportError:
        pass
    project = os.getenv("PROJECT_ID", "").strip()
    gcp = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if project and not gcp:
        os.environ["GOOGLE_CLOUD_PROJECT"] = gcp = project
    elif gcp and not project:
        os.environ["PROJECT_ID"] = project = gcp

    if not project or project == PLACEHOLDER:
        raise RuntimeError(
            "Google Cloud project is not set. Copy waiver_level_extraction_agent/"
            ".env.example to .env and set PROJECT_ID and GOOGLE_CLOUD_PROJECT to "
            "your project ID (see the setup guide)."
        )
    if project != gcp:
        raise RuntimeError(
            f"PROJECT_ID ({project}) and GOOGLE_CLOUD_PROJECT ({gcp}) differ. "
            "Set both to the same Google Cloud project ID in .env (or unset one "
            "of them in your shell)."
        )


def check_on_import() -> None:
    """Called when an agent package loads. Under `adk web` / `adk run` a bad
    config stops the load with the fix-it message; any other import (tests,
    reading CSVs, analytics) only gets a warning on stderr."""
    import sys
    try:
        check_cloud_config()
    except RuntimeError as e:
        if Path(sys.argv[0]).name == "adk":
            raise
        print(f"WARNING: {e}", file=sys.stderr)
