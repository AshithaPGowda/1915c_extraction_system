"""
Waiver extraction pipeline runner.
Processes one or more PDF files through the extraction pipeline.
Uses canonical PDF list to select highest version for each waiver.

Features:
- Idempotent: skips already-processed PDFs (check SQLite)
- Max page filter: skips PDFs over threshold and logs them
- Diverse selection: can select PDFs from different states
"""

import os
import json
import glob
import asyncio
from pathlib import Path

# Load .env file only if it exists (prefer system env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # Don't override existing system env vars
except ImportError:
    pass

# Optional: pymupdf for page counting
try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# Path to canonical PDFs JSON
from ..shared_libraries.paths import PROJECT_ROOT, CANONICAL_PDFS_PATH

# Default max pages threshold
DEFAULT_MAX_PAGES = 1000


def load_canonical_pdfs() -> dict | None:
    """
    Load the canonical PDFs mapping.

    Returns:
        Dict mapping base_id → canonical pdf_path, or None if not found
    """
    if not CANONICAL_PDFS_PATH.exists():
        return None
    with open(CANONICAL_PDFS_PATH) as f:
        return json.load(f)


def get_pdf_files(pdf_dir: str, use_canonical: bool = True) -> list[str]:
    """
    Return list of PDF file paths to process.

    If use_canonical=True and canonical_pdfs.json exists, returns only
    the canonical (highest version) PDFs. Otherwise returns all PDFs.

    Args:
        pdf_dir: Directory containing PDFs
        use_canonical: Whether to filter to canonical PDFs only

    Returns:
        Sorted list of PDF file paths
    """
    # Discover all PDFs
    pattern = os.path.join(pdf_dir, "**/*.pdf")
    files = glob.glob(pattern, recursive=True)
    pattern2 = os.path.join(pdf_dir, "*.pdf")
    files += [f for f in glob.glob(pattern2) if f not in files]
    # Also check .PDF
    pattern3 = os.path.join(pdf_dir, "*.PDF")
    files += [f for f in glob.glob(pattern3) if f not in files]

    all_files = sorted(set(files))

    if not use_canonical:
        return all_files

    # Try to load canonical list
    canonical = load_canonical_pdfs()
    if canonical is None:
        print("Note: canonical_pdfs.json not found. Processing all PDFs.")
        print("Run 'python scripts/select_canonical_pdfs.py' to generate it.")
        return all_files

    # Filter to only canonical PDFs
    canonical_paths = set(canonical.values())
    # Also match by basename for relative/absolute path differences
    canonical_basenames = {os.path.basename(p) for p in canonical_paths}

    filtered = []
    for f in all_files:
        if f in canonical_paths or os.path.basename(f) in canonical_basenames:
            filtered.append(f)

    if not filtered:
        print("Warning: No canonical PDFs found in directory. Processing all PDFs.")
        return all_files

    print(f"Using canonical PDF list: {len(filtered)} of {len(all_files)} PDFs selected")
    return sorted(filtered)


def get_page_count(pdf_path: str) -> int | None:
    """
    Get the page count of a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages, or None if unable to determine
    """
    if not HAS_FITZ:
        return None

    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception:
        return None


def filter_by_page_count(
    pdf_files: list[str],
    max_pages: int = DEFAULT_MAX_PAGES,
) -> tuple[list[str], list[dict]]:
    """
    Filter PDF files by page count.

    Args:
        pdf_files: List of PDF file paths
        max_pages: Maximum allowed pages (PDFs over this are skipped)

    Returns:
        Tuple of (filtered_files, skipped_records)
        skipped_records contains dicts with pdf_path, page_count, reason
    """
    if not HAS_FITZ:
        print("Warning: pymupdf not installed, skipping page count filter")
        return pdf_files, []

    filtered = []
    skipped = []

    for pdf_path in pdf_files:
        page_count = get_page_count(pdf_path)

        if page_count is None:
            # Can't determine page count, include it
            filtered.append(pdf_path)
        elif page_count > max_pages:
            skipped.append({
                "pdf_path": pdf_path,
                "page_count": page_count,
                "reason": f"too_large (>{max_pages} pages)",
            })
        else:
            filtered.append(pdf_path)

    return filtered, skipped


def filter_already_processed(
    pdf_files: list[str],
    extraction_type: str = "overview",
) -> tuple[list[str], list[str]]:
    """
    Filter out PDFs that have already been processed (idempotency).

    Args:
        pdf_files: List of PDF file paths
        extraction_type: Type of extraction to check

    Returns:
        Tuple of (unprocessed_files, already_processed_keys)
    """
    from ..shared_libraries.file_utils import waiver_key_from_filename
    from ..storage.sqlite_store import extraction_exists

    unprocessed = []
    already_done = []

    for pdf_path in pdf_files:
        waiver_key = waiver_key_from_filename(pdf_path)
        if extraction_exists(waiver_key, extraction_type):
            already_done.append(waiver_key)
        else:
            unprocessed.append(pdf_path)

    return unprocessed, already_done


def select_diverse_pdfs(
    pdf_files: list[str],
    count: int = 50,
) -> list[str]:
    """
    Select a diverse set of PDFs from different states.

    Distributes selection across states to ensure coverage.

    Args:
        pdf_files: List of PDF file paths
        count: Number of PDFs to select

    Returns:
        List of selected PDF paths
    """
    from collections import defaultdict

    # Group by state (first 2 chars of filename)
    by_state = defaultdict(list)
    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        state = basename[:2].upper()
        by_state[state].append(pdf_path)

    # Sort states by count (ascending) so we take from smaller states first
    states = sorted(by_state.keys(), key=lambda s: len(by_state[s]))

    selected = []
    remaining = count

    # Round-robin selection across states
    while remaining > 0 and any(by_state[s] for s in states):
        for state in states:
            if remaining <= 0:
                break
            if by_state[state]:
                selected.append(by_state[state].pop(0))
                remaining -= 1

    return selected


def log_skipped_pdfs(
    skipped_large: list[dict],
    already_done: list[str],
) -> None:
    """
    Log skipped PDFs to a file for audit purposes.

    Args:
        skipped_large: List of dicts with pdf_path, page_count, reason
        already_done: List of waiver keys already processed
    """
    from datetime import datetime

    from ..shared_libraries.paths import LOGS_DIR
    log_path = LOGS_DIR / "skipped_pdfs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()

    with open(log_path, "a") as f:
        for rec in skipped_large:
            entry = {
                "timestamp": timestamp,
                "pdf_path": rec["pdf_path"],
                "reason": "too_large",
                "page_count": rec["page_count"],
            }
            f.write(json.dumps(entry) + "\n")

        for waiver_key in already_done:
            entry = {
                "timestamp": timestamp,
                "waiver_key": waiver_key,
                "reason": "already_processed",
            }
            f.write(json.dumps(entry) + "\n")


async def process_single_waiver(pdf_path: str) -> dict:
    """
    Run the full extraction pipeline for a single waiver PDF.
    Returns a result dict with status, waiver_key, and output paths.
    """
    from ..shared_libraries.file_utils import (
        waiver_key_from_filename,
        log_event,
    )
    from .pipeline import run_overview_extraction

    waiver_key = waiver_key_from_filename(pdf_path)

    log_event(waiver_key, {
        "agent": "runner",
        "action": "pipeline_start",
        "pdf_path": pdf_path,
    })

    try:
        result = await run_overview_extraction(pdf_path, waiver_key)

        log_event(waiver_key, {
            "agent": "runner",
            "action": "pipeline_complete",
            "status": result.get("status"),
        })

        return result

    except Exception as e:
        log_event(waiver_key, {
            "agent": "runner",
            "action": "pipeline_error",
            "error": str(e),
        })
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "error": str(e),
        }


async def run_batch(
    pdf_dir: str,
    limit: int = None,
    use_canonical: bool = True,
    max_pages: int = DEFAULT_MAX_PAGES,
    skip_existing: bool = True,
    diverse: bool = False,
) -> list[dict]:
    """
    Process PDFs in the given directory.

    Args:
        pdf_dir: Directory containing PDFs
        limit: If set, only process the first N files (useful for testing)
        use_canonical: If True, only process canonical (highest version) PDFs
        max_pages: Skip PDFs with more than this many pages (default: 1000)
        skip_existing: If True, skip already-processed PDFs (idempotent)
        diverse: If True, select PDFs from different states for coverage

    Returns:
        List of result dicts
    """
    pdf_files = get_pdf_files(pdf_dir, use_canonical=use_canonical)

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    # Filter by page count (skip large PDFs)
    pdf_files, skipped_large = filter_by_page_count(pdf_files, max_pages)
    if skipped_large:
        print(f"\nSkipped {len(skipped_large)} PDF(s) exceeding {max_pages} pages:")
        for rec in skipped_large:
            print(f"  - {os.path.basename(rec['pdf_path'])}: {rec['page_count']} pages")

    # Filter already processed (idempotency)
    already_done = []
    if skip_existing:
        pdf_files, already_done = filter_already_processed(pdf_files)
        if already_done:
            print(f"\nSkipped {len(already_done)} already-processed PDF(s)")

    # Log skipped PDFs for audit
    if skipped_large or already_done:
        log_skipped_pdfs(skipped_large, already_done)

    if not pdf_files:
        print("No PDFs remaining after filters")
        return []

    # Select diverse set if requested
    if diverse and limit:
        pdf_files = select_diverse_pdfs(pdf_files, count=limit)
        print(f"\nSelected {len(pdf_files)} diverse PDFs across states")
    elif limit:
        pdf_files = pdf_files[:limit]

    print(f"\nProcessing {len(pdf_files)} PDF(s)")

    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
        result = await process_single_waiver(pdf_path)
        results.append(result)

        status = result.get("status", "unknown")
        attempt1_acc = result.get("attempt1_accuracy", 0)
        final_acc = result.get("final_accuracy", 0)
        reextract = result.get("reextraction_performed", False)

        print(f"  -> Status: {status}")
        print(f"  -> Attempt 1 accuracy: {attempt1_acc:.1%}")
        if reextract:
            print(f"  -> Final accuracy (after re-extraction): {final_acc:.1%}")
            unresolved = result.get("unresolved_fields_count", 0)
            if unresolved:
                print(f"  -> Unresolved fields: {unresolved}")

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table of processing results."""
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "validation_failed"]
    errors = [r for r in results if r.get("status") == "error"]

    print(f"Total processed      : {len(results)}")
    print(f"Success              : {len(success)}")
    print(f"Validation failed    : {len(failed)}")
    print(f"Errors               : {len(errors)}")

    # Accuracy summary
    if results:
        attempt1_accs = [r.get("attempt1_accuracy", 0) for r in results if "attempt1_accuracy" in r]
        final_accs = [r.get("final_accuracy", 0) for r in results if "final_accuracy" in r]

        if attempt1_accs:
            avg_a1 = sum(attempt1_accs) / len(attempt1_accs)
            print(f"\nAverage Attempt 1 accuracy: {avg_a1:.1%}")

        if final_accs:
            avg_final = sum(final_accs) / len(final_accs)
            print(f"Average final accuracy    : {avg_final:.1%}")

        # Re-extraction stats
        reextracted = [r for r in results if r.get("reextraction_performed")]
        if reextracted:
            print(f"\nRe-extraction performed: {len(reextracted)} cases")
            # Improvement from re-extraction
            improvements = []
            for r in reextracted:
                a1 = r.get("attempt1_accuracy", 0)
                final = r.get("final_accuracy", 0)
                if a1 > 0:
                    improvements.append(final - a1)
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                print(f"Average improvement from re-extraction: {avg_improvement:+.1%}")

    # Show failed/error cases
    if failed:
        print("\nValidation failed:")
        for r in failed:
            unresolved = r.get("unresolved_fields_count", 0)
            print(f"  - {r.get('waiver_key', 'unknown')}: final accuracy {r.get('final_accuracy', 0):.1%}, {unresolved} unresolved fields")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  - {r.get('waiver_key', 'unknown')}: {r.get('error', 'Unknown error')}")

    print("="*70)
