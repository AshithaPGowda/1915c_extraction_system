#!/usr/bin/env python3
"""
1915(c) Waiver Services Extraction Pipeline — Entry Point

Extracts detailed service-level data from waiver PDFs.
Produces one row per service per waiver with parent/child hierarchy.

Usage:
  # Process a single waiver (uses overview.json for discovered services)
  python run_services_pipeline.py --waiver AK0260R0512

  # Process a single waiver in standalone mode (discovers services from PDF)
  python run_services_pipeline.py --waiver AK0260R0512 --standalone

  # Process all waivers that have overview.json
  python run_services_pipeline.py --all

  # Process all with limit
  python run_services_pipeline.py --all --limit 10

  # Skip waivers that already have services extracted
  python run_services_pipeline.py --all --skip-existing

  # Export all services to CSV
  python run_services_pipeline.py --export-csv data/outputs/my_export.csv

  # Process a specific PDF file
  python run_services_pipeline.py --file data/pdfs_267_latest/AK0260R0512.pdf   # any folder works
  python run_services_pipeline.py --all --pdf-dir ~/Desktop/new_pdfs --output-csv my_run.csv
"""

import asyncio
import argparse
import os
import sys

# Load .env file only if it exists (prefer system env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))


async def process_single_waiver(
    waiver_key: str,
    pdf_dir: str | None = None,
    standalone: bool = False,
    resume: bool = False,
) -> dict:
    """Process a single waiver for services extraction."""
    from waiver_level_extraction_agent.core.services_pipeline import run_services_extraction
    from waiver_level_extraction_agent.tools.services_tools import get_pdf_path_for_waiver
    from waiver_level_extraction_agent.shared_libraries.file_utils import normalize_waiver_key

    waiver_key = normalize_waiver_key(waiver_key)

    # Find PDF
    pdf_path = get_pdf_path_for_waiver(waiver_key, pdf_dir)
    if not pdf_path:
        print(f"Error: PDF not found for waiver {waiver_key}")
        return {
            "status": "error",
            "waiver_key": waiver_key,
            "error": "PDF not found",
        }

    print(f"Processing waiver: {waiver_key}")
    print(f"  PDF: {pdf_path}")

    result = await run_services_extraction(
        pdf_path=pdf_path,
        waiver_key=waiver_key,
        standalone=standalone,
        resume=resume,
    )

    return result


async def process_file(pdf_path: str, standalone: bool = False) -> dict:
    """Process a specific PDF file."""
    from waiver_level_extraction_agent.core.services_pipeline import run_services_extraction
    from waiver_level_extraction_agent.shared_libraries.file_utils import waiver_key_from_filename

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return {
            "status": "error",
            "pdf_path": pdf_path,
            "error": "File not found",
        }

    waiver_key = waiver_key_from_filename(pdf_path)
    print(f"Processing file: {pdf_path}")
    print(f"  Waiver key: {waiver_key}")

    result = await run_services_extraction(
        pdf_path=pdf_path,
        waiver_key=waiver_key,
        standalone=standalone,
    )

    return result


async def process_all_waivers(
    pdf_dir: str | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    standalone: bool = False,
) -> list[dict]:
    """Process all waivers that have overview.json."""
    from waiver_level_extraction_agent.tools.services_tools import (
        list_waivers_with_overview,
        get_pdf_path_for_waiver,
    )
    from waiver_level_extraction_agent.storage.services_store import services_exist
    from waiver_level_extraction_agent.core.services_pipeline import run_services_extraction

    # Get waivers with overview
    waivers = list_waivers_with_overview()
    print(f"Found {len(waivers)} waivers with overview.json")

    # Filter out existing if requested
    if skip_existing:
        original_count = len(waivers)
        waivers = [w for w in waivers if not services_exist(w)]
        skipped = original_count - len(waivers)
        if skipped > 0:
            print(f"Skipped {skipped} waivers with existing services")

    # Apply limit
    if limit:
        waivers = waivers[:limit]

    if not waivers:
        print("No waivers to process")
        return []

    print(f"Processing {len(waivers)} waivers")

    results = []
    for i, waiver_key in enumerate(waivers, 1):
        print(f"\n[{i}/{len(waivers)}] Processing: {waiver_key}")

        # Find PDF
        pdf_path = get_pdf_path_for_waiver(waiver_key, pdf_dir)
        if not pdf_path:
            print(f"  Warning: PDF not found, skipping")
            results.append({
                "status": "error",
                "waiver_key": waiver_key,
                "error": "PDF not found",
            })
            continue

        try:
            result = await run_services_extraction(
                pdf_path=pdf_path,
                waiver_key=waiver_key,
                standalone=standalone,
            )
            results.append(result)

            # Print summary
            print(f"  Status: {result.get('status')}")
            print(f"  Services: {result.get('services_extracted', 0)}")
            print(f"  Total rows: {result.get('total_rows', 0)}")
            if result.get("accuracy") is not None:
                print(f"  Accuracy: {result['accuracy']:.1%} "
                      f"(attempts: {result.get('attempts', 1)}, "
                      f"acroform: {result.get('acroform_confidence')})")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "status": "error",
                "waiver_key": waiver_key,
                "error": str(e),
            })

    return results


def export_csv(output_path: str) -> None:
    """Export all services to CSV."""
    from waiver_level_extraction_agent.storage.services_store import (
        export_all_services_to_csv,
        get_services_summary,
    )

    print(f"Exporting services to {output_path}")

    summary = get_services_summary()
    print(f"  Total waivers: {summary['total_waivers']}")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Parent rows: {summary['parent_rows']}")
    print(f"  Child rows: {summary['child_rows']}")

    rows = export_all_services_to_csv(output_path)
    print(f"Exported {rows} rows to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print a summary of processing results."""
    print("\n" + "=" * 70)
    print("SERVICES EXTRACTION SUMMARY")
    print("=" * 70)

    success = [r for r in results if r.get("status") == "success"]
    human = [r for r in results if r.get("status") == "human_review_required"]
    failed = [r for r in results if r.get("status") == "validation_failed"]
    errors = [r for r in results if r.get("status") == "error"]

    print(f"Total processed      : {len(results)}")
    print(f"Success              : {len(success)}")
    print(f"Human review needed  : {len(human)}")
    print(f"Validation failed    : {len(failed)}")
    print(f"Errors               : {len(errors)}")

    accuracies = [r["accuracy"] for r in results
                  if r.get("accuracy") is not None]
    if accuracies:
        print(f"Mean accuracy        : {sum(accuracies) / len(accuracies):.1%}")

    success = success + human + failed  # all produced rows

    if success:
        total_services = sum(r.get("services_extracted", 0) for r in success)
        total_rows = sum(r.get("total_rows", 0) for r in success)
        total_parents = sum(r.get("parent_rows", 0) for r in success)
        total_children = sum(r.get("child_rows", 0) for r in success)

        print(f"\nServices extracted   : {total_services}")
        print(f"Total rows           : {total_rows}")
        print(f"  Parent rows        : {total_parents}")
        print(f"  Child rows         : {total_children}")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  - {r.get('waiver_key', 'unknown')}: {r.get('error', 'Unknown error')}")

    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="1915(c) Waiver Services Extraction Pipeline"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--waiver", type=str,
        help="Process a single waiver by key (e.g., AK0260R0512)"
    )
    input_group.add_argument(
        "--file", type=str,
        help="Process a specific PDF file"
    )
    input_group.add_argument(
        "--all", action="store_true",
        help="Process all waivers with overview.json"
    )
    input_group.add_argument(
        "--export-csv", type=str, metavar="PATH",
        help="Export all services to CSV (no extraction)"
    )
    input_group.add_argument(
        "--rebuild", type=str, metavar="WAIVER_KEY",
        help="Rebuild outputs from the latest checkpoint of a crashed run "
             "(no model calls)"
    )

    # Options
    parser.add_argument(
        "--standalone", action="store_true",
        help="Discover services from PDF instead of using overview.json"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of waivers to process (with --all)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip waivers that already have services extracted"
    )
    parser.add_argument(
        "--pdf-dir", type=str,
        help="Directory containing PDF files (default: data/pdfs_267_latest)"
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Write results to this CSV instead of the default "
             "data/outputs/service_level_extraction.csv (bare name -> data/outputs/)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="With --waiver: skip services already checkpointed by a prior "
             "killed/crashed run of this waiver, extracting only the "
             "remainder (falls back to a normal fresh run if no checkpoint "
             "exists)"
    )
    parser.add_argument(
        "--only-service", type=str, metavar="SERVICE_NAME",
        help="With --waiver: extract just this one named service (one "
             "model call) and append it to the checkpoint - no critique, "
             "no DB save. For manually retrying one service that keeps "
             "failing without re-running the whole waiver. Follow with "
             "--resume (or --rebuild) to fold it into the saved output."
    )

    args = parser.parse_args()

    from waiver_level_extraction_agent.shared_libraries.config_check import check_cloud_config
    try:
        check_cloud_config()
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    if args.output_csv:
        from pathlib import Path
        p = Path(args.output_csv).expanduser()
        if not p.is_absolute() and p.parent == Path("."):
            p = Path(__file__).resolve().parent / "data" / "outputs" / p.name
        os.environ["SERVICE_LEVEL_CSV"] = str(p.resolve())

    # Handle export-only
    if args.export_csv:
        export_csv(args.export_csv)
        return

    # Handle checkpoint rebuild (no model calls)
    if args.rebuild:
        from waiver_level_extraction_agent.core.services_pipeline import rebuild_from_checkpoint
        from waiver_level_extraction_agent.shared_libraries.file_utils import normalize_waiver_key
        result = rebuild_from_checkpoint(normalize_waiver_key(args.rebuild))
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    # Process based on input type
    if args.waiver and args.only_service:
        import json
        from waiver_level_extraction_agent.core.services_pipeline import extract_one_service
        from waiver_level_extraction_agent.tools.services_tools import get_pdf_path_for_waiver
        from waiver_level_extraction_agent.shared_libraries.file_utils import normalize_waiver_key

        waiver_key = normalize_waiver_key(args.waiver)
        pdf_path = get_pdf_path_for_waiver(waiver_key, args.pdf_dir)
        if not pdf_path:
            print(f"Error: PDF not found for waiver {waiver_key}")
            return
        print(f"Extracting single service '{args.only_service}' for {waiver_key}...")
        result = extract_one_service(waiver_key, pdf_path, args.only_service)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.waiver:
        result = await process_single_waiver(
            waiver_key=args.waiver,
            pdf_dir=args.pdf_dir,
            standalone=args.standalone,
            resume=args.resume,
        )
        print_summary([result])

    elif args.file:
        result = await process_file(
            pdf_path=args.file,
            standalone=args.standalone,
        )
        print_summary([result])

    elif args.all:
        results = await process_all_waivers(
            pdf_dir=args.pdf_dir,
            limit=args.limit,
            skip_existing=args.skip_existing,
            standalone=args.standalone,
        )
        print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
