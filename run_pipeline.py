#!/usr/bin/env python3
"""
1915(c) Waiver Extraction Pipeline — Entry Point

Usage:
  # Process canonical PDFs (highest version per waiver) in the default directory
  python run_pipeline.py

  # Process only the first 50 PDFs (for testing)
  python run_pipeline.py --limit 50

  # Process 50 diverse PDFs from different states
  python run_pipeline.py --limit 50 --diverse

  # Process a single specific PDF file
  python run_pipeline.py --file data/pdfs_267_latest/AL0407R0400.pdf   # any folder works
  python run_pipeline.py --waiver AL0407R0400                          # by key, from the default or --dir folder
  python run_pipeline.py --dir ~/Desktop/new_pdfs --output-csv my_run.csv

  # Process all PDFs (including older versions)
  python run_pipeline.py --all-versions

  # Process all PDFs in a custom directory
  python run_pipeline.py --dir /path/to/pdfs

  # Re-process PDFs (don't skip already processed)
  python run_pipeline.py --reprocess

  # Set max pages threshold (skip PDFs over this)
  python run_pipeline.py --max-pages 500
"""

import asyncio
import argparse
import os
import sys

# Load .env file only if it exists (prefer system env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # Don't override existing system env vars
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))


async def main():
    parser = argparse.ArgumentParser(description="1915(c) Waiver Extraction Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of PDFs to process (for testing)")
    parser.add_argument("--file", type=str, default=None,
                        help="Process a single specific PDF file")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing PDFs to process (default: data/pdfs_267_latest)")
    parser.add_argument("--waiver", type=str, default=None,
                        help="Process one waiver by key (e.g. PA0324R0600), looked up in --dir "
                             "or the default PDF folder")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Write results to this CSV instead of the default "
                             "data/outputs/waiver_level_extraction.csv (bare name -> data/outputs/)")
    parser.add_argument("--all-versions", action="store_true",
                        help="Process all PDFs, not just canonical (highest version)")
    parser.add_argument("--max-pages", type=int, default=1000,
                        help="Skip PDFs with more than this many pages (default: 1000)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-process PDFs even if already done (disable idempotency)")
    parser.add_argument("--diverse", action="store_true",
                        help="Select diverse PDFs from different states (use with --limit)")
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
        os.environ["WAIVER_LEVEL_CSV"] = str(p.resolve())

    from waiver_level_extraction_agent.core.runner import (
        run_batch, process_single_waiver, print_summary
    )

    if args.waiver:
        from waiver_level_extraction_agent.shared_libraries.paths import PDF_DIR
        from waiver_level_extraction_agent.tools.services_tools import get_pdf_path_for_waiver
        from waiver_level_extraction_agent.shared_libraries.file_utils import normalize_waiver_key
        key = normalize_waiver_key(args.waiver)
        pdf = get_pdf_path_for_waiver(key, args.dir or str(PDF_DIR))
        if not pdf:
            print(f"Error: PDF not found for waiver {key}")
            sys.exit(1)
        args.file = pdf

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        print(f"Processing single file: {args.file}")
        result = await process_single_waiver(args.file)
        print_summary([result])
    else:
        from waiver_level_extraction_agent.shared_libraries.paths import PDF_DIR
        pdf_dir = args.dir or str(PDF_DIR)
        use_canonical = not args.all_versions
        results = await run_batch(
            pdf_dir,
            limit=args.limit,
            use_canonical=use_canonical,
            max_pages=args.max_pages,
            skip_existing=not args.reprocess,
            diverse=args.diverse,
        )
        print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
