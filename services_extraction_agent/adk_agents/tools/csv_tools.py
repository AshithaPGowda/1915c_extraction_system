"""
Opt-in CSV tool for the services ADK app. Finalize/save writes only the
waiver's JSON outputs and SQLite rows; the master all_services.csv changes
only when this tool is called.
"""

from waiver_level_extraction_agent.storage.csv_merge import merge_services_into_csv


def write_services_to_csv(waiver_key: str, dry_run: bool = False, output_csv: str = "") -> dict:
    """Merge this waiver's saved services rows into the service-level results
    CSV (default data/outputs/service_level_extraction.csv).

    Replaces ONLY this waiver's existing rows (if any) and appends the new
    ones; every other row is left untouched, and the previous CSV is backed up
    to data/csv_backups/ first. Never writes the verified masters in
    data/verified/. Run it after finalize_and_save.

    Args:
        waiver_key: The waiver identifier
        dry_run: If True, report what would change without writing anything
        output_csv: Optional different CSV to write instead of the default - a
            bare file name (saved in data/outputs/) or a full path

    Returns:
        dict with rows_replaced, rows_written, rows_total_after, backup_path
    """
    return merge_services_into_csv(waiver_key, dry_run=dry_run, output_csv=output_csv)
