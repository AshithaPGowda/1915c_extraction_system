"""
Per-waiver merge into the master CSVs (all_waivers.csv / all_services.csv).

Unlike scripts/export_to_csv.py (append-only, skips waivers already present)
and storage.services_store.export_all_services_to_csv (rewrites the WHOLE file
from SQLite), these functions replace ONLY the target waiver's rows and keep
every other row byte-for-byte as it was. The previous file is backed up first.
Used by the ADK "write to CSV" tools; nothing runs them automatically.
"""

import csv
import importlib.util
import os
import shutil
from datetime import datetime
from pathlib import Path

from ..shared_libraries.paths import (
    CUSTOM_VALUES_CSV, CUSTOM_SERVICE_VALUES_CSV,
    OUTPUTS_DIR as OUTPUT_DIR, WAIVER_LEVEL_CSV, SERVICE_LEVEL_CSV,
    CSV_BACKUP_DIR as BACKUP_DIR, SCRIPTS_DIR, VERIFIED_DIR,
)


def _target(output_csv, default: Path) -> Path:
    """Resolve the CSV to write: the default new-results CSV, or a caller-chosen
    file (a bare file name goes in data/outputs/). The verified masters in
    data/verified/ are never accepted as a target."""
    if not output_csv:
        return default
    p = Path(output_csv).expanduser()
    if not p.is_absolute() and p.parent == Path("."):
        p = OUTPUT_DIR / p.name
    p = p.resolve()
    if VERIFIED_DIR.resolve() in p.parents:
        raise ValueError(f"{p} is in the verified folder, which is read-only")
    return p


def _merge(csv_path: Path, header: list, new_rows: list, key_col: str,
           waiver_key: str, dry_run: bool) -> dict:
    kept, replaced = [], 0
    existing_header = None
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
            if existing_header != header:
                return {"status": "error", "error": (
                    f"{csv_path.name} header differs from the expected columns "
                    f"({len(existing_header)} vs {len(header)}); refusing to merge.")}
            ki = header.index(key_col)
            for row in reader:
                if len(row) > ki and row[ki] == waiver_key:
                    replaced += 1
                else:
                    kept.append(row)
    result = {
        "status": "dry_run" if dry_run else "success",
        "csv_path": str(csv_path),
        "waiver_key": waiver_key,
        "rows_replaced": replaced,
        "rows_written": len(new_rows),
        "rows_total_after": len(kept) + len(new_rows),
        "other_waivers_rows_kept": len(kept),
    }
    if dry_run:
        return result

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_header is not None:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"{csv_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(csv_path, backup)
        result["backup_path"] = str(backup)
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
        w.writerows(new_rows)
    os.replace(tmp, csv_path)
    return result


def merge_overview_into_csv(waiver_key: str, dry_run: bool = False, output_csv: str = "") -> dict:
    """Replace this waiver's row in the waiver-level results CSV
    (default data/outputs/waiver_level_extraction.csv) with its overview.json."""
    try:
        csv_path = _target(output_csv, WAIVER_LEVEL_CSV)
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    spec = importlib.util.spec_from_file_location(
        "_export_to_csv", SCRIPTS_DIR / "export_to_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = mod.read_overview_json(OUTPUT_DIR / waiver_key)
    if not data:
        return {"status": "error", "error": f"No overview.json for {waiver_key} - finalize it first"}
    row = mod.extract_main_fields(data)
    header = list(mod.CANONICAL_FIELDS)
    return _merge(csv_path, header, [[row.get(c, "") for c in header]],
                  "waiver_key", waiver_key, dry_run)


def merge_services_into_csv(waiver_key: str, dry_run: bool = False, output_csv: str = "") -> dict:
    """Replace this waiver's rows in the service-level results CSV
    (default data/outputs/service_level_extraction.csv) with its rows in the services DB."""
    try:
        csv_path = _target(output_csv, SERVICE_LEVEL_CSV)
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    from .services_store import get_services, sheet_columns, _csv_value

    rows = get_services(waiver_key)
    if not rows:
        return {"status": "error", "error": f"No saved services rows for {waiver_key} - finalize it first"}
    columns = sheet_columns()
    extras = [("Row Type", "row_type"), ("Parent Service", "parent_service"),
              ("Waiver Key", "waiver_key")]
    header = [h for h, _ in columns] + [h for h, _ in extras]
    new_rows = []
    for r in rows:
        data = ["" if key is None else _csv_value(r.get(key)) for _, key in columns]
        data += [_csv_value(r.get(key)) for _, key in extras]
        new_rows.append(data)
    return _merge(csv_path, header, new_rows, "Waiver Key", waiver_key, dry_run)


CUSTOM_HEADER = ["waiver_key", "label", "value", "page", "evidence", "confidence",
                 "description", "value_type", "model", "extracted_at", "approved_at"]


def _merge_custom(row: dict, header: list, key_len: int, default_path: Path,
                  output_csv: str, dry_run: bool) -> dict:
    """Add/replace one row, keyed by the first `key_len` columns, in a
    long-format custom-values CSV. Same backup + never-the-verified-folder rules
    as the other merges."""
    try:
        csv_path = _target(output_csv, default_path)
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    new = [str(row.get(c, "") if row.get(c) is not None else "") for c in header]
    kept, replaced = [], 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            head = next(reader)
            if head != header:
                return {"status": "error", "error": f"{csv_path.name} has different columns; refusing to merge."}
            for r in reader:
                if r[:key_len] == new[:key_len]:
                    replaced += 1
                else:
                    kept.append(r)
    result = {"status": "dry_run" if dry_run else "success", "csv_path": str(csv_path),
              "rows_replaced": replaced, "rows_total_after": len(kept) + 1}
    if dry_run:
        return result
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"{csv_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(csv_path, backup)
        result["backup_path"] = str(backup)
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
        w.writerow(new)
    os.replace(tmp, csv_path)
    return result


def merge_custom_value(row: dict, output_csv: str = "", dry_run: bool = False) -> dict:
    """Add/replace one user-defined waiver-level value (keyed by waiver_key + label)
    in data/outputs/custom_extractions.csv."""
    return _merge_custom(row, CUSTOM_HEADER, 2, CUSTOM_VALUES_CSV, output_csv, dry_run)


CUSTOM_SERVICE_HEADER = ["waiver_key", "service_name", "label", "value", "page", "evidence",
                         "confidence", "description", "value_type", "model", "extracted_at", "approved_at"]


def merge_custom_service_value(row: dict, output_csv: str = "", dry_run: bool = False) -> dict:
    """Add/replace one user-defined service-specific value (keyed by waiver_key +
    service_name + label; service_name is empty for a waiver-wide value about
    services) in data/outputs/custom_service_extractions.csv."""
    return _merge_custom(row, CUSTOM_SERVICE_HEADER, 3, CUSTOM_SERVICE_VALUES_CSV, output_csv, dry_run)


def _overview_header() -> list:
    spec = importlib.util.spec_from_file_location("_export_to_csv", SCRIPTS_DIR / "export_to_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CANONICAL_FIELDS)


def _services_header() -> list:
    from .services_store import sheet_columns
    extras = ["Row Type", "Parent Service", "Waiver Key"]
    return [h for h, _ in sheet_columns()] + extras


def ensure_results_csvs() -> dict:
    """Create any missing results CSV (header row only) so the files exist as
    soon as an agent boots. Existing files are never touched. Returns
    {path: "created" | "exists" | "error: ..."}."""
    out = {}
    for path, header_fn in (
        (WAIVER_LEVEL_CSV, _overview_header),
        (SERVICE_LEVEL_CSV, _services_header),
        (CUSTOM_VALUES_CSV, lambda: CUSTOM_HEADER),
        (CUSTOM_SERVICE_VALUES_CSV, lambda: CUSTOM_SERVICE_HEADER),
    ):
        try:
            if path.exists():
                out[str(path)] = "exists"
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header_fn())
            out[str(path)] = "created"
        except Exception as e:  # noqa: BLE001 - never block agent startup
            out[str(path)] = f"error: {e}"
    return out
