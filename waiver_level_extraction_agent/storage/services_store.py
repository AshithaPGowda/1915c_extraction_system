"""
SQLite storage for waiver services extraction.

Stores service-level extraction results in a local SQLite database
with flat rows (one per service, parent or child) matching the
"Waiver Services - PD_Append C&E" sheet (49 columns), plus a per-waiver
run-metadata table (accuracy, attempts, final status).
"""

import os
import re
import csv
import json
import sqlite3
from datetime import datetime
from typing import Any


# Database path (same as extractions DB)
from ..shared_libraries.paths import DB_PATH as _DB
DB_PATH = str(_DB)

_LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "layouts", "layout_services_pass2_rich.json"
)

# Per-year internal field suffixes, in sheet order within each year block.
YEAR_SUFFIXES = [
    "capitation",
    "unit",
    "users",
    "avg_units_per_user",
    "avg_cost_per_unit",
    "component_cost",
    "total_cost",
]

# Internal column list for the services table, in sheet order.
SERVICE_COLUMNS = (
    [
        "waiver_key", "state", "waiver_title", "file_name",
        "row_type", "parent_service", "service_name", "hcbs_taxonomy",
        "service_type", "delivery_method", "provider_eligibility",
        "provider_specifications", "self_direction_authority",
        "information_assistance",
    ]
    + [
        f"year_{year}_{suffix}"
        for year in range(1, 6)
        for suffix in YEAR_SUFFIXES
    ]
    + ["row_index", "timestamp"]
)

_COLUMN_TYPES = {
    "service_type": "INTEGER",
    "delivery_method": "INTEGER",
    "provider_specifications": "INTEGER",
    "self_direction_authority": "INTEGER",
    "information_assistance": "INTEGER",
    "row_index": "INTEGER",
    **{f"year_{y}_capitation": "INTEGER" for y in range(1, 6)},
    **{f"year_{y}_users": "INTEGER" for y in range(1, 6)},
    **{f"year_{y}_avg_units_per_user": "REAL" for y in range(1, 6)},
    **{f"year_{y}_avg_cost_per_unit": "REAL" for y in range(1, 6)},
    **{f"year_{y}_component_cost": "REAL" for y in range(1, 6)},
}


def _classify_sheet_header(header: str) -> str | None:
    """Map a sheet header to its internal column name (None = blank column)."""
    m = re.search(r"Year\s*(\d)", header)
    if m:
        year = m.group(1)
        if header.startswith("J-2d"):
            return f"year_{year}_capitation"
        if header.startswith("Unit"):
            return f"year_{year}_unit"
        if header.startswith("# Users"):
            return f"year_{year}_users"
        if header.startswith("Avg. Units"):
            return f"year_{year}_avg_units_per_user"
        if header.startswith("Avg. Cost"):
            return f"year_{year}_avg_cost_per_unit"
        if header.startswith("Component Cost"):
            return f"year_{year}_component_cost"
        if header.startswith("Total Cost"):
            return f"year_{year}_total_cost"
        return None
    if header.startswith("1.A"):
        return "state"
    if header.startswith("1.B"):
        return "waiver_title"
    if header.startswith("1.C"):
        return "file_name"
    if header.startswith("C-1/C-3 Waiver Service"):
        return "service_name"
    if header.startswith("HCBS Taxonomy"):
        return "hcbs_taxonomy"
    if header.startswith("C-1/C-3 Service Type"):
        return "service_type"
    if header.startswith("C-1/C-3 Service Delivery Method"):
        return "delivery_method"
    if header.startswith("C-1/C-3 Specify whether"):
        return "provider_eligibility"
    if header.startswith("C-1/C-3: Provider Specifications"):
        return "provider_specifications"
    if header.startswith("E-1g"):
        return "self_direction_authority"
    if header.startswith("E-1j"):
        return "information_assistance"
    if header.startswith("Transition"):
        return None  # blank link columns in the sheet
    raise ValueError(f"Unrecognized sheet header: {header!r}")


def sheet_columns() -> list[tuple[str, str | None]]:
    """
    The 49 sheet headers (verbatim, in order) paired with internal column
    names. Headers come from the layout file, which was generated from the
    sheet, so whitespace quirks (trailing spaces, "Year2" vs "Year 2")
    survive round-trips exactly.
    """
    with open(_LAYOUT_PATH) as f:
        layout = json.load(f)
    return [(header, _classify_sheet_header(header)) for header in layout]


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection, creating/migrating tables as needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Migrate: drop a pre-sheet-2 services table (the old 20-field schema
    # was never populated in production runs).
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='services'"
    )
    if cursor.fetchone() is not None:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(services)")}
        if "hcbs_taxonomy" not in existing:
            conn.execute("DROP TABLE services")

    col_defs = []
    for col in SERVICE_COLUMNS:
        col_type = _COLUMN_TYPES.get(col, "TEXT")
        constraint = " NOT NULL" if col in ("waiver_key", "row_type", "row_index") else ""
        col_defs.append(f"{col} {col_type}{constraint}")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS services (
            {', '.join(col_defs)},
            PRIMARY KEY (waiver_key, row_index)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS services_runs (
            waiver_key TEXT PRIMARY KEY,
            accuracy REAL,
            attempts INTEGER,
            final_status TEXT,
            services_count INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()

    return conn


def save_services(waiver_key: str, rows: list[dict]) -> int:
    """
    Save service rows to SQLite.

    Deletes existing rows for this waiver and inserts new ones.
    This ensures clean re-runs.

    Args:
        waiver_key: Normalized waiver key (e.g., "AK0260R0512")
        rows: List of flat row dicts (from flatten_services_to_rows)

    Returns:
        Number of rows inserted
    """
    conn = _get_connection()
    timestamp = datetime.utcnow().isoformat()

    # Delete existing rows for this waiver
    conn.execute("DELETE FROM services WHERE waiver_key = ?", (waiver_key,))

    placeholders = ", ".join("?" for _ in SERVICE_COLUMNS)
    insert_sql = (
        f"INSERT INTO services ({', '.join(SERVICE_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )

    for row in rows:
        values = []
        for col in SERVICE_COLUMNS:
            if col == "timestamp":
                values.append(timestamp)
            elif col.endswith("_total_cost"):
                values.append(_format_cost(row.get(col)))
            else:
                values.append(row.get(col))
        conn.execute(insert_sql, values)

    conn.commit()
    rows_inserted = len(rows)
    conn.close()

    return rows_inserted


def save_services_run(
    waiver_key: str,
    accuracy: float | None,
    attempts: int,
    final_status: str,
    services_count: int,
) -> None:
    """Record per-waiver run metadata (accuracy, attempts, final status)."""
    conn = _get_connection()
    conn.execute("""
        INSERT INTO services_runs
            (waiver_key, accuracy, attempts, final_status, services_count, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(waiver_key) DO UPDATE SET
            accuracy = excluded.accuracy,
            attempts = excluded.attempts,
            final_status = excluded.final_status,
            services_count = excluded.services_count,
            timestamp = excluded.timestamp
    """, (
        waiver_key, accuracy, attempts, final_status, services_count,
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def _format_cost(cost: Any) -> str | None:
    """Format cost value for storage. Preserves 'N/A' as string."""
    if cost is None:
        return None
    if isinstance(cost, str):
        if cost.upper() == "N/A":
            return "N/A"
        # Try to clean numeric string
        try:
            return str(float(cost.replace(",", "").replace("$", "")))
        except ValueError:
            return cost
    if isinstance(cost, (int, float)):
        return str(cost)
    return str(cost)


def get_services(waiver_key: str) -> list[dict]:
    """
    Retrieve all service rows for a waiver.

    Args:
        waiver_key: Normalized waiver key

    Returns:
        List of service row dicts, ordered by row_index
    """
    conn = _get_connection()

    cursor = conn.execute("""
        SELECT * FROM services
        WHERE waiver_key = ?
        ORDER BY row_index
    """, (waiver_key,))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return rows


def services_exist(waiver_key: str) -> bool:
    """
    Check if services have been extracted for a waiver.

    Args:
        waiver_key: Normalized waiver key

    Returns:
        True if services exist, False otherwise
    """
    conn = _get_connection()

    cursor = conn.execute("""
        SELECT 1 FROM services
        WHERE waiver_key = ?
        LIMIT 1
    """, (waiver_key,))

    exists = cursor.fetchone() is not None
    conn.close()

    return exists


def list_waiver_keys_with_services() -> list[str]:
    """
    List all waiver keys that have services extracted.

    Returns:
        List of waiver keys
    """
    conn = _get_connection()

    cursor = conn.execute("""
        SELECT DISTINCT waiver_key FROM services
        ORDER BY waiver_key
    """)

    keys = [row["waiver_key"] for row in cursor.fetchall()]
    conn.close()

    return keys


def export_waiver_services_to_csv(waiver_key: str, output_path: str) -> int:
    """
    Export services for a single waiver to CSV.

    Args:
        waiver_key: Normalized waiver key
        output_path: Path for the CSV file

    Returns:
        Number of rows exported
    """
    rows = get_services(waiver_key)
    if not rows:
        return 0

    return _write_csv(rows, output_path)


def export_all_services_to_csv(output_path: str) -> int:
    """
    Export all services from all waivers to a single CSV.

    Args:
        output_path: Path for the CSV file

    Returns:
        Number of rows exported
    """
    conn = _get_connection()

    # Oldest extraction first: the CSV behaves append-only - waivers keep
    # their position from when they were first extracted, and a re-run
    # moves that waiver to the bottom like a fresh append.
    cursor = conn.execute("""
        SELECT * FROM services
        ORDER BY timestamp ASC, waiver_key ASC, row_index ASC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if not rows:
        return 0

    return _write_csv(rows, output_path)


def _write_csv(rows: list[dict], output_path: str) -> int:
    """
    Write rows to CSV using the 49 verbatim sheet headers (in sheet order),
    followed by bookkeeping columns (Row Type, Parent Service, Waiver Key).
    """
    if not rows:
        return 0

    columns = sheet_columns()
    extras = [("Row Type", "row_type"), ("Parent Service", "parent_service"),
              ("Waiver Key", "waiver_key")]

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([h for h, _ in columns] + [h for h, _ in extras])
        for row in rows:
            data = [
                "" if key is None else _csv_value(row.get(key))
                for _, key in columns
            ]
            data += [_csv_value(row.get(key)) for _, key in extras]
            writer.writerow(data)

    return len(rows)


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def delete_services(waiver_key: str) -> int:
    """
    Delete all services for a waiver.

    Args:
        waiver_key: Normalized waiver key

    Returns:
        Number of rows deleted
    """
    conn = _get_connection()

    cursor = conn.execute(
        "DELETE FROM services WHERE waiver_key = ?",
        (waiver_key,)
    )

    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    return deleted


def get_services_summary() -> dict:
    """
    Get summary statistics for all extracted services.

    Returns:
        Dict with summary statistics
    """
    conn = _get_connection()

    # Total rows
    cursor = conn.execute("SELECT COUNT(*) as count FROM services")
    total_rows = cursor.fetchone()["count"]

    # Unique waivers
    cursor = conn.execute("SELECT COUNT(DISTINCT waiver_key) as count FROM services")
    total_waivers = cursor.fetchone()["count"]

    # Parent vs child breakdown
    cursor = conn.execute("""
        SELECT row_type, COUNT(*) as count
        FROM services
        GROUP BY row_type
    """)
    type_counts = {row["row_type"]: row["count"] for row in cursor.fetchall()}

    conn.close()

    return {
        "total_rows": total_rows,
        "total_waivers": total_waivers,
        "parent_rows": type_counts.get("parent", 0),
        "child_rows": type_counts.get("child", 0),
    }
