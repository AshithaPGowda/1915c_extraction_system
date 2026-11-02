"""
SQLite storage for waiver extractions.

Stores normalised extraction results in a local SQLite database
for querying and analysis.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Any


# Database path
from ..shared_libraries.paths import DB_PATH as _DB
DB_PATH = str(_DB)


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection, creating DB and table if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extractions (
            waiver_key TEXT NOT NULL,
            extraction_type TEXT NOT NULL,
            extracted_json TEXT NOT NULL,
            accuracy REAL,
            attempts INTEGER,
            final_status TEXT,
            timestamp TEXT,
            norm_errors INTEGER,
            unknown_fields INTEGER,
            PRIMARY KEY (waiver_key, extraction_type)
        )
    """)

    # Add checkbox_resolver_method column if it doesn't exist
    try:
        conn.execute("""
            ALTER TABLE extractions ADD COLUMN checkbox_resolver_method TEXT
        """)
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Add gt_suspect_count column if it doesn't exist
    try:
        conn.execute("""
            ALTER TABLE extractions ADD COLUMN gt_suspect_count INTEGER DEFAULT 0
        """)
    except sqlite3.OperationalError:
        # Column already exists
        pass

    conn.commit()

    return conn


def save_extraction(
    waiver_key: str,
    extraction_type: str,
    normalised_dict: dict,
    pipeline_result: dict
) -> int:
    """
    Save an extraction result to SQLite.

    Uses upsert (INSERT OR REPLACE) so re-runs overwrite cleanly.

    Args:
        waiver_key: Normalized waiver key (e.g., "AK0260R0512")
        extraction_type: Type of extraction (e.g., "overview", "services")
        normalised_dict: The normalised extraction data
        pipeline_result: Result dict from pipeline containing accuracy, attempts, status

    Returns:
        Number of rows affected (always 1 for upsert)
    """
    conn = _get_connection()

    # Extract values from pipeline result
    accuracy = pipeline_result.get("accuracy")
    attempts = pipeline_result.get("attempts")
    final_status = pipeline_result.get("status")
    norm_errors = pipeline_result.get("norm_errors", 0)
    unknown_fields = pipeline_result.get("unknown_fields", 0)
    gt_suspect_count = pipeline_result.get("gt_suspect_count", 0)

    # Serialize checkbox resolver summary as JSON
    checkbox_resolver_summary = pipeline_result.get("checkbox_resolver_summary")
    checkbox_resolver_method = (
        json.dumps(checkbox_resolver_summary) if checkbox_resolver_summary else None
    )

    # Serialize the extraction data
    extracted_json = json.dumps(normalised_dict, indent=2, default=str)

    # Current timestamp
    timestamp = datetime.utcnow().isoformat()

    # Upsert
    cursor = conn.execute("""
        INSERT OR REPLACE INTO extractions (
            waiver_key,
            extraction_type,
            extracted_json,
            accuracy,
            attempts,
            final_status,
            timestamp,
            norm_errors,
            unknown_fields,
            checkbox_resolver_method,
            gt_suspect_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        waiver_key,
        extraction_type,
        extracted_json,
        accuracy,
        attempts,
        final_status,
        timestamp,
        norm_errors,
        unknown_fields,
        checkbox_resolver_method,
        gt_suspect_count,
    ))

    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()

    return rows_affected


def extraction_exists(waiver_key: str, extraction_type: str = "overview") -> bool:
    """
    Check if an extraction already exists in SQLite.

    Args:
        waiver_key: Normalized waiver key
        extraction_type: Type of extraction (default: "overview")

    Returns:
        True if extraction exists, False otherwise
    """
    conn = _get_connection()

    cursor = conn.execute("""
        SELECT 1 FROM extractions
        WHERE waiver_key = ? AND extraction_type = ?
        LIMIT 1
    """, (waiver_key, extraction_type))

    exists = cursor.fetchone() is not None
    conn.close()

    return exists


def get_extraction(waiver_key: str, extraction_type: str) -> dict | None:
    """
    Retrieve an extraction from SQLite.

    Args:
        waiver_key: Normalized waiver key
        extraction_type: Type of extraction

    Returns:
        Dict with extraction data and metadata, or None if not found
    """
    conn = _get_connection()

    cursor = conn.execute("""
        SELECT * FROM extractions
        WHERE waiver_key = ? AND extraction_type = ?
    """, (waiver_key, extraction_type))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    # Parse checkbox_resolver_method JSON if present
    checkbox_resolver_method = None
    try:
        if row["checkbox_resolver_method"]:
            checkbox_resolver_method = json.loads(row["checkbox_resolver_method"])
    except (KeyError, json.JSONDecodeError):
        pass

    # Handle gt_suspect_count (may not exist in older rows)
    try:
        gt_suspect_count = row["gt_suspect_count"] or 0
    except KeyError:
        gt_suspect_count = 0

    return {
        "waiver_key": row["waiver_key"],
        "extraction_type": row["extraction_type"],
        "extracted_data": json.loads(row["extracted_json"]),
        "accuracy": row["accuracy"],
        "attempts": row["attempts"],
        "final_status": row["final_status"],
        "timestamp": row["timestamp"],
        "norm_errors": row["norm_errors"],
        "unknown_fields": row["unknown_fields"],
        "checkbox_resolver_method": checkbox_resolver_method,
        "gt_suspect_count": gt_suspect_count,
    }


def list_extractions(extraction_type: str | None = None) -> list[dict]:
    """
    List all extractions, optionally filtered by type.

    Args:
        extraction_type: Optional filter by extraction type

    Returns:
        List of extraction metadata dicts (without full JSON)
    """
    conn = _get_connection()

    if extraction_type:
        cursor = conn.execute("""
            SELECT waiver_key, extraction_type, accuracy, attempts,
                   final_status, timestamp, norm_errors, unknown_fields
            FROM extractions
            WHERE extraction_type = ?
            ORDER BY timestamp DESC
        """, (extraction_type,))
    else:
        cursor = conn.execute("""
            SELECT waiver_key, extraction_type, accuracy, attempts,
                   final_status, timestamp, norm_errors, unknown_fields
            FROM extractions
            ORDER BY timestamp DESC
        """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
