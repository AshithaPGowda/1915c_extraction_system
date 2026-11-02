"""
Normalisation Agent for Services Extraction.

Cleans and standardizes extracted service values before storage.
Ensures consistent formatting for:
- Cost values (numeric, no currency symbols)
- User counts (integer)
- Coded fields (integer values)
- Service names / unit strings (trimmed whitespace)
- Component cost (computed as users x avg units x avg cost per unit,
  matching the sheet's formula column - never extracted)
"""

import re
from typing import Any

from ..shared_libraries.file_utils import log_event


def normalise_cost(value: Any) -> Any:
    """
    Normalise cost values to numeric or "N/A".

    Args:
        value: Raw cost value

    Returns:
        Numeric value, "N/A", or None
    """
    if value is None:
        return None

    # Handle string "N/A"
    if isinstance(value, str):
        val_upper = value.strip().upper()
        if val_upper == "N/A" or val_upper == "NA":
            return "N/A"

        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,\s]', '', value)
        if not cleaned:
            return None

        try:
            # Try float conversion
            num = float(cleaned)
            # Round to 2 decimal places for currency
            return round(num, 2)
        except ValueError:
            # Return original if can't parse
            return value

    # Already numeric
    if isinstance(value, (int, float)):
        return round(float(value), 2)

    return value


def normalise_user_count(value: Any) -> int | None:
    """
    Normalise user count to integer.

    Args:
        value: Raw user count value

    Returns:
        Integer or None
    """
    if value is None:
        return None

    if isinstance(value, str):
        val_str = value.strip()
        if not val_str or val_str.upper() == "N/A":
            return None

        # Remove commas
        cleaned = val_str.replace(",", "")
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    if isinstance(value, (int, float)):
        return int(value)

    return None


def normalise_coded_field(value: Any, valid_range: tuple[int, int]) -> int | None:
    """
    Normalise a coded field to integer within valid range.

    Args:
        value: Raw field value
        valid_range: Tuple of (min, max) valid values

    Returns:
        Integer or None
    """
    if value is None:
        return None

    try:
        num = int(value)
        if valid_range[0] <= num <= valid_range[1]:
            return num
        else:
            return None  # Out of range
    except (ValueError, TypeError):
        return None


def normalise_provider_eligibility(value: Any) -> str | None:
    """
    Normalise provider eligibility to string of digits 0-3.

    Args:
        value: Raw value (e.g., "123", "023", 123)

    Returns:
        String of digits or None
    """
    if value is None:
        return None

    val_str = str(value).strip()
    if not val_str:
        return None

    # Remove any non-digit characters
    digits = re.sub(r'[^0-3]', '', val_str)

    if not digits:
        return None

    # Sort and dedupe to normalize
    # Actually, preserve order as it may be significant
    return digits


def normalise_decimal(value: Any) -> float | None:
    """
    Normalise a decimal value (avg units per user, avg cost per unit).

    Args:
        value: Raw value

    Returns:
        Float or None
    """
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = re.sub(r'[$,\s]', '', value)
        if not cleaned or cleaned.upper() in ("N/A", "NA"):
            return None
        try:
            return round(float(cleaned), 4)
        except ValueError:
            return None

    if isinstance(value, (int, float)):
        return round(float(value), 4)

    return None


def normalise_unit(value: Any) -> str | None:
    """
    Normalise a unit-of-service string (e.g. "15 min", "per hour").

    Args:
        value: Raw unit value

    Returns:
        Trimmed string or None
    """
    if value is None:
        return None
    unit = str(value).strip()
    unit = re.sub(r'\s+', ' ', unit)
    return unit if unit else None


def compute_component_cost(record: dict, year: int) -> float | None:
    """
    Compute the component cost for a year: users x avg units x avg cost/unit.

    Mirrors the sheet's formula column; returns None unless all three
    factors are present and numeric.
    """
    users = record.get(f"year_{year}_users")
    avg_units = record.get(f"year_{year}_avg_units_per_user")
    avg_cost = record.get(f"year_{year}_avg_cost_per_unit")
    if users is None or avg_units is None or avg_cost is None:
        return None
    try:
        return round(float(users) * float(avg_units) * float(avg_cost), 2)
    except (ValueError, TypeError):
        return None


def normalise_service_name(name: Any) -> str | None:
    """
    Normalise service name string.

    Args:
        name: Raw service name

    Returns:
        Cleaned string or None
    """
    if name is None:
        return None

    if not isinstance(name, str):
        name = str(name)

    # Strip whitespace
    name = name.strip()

    # Collapse internal whitespace
    name = re.sub(r'\s+', ' ', name)

    return name if name else None


def normalise_parent(parent: dict) -> dict:
    """
    Normalise a parent service record.

    Args:
        parent: Raw parent dict

    Returns:
        Normalised parent dict
    """
    normalised = {
        "row_type": "parent",
        "service_name": normalise_service_name(parent.get("service_name")),
        "state": normalise_service_name(parent.get("state")),
        "waiver_title": normalise_service_name(parent.get("waiver_title")),
        "file_name": parent.get("file_name"),
        "hcbs_taxonomy": normalise_service_name(parent.get("hcbs_taxonomy")),
        "service_type": normalise_coded_field(parent.get("service_type"), (1, 4)),
        "service_delivery_method": normalise_coded_field(
            parent.get("service_delivery_method"), (1, 3)
        ),
        "provider_eligibility": normalise_provider_eligibility(
            parent.get("provider_eligibility")
        ),
        "provider_specifications": normalise_coded_field(
            parent.get("provider_specifications"), (1, 3)
        ),
        "self_direction_authority": normalise_coded_field(
            parent.get("self_direction_authority"), (1, 3)
        ),
        "information_assistance": normalise_coded_field(
            parent.get("information_assistance"), (1, 1)
        ),
    }

    # Normalise year data
    for year in range(1, 6):
        normalised[f"year_{year}_capitation"] = normalise_coded_field(
            parent.get(f"year_{year}_capitation"), (0, 1)
        )
        normalised[f"year_{year}_unit"] = normalise_unit(
            parent.get(f"year_{year}_unit")
        )
        normalised[f"year_{year}_users"] = normalise_user_count(
            parent.get(f"year_{year}_users")
        )
        normalised[f"year_{year}_avg_units_per_user"] = normalise_decimal(
            parent.get(f"year_{year}_avg_units_per_user")
        )
        normalised[f"year_{year}_avg_cost_per_unit"] = normalise_decimal(
            parent.get(f"year_{year}_avg_cost_per_unit")
        )
        normalised[f"year_{year}_total_cost"] = normalise_cost(
            parent.get(f"year_{year}_total_cost")
        )
        normalised[f"year_{year}_component_cost"] = compute_component_cost(
            normalised, year
        )

    # Preserve evidence if present
    if "evidence" in parent:
        normalised["evidence"] = parent["evidence"]

    return normalised


def normalise_child(child: dict, parent_service: str | None = None) -> dict:
    """
    Normalise a child service record.

    Args:
        child: Raw child dict
        parent_service: Parent service name for reference

    Returns:
        Normalised child dict
    """
    normalised = {
        "row_type": "child",
        "service_name": normalise_service_name(child.get("service_name")),
        "parent_service": parent_service or normalise_service_name(
            child.get("parent_service")
        ),
        "hcbs_taxonomy": normalise_service_name(child.get("hcbs_taxonomy")),
    }

    # Normalise year data
    for year in range(1, 6):
        normalised[f"year_{year}_unit"] = normalise_unit(
            child.get(f"year_{year}_unit")
        )
        normalised[f"year_{year}_users"] = normalise_user_count(
            child.get(f"year_{year}_users")
        )
        normalised[f"year_{year}_avg_units_per_user"] = normalise_decimal(
            child.get(f"year_{year}_avg_units_per_user")
        )
        normalised[f"year_{year}_avg_cost_per_unit"] = normalise_decimal(
            child.get(f"year_{year}_avg_cost_per_unit")
        )
        normalised[f"year_{year}_component_cost"] = compute_component_cost(
            normalised, year
        )

        # Child costs should be "N/A"
        cost_key = f"year_{year}_total_cost"
        raw_cost = child.get(cost_key)
        if raw_cost is None or str(raw_cost).upper() == "N/A":
            normalised[cost_key] = "N/A"
        else:
            # If somehow a cost was extracted, preserve it but flag
            normalised[cost_key] = normalise_cost(raw_cost)

    # Preserve evidence if present
    if "evidence" in child:
        normalised["evidence"] = child["evidence"]

    return normalised


def normalise_service(service_data: dict) -> dict:
    """
    Normalise a complete service record (parent + children).

    Args:
        service_data: Dict with 'parent' and 'children' keys

    Returns:
        Normalised service dict
    """
    parent = service_data.get("parent", {})
    children = service_data.get("children", [])

    normalised_parent = normalise_parent(parent)
    parent_name = normalised_parent.get("service_name")

    normalised_children = [
        normalise_child(child, parent_name)
        for child in children
    ]

    return {
        "parent": normalised_parent,
        "children": normalised_children,
    }


def normalise_services(
    services_data: list[dict],
    waiver_key: str | None = None,
) -> list[dict]:
    """
    Normalise all services for a waiver.

    Args:
        services_data: List of service dicts with 'parent' and 'children'
        waiver_key: Optional waiver key for logging

    Returns:
        List of normalised service dicts
    """
    normalised = []
    errors = []

    for i, service in enumerate(services_data):
        try:
            normalised.append(normalise_service(service))
        except Exception as e:
            errors.append({
                "index": i,
                "service": service.get("parent", {}).get("service_name"),
                "error": str(e),
            })
            # Still try to include with minimal normalisation
            normalised.append(service)

    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_normalisation",
            "action": "normalisation_complete",
            "services_count": len(normalised),
            "error_count": len(errors),
            "errors": errors if errors else None,
        })

    return normalised
