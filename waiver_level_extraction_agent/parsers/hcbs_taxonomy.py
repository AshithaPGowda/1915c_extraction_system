"""
HCBS Taxonomy resolution in the reference sheet's vocabulary.

Precedence (per researcher guidance, 2026-07-29):
1. Curated service-name crosswalk (data/layouts/hcbs_taxonomy_crosswalk.json)
   - beats everything, including the PDF's own taxonomy table (the sheet
   classifies e.g. Pest Control as "Tech mods equip" even though the form
   prints "17 Other Services").
2. The PDF's parsed taxonomy table, translated into sheet vocabulary -
   but only when it maps to something more specific than "Other".
3. LLM crosswalk inference (services_agent.infer_taxonomy_fallback),
   constrained to the sheet labels, "Other" only as a last resort.
"""

import os
import re
import json
from functools import lru_cache

_CROSSWALK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "layouts", "hcbs_taxonomy_crosswalk.json"
)

# Official taxonomy category (MMRR 2014 / PDF form) -> sheet vocabulary.
OFFICIAL_TO_SHEET = {
    "case management": "Case management",
    "round-the-clock services": "Round-the-clock services",
    "supported employment": "Supported employment",
    "day services": "Day services",
    "nursing": "Nursing services",
    "home-delivered meals": "Home delivered meals",
    "home delivered meals": "Home delivered meals",
    "home-based services": "Home-based services",
    "caregiver support": "Caregiver support services",
    "services supporting participant direction": "Support partic direct",
    "participant training": "Participant training",
    "equipment, technology, and modifications": "Tech mods equip",
    "nonmedical transportation": "Non-medical transportation",
    "non-medical transportation": "Non-medical transportation",
    "community transition services": "Community transition svcs",
    "rent and food expenses for live-in caregiver": "Other",
    "other mental health and behavioral services": "Other",
    "other health and therapeutic services": "Other",
    "other services": "Other",
    "unknown": "Other",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


@lru_cache(maxsize=1)
def _load_crosswalk() -> tuple[dict, list]:
    with open(_CROSSWALK_PATH) as f:
        data = json.load(f)
    mapping = {_norm(k): v for k, v in data["mapping"].items()}
    return mapping, data["labels"]


def sheet_labels() -> list[str]:
    """The taxonomy vocabulary used by the reference sheet."""
    return list(_load_crosswalk()[1])


def lookup_curated(service_name: str | None) -> str | None:
    """
    Curated crosswalk lookup: exact normalized match first, then
    longest-key containment (so "Respite - Per Diem" hits "Respite").
    Short keys (< 6 chars, e.g. "CLS", "PERS") only match exactly, to
    avoid accidental substring hits.
    """
    if not service_name:
        return None
    mapping, _ = _load_crosswalk()
    name = _norm(service_name)

    if name in mapping:
        return mapping[name]

    best_key = None
    for key in mapping:
        if len(key) < 6:
            continue
        if key in name and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return mapping[best_key] if best_key else None


def translate_pdf_category(category: str | None) -> str | None:
    """
    Translate a PDF-printed taxonomy category ("17 Other Services",
    "08 Home-Based Services") into sheet vocabulary.
    """
    if not category:
        return None
    stripped = re.sub(r"^\s*\d+\s*", "", str(category)).strip()
    if not stripped:
        return None
    return OFFICIAL_TO_SHEET.get(_norm(stripped), stripped)


def resolve_taxonomy(
    service_name: str | None,
    pdf_categories: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Deterministic taxonomy resolution for one service.

    Args:
        service_name: The service name
        pdf_categories: Raw category strings parsed from the PDF's
            HCBS Taxonomy table (may be empty)

    Returns:
        (label, source) where source is "curated_crosswalk" |
        "pdf_taxonomy_table" | None. A None label means the LLM inference
        fallback should decide.
    """
    curated = lookup_curated(service_name)
    if curated:
        return curated, "curated_crosswalk"

    for category in pdf_categories or []:
        label = translate_pdf_category(category)
        if label and label != "Other":
            return label, "pdf_taxonomy_table"

    return None, None
