"""
Stage Storage Tools for ADK Pipeline.

Provides intermediate storage between pipeline stages so agents can:
1. Save their output for the next stage
2. Load previous stage output
3. Restart from any stage
4. Cache page images per PDF for attempt 3 restarts
"""

import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

# Stage directory for intermediate outputs
from ...shared_libraries.paths import STAGES_DIR

# Valid stage names in pipeline order
VALID_STAGES = [
    "checkbox_resolution",
    "extraction_attempt_1",
    "critic_attempt_1",
    "extraction_attempt_2",
    "critic_attempt_2",
    "extraction_attempt_3",
    "critic_attempt_3",
    "normalization",
]


def _get_waiver_stage_dir(waiver_key: str) -> Path:
    """Get the stage directory for a waiver."""
    stage_dir = STAGES_DIR / waiver_key
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _get_images_dir(waiver_key: str) -> Path:
    """Get the images cache directory for a waiver."""
    images_dir = _get_waiver_stage_dir(waiver_key) / "page_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def save_stage_output(waiver_key: str, stage: str, data: dict) -> dict:
    """Save output from a pipeline stage.

    This persists the output so subsequent agents can load it,
    and allows restarting from any stage.

    Args:
        waiver_key: The waiver identifier (e.g., "NV2426R0001")
        stage: Stage name - one of:
            - checkbox_resolution
            - extraction_attempt_1
            - critic_attempt_1
            - extraction_attempt_2
            - critic_attempt_2
            - extraction_attempt_3
            - critic_attempt_3
            - normalization
        data: The stage output data to save

    Returns:
        dict with save status and path
    """
    if stage not in VALID_STAGES:
        return {
            "status": "error",
            "error": f"Invalid stage '{stage}'. Valid stages: {VALID_STAGES}",
        }

    try:
        stage_dir = _get_waiver_stage_dir(waiver_key)
        stage_file = stage_dir / f"{stage}.json"

        # Add metadata
        output = {
            "waiver_key": waiver_key,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        with open(stage_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Update status file
        _update_status(waiver_key, stage, "complete")

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "stage": stage,
            "path": str(stage_file),
            "timestamp": output["timestamp"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def load_stage_output(waiver_key: str, stage: str) -> dict:
    """Load output from a previous pipeline stage.

    Use this to get the results from a prior stage so you can
    continue the pipeline from where it left off.

    Args:
        waiver_key: The waiver identifier
        stage: Stage name to load

    Returns:
        dict with stage data or error
    """
    if stage not in VALID_STAGES:
        return {
            "status": "error",
            "error": f"Invalid stage '{stage}'. Valid stages: {VALID_STAGES}",
        }

    try:
        stage_dir = _get_waiver_stage_dir(waiver_key)
        stage_file = stage_dir / f"{stage}.json"

        if not stage_file.exists():
            return {
                "status": "not_found",
                "waiver_key": waiver_key,
                "stage": stage,
                "message": f"Stage '{stage}' has not been completed yet",
            }

        with open(stage_file) as f:
            output = json.load(f)

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "stage": stage,
            "timestamp": output.get("timestamp"),
            "data": output.get("data", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_pipeline_status(waiver_key: str) -> dict:
    """Get the current pipeline status for a waiver.

    Shows which stages are complete, which are pending, and
    where to restart from.

    Args:
        waiver_key: The waiver identifier

    Returns:
        dict with stage completion status
    """
    try:
        stage_dir = _get_waiver_stage_dir(waiver_key)
        status_file = stage_dir / "status.json"

        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
        else:
            status = {"stages": {}}

        # Check actual files
        completed_stages = []
        pending_stages = []

        for stage in VALID_STAGES:
            stage_file = stage_dir / f"{stage}.json"
            if stage_file.exists():
                completed_stages.append(stage)
            else:
                pending_stages.append(stage)

        # Determine next stage
        next_stage = pending_stages[0] if pending_stages else None

        # Check for images cache
        images_dir = stage_dir / "page_images"
        cached_images = []
        if images_dir.exists():
            cached_images = [f.stem for f in images_dir.glob("*.png")]

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "completed_stages": completed_stages,
            "pending_stages": pending_stages,
            "next_stage": next_stage,
            "cached_page_images": len(cached_images),
            "can_restart_from": completed_stages[-1] if completed_stages else None,
            "last_updated": status.get("last_updated"),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def clear_stage(waiver_key: str, stage: str, clear_subsequent: bool = True) -> dict:
    """Clear a stage to restart from that point.

    By default, also clears all subsequent stages since they depend
    on the cleared stage's output.

    Args:
        waiver_key: The waiver identifier
        stage: Stage to clear
        clear_subsequent: If True, also clear all stages after this one

    Returns:
        dict with cleared stages
    """
    if stage not in VALID_STAGES:
        return {
            "status": "error",
            "error": f"Invalid stage '{stage}'. Valid stages: {VALID_STAGES}",
        }

    try:
        stage_dir = _get_waiver_stage_dir(waiver_key)
        stage_idx = VALID_STAGES.index(stage)

        stages_to_clear = [stage]
        if clear_subsequent:
            stages_to_clear = VALID_STAGES[stage_idx:]

        cleared = []
        for s in stages_to_clear:
            stage_file = stage_dir / f"{s}.json"
            if stage_file.exists():
                stage_file.unlink()
                cleared.append(s)
                _update_status(waiver_key, s, "cleared")

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "cleared_stages": cleared,
            "restart_from": stage,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def save_page_images_cache(waiver_key: str, pdf_path: str, page_numbers: list[int] = None) -> dict:
    """Extract and cache page images from a PDF for later use.

    Caches images per PDF so attempt 3 (image-based extraction) can be
    restarted without re-extracting images from the PDF.

    Args:
        waiver_key: The waiver identifier
        pdf_path: Path to the PDF file
        page_numbers: Optional list of specific pages to cache (1-indexed).
                     If None, caches all pages.

    Returns:
        dict with cached image info
    """
    try:
        import fitz  # PyMuPDF

        images_dir = _get_images_dir(waiver_key)

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if page_numbers is None:
            page_numbers = list(range(1, total_pages + 1))

        cached = []
        for page_num in page_numbers:
            if page_num < 1 or page_num > total_pages:
                continue

            # PyMuPDF uses 0-indexed pages
            page = doc[page_num - 1]

            # Render at 150 DPI for good quality
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)

            # Save as PNG
            image_path = images_dir / f"page_{page_num:03d}.png"
            pix.save(str(image_path))

            cached.append({
                "page": page_num,
                "path": str(image_path),
                "width": pix.width,
                "height": pix.height,
            })

        doc.close()

        # Save cache metadata
        cache_meta = {
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "total_pages": total_pages,
            "cached_pages": [c["page"] for c in cached],
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(images_dir / "cache_meta.json", "w") as f:
            json.dump(cache_meta, f, indent=2)

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "total_pages": total_pages,
            "cached_count": len(cached),
            "cached_pages": [c["page"] for c in cached],
            "cache_dir": str(images_dir),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def load_page_images_cache(waiver_key: str, page_numbers: list[int] = None) -> dict:
    """Load cached page images for a waiver.

    Returns base64-encoded images ready for LLM consumption.

    Args:
        waiver_key: The waiver identifier
        page_numbers: Optional list of specific pages to load (1-indexed).
                     If None, loads all cached pages.

    Returns:
        dict with image data (base64 encoded)
    """
    try:
        images_dir = _get_images_dir(waiver_key)
        meta_file = images_dir / "cache_meta.json"

        if not meta_file.exists():
            return {
                "status": "not_found",
                "waiver_key": waiver_key,
                "message": "No cached images found. Use save_page_images_cache first.",
            }

        with open(meta_file) as f:
            meta = json.load(f)

        if page_numbers is None:
            page_numbers = meta.get("cached_pages", [])

        images = []
        for page_num in page_numbers:
            image_path = images_dir / f"page_{page_num:03d}.png"

            if not image_path.exists():
                continue

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            images.append({
                "page": page_num,
                "media_type": "image/png",
                "base64": base64.b64encode(image_bytes).decode("utf-8"),
            })

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "total_cached": len(meta.get("cached_pages", [])),
            "loaded_count": len(images),
            "images": images,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_cached_image_pages(waiver_key: str) -> dict:
    """Get list of cached page numbers without loading images.

    Useful to check what's cached before deciding which pages to load.

    Args:
        waiver_key: The waiver identifier

    Returns:
        dict with cached page info
    """
    try:
        images_dir = _get_images_dir(waiver_key)
        meta_file = images_dir / "cache_meta.json"

        if not meta_file.exists():
            return {
                "status": "not_found",
                "waiver_key": waiver_key,
                "cached_pages": [],
            }

        with open(meta_file) as f:
            meta = json.load(f)

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "pdf_path": meta.get("pdf_path"),
            "total_pages": meta.get("total_pages"),
            "cached_pages": meta.get("cached_pages", []),
            "cache_timestamp": meta.get("timestamp"),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def _update_status(waiver_key: str, stage: str, state: str):
    """Update the status file for a waiver."""
    stage_dir = _get_waiver_stage_dir(waiver_key)
    status_file = stage_dir / "status.json"

    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
    else:
        status = {"stages": {}}

    status["stages"][stage] = {
        "state": state,
        "timestamp": datetime.utcnow().isoformat(),
    }
    status["last_updated"] = datetime.utcnow().isoformat()

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)


# =============================================================================
# AcroForm Caching - Cache checkbox/form field metadata per PDF
# =============================================================================

def _get_acroform_cache_path(waiver_key: str) -> Path:
    """Get the AcroForm cache file path for a waiver."""
    return _get_waiver_stage_dir(waiver_key) / "acroform_cache.json"


def save_acroform_cache(waiver_key: str, pdf_path: str, acroform_data: dict) -> dict:
    """Cache AcroForm metadata extracted from a PDF.

    This stores checkbox values, form field names, and other AcroForm data
    so it doesn't need to be re-extracted on every pipeline run.

    Args:
        waiver_key: The waiver identifier (e.g., "NV2426R0001")
        pdf_path: Path to the source PDF file
        acroform_data: The extracted AcroForm data including:
            - checkbox_values: dict of field_name -> boolean
            - field_names: list of all form field names
            - page_hints: dict of field_name -> page number
            - raw_form_fields: original form field data

    Returns:
        dict with save status
    """
    try:
        cache_path = _get_acroform_cache_path(waiver_key)

        cache_data = {
            "waiver_key": waiver_key,
            "pdf_path": pdf_path,
            "timestamp": datetime.utcnow().isoformat(),
            "acroform": acroform_data,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)

        checkbox_count = len(acroform_data.get("checkbox_values", {}))
        field_count = len(acroform_data.get("field_names", []))

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "cache_path": str(cache_path),
            "checkbox_count": checkbox_count,
            "field_count": field_count,
            "timestamp": cache_data["timestamp"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def load_acroform_cache(waiver_key: str) -> dict:
    """Load cached AcroForm data for a waiver.

    Returns the previously extracted AcroForm metadata if it exists.
    Use this to skip re-parsing the PDF's form fields.

    Args:
        waiver_key: The waiver identifier

    Returns:
        dict with cached AcroForm data or not_found status
    """
    try:
        cache_path = _get_acroform_cache_path(waiver_key)

        if not cache_path.exists():
            return {
                "status": "not_found",
                "waiver_key": waiver_key,
                "message": "No AcroForm cache found. Extract AcroForm data first.",
            }

        with open(cache_path) as f:
            cache_data = json.load(f)

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "pdf_path": cache_data.get("pdf_path"),
            "timestamp": cache_data.get("timestamp"),
            "acroform": cache_data.get("acroform", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def has_acroform_cache(waiver_key: str) -> dict:
    """Check if AcroForm cache exists for a waiver without loading it.

    Quick check to determine if we need to extract AcroForm data or can
    use cached data.

    Args:
        waiver_key: The waiver identifier

    Returns:
        dict with exists status and cache info
    """
    try:
        cache_path = _get_acroform_cache_path(waiver_key)

        if not cache_path.exists():
            return {
                "status": "success",
                "waiver_key": waiver_key,
                "exists": False,
            }

        # Get basic info without loading full data
        with open(cache_path) as f:
            cache_data = json.load(f)

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "exists": True,
            "timestamp": cache_data.get("timestamp"),
            "pdf_path": cache_data.get("pdf_path"),
            "checkbox_count": len(cache_data.get("acroform", {}).get("checkbox_values", {})),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def extract_and_cache_acroform(waiver_key: str, pdf_path: str, force: bool = False) -> dict:
    """Extract AcroForm data from PDF and cache it.

    Combines extraction and caching in one operation. If cache already exists,
    returns cached data unless force=True.

    Uses the checkbox resolver which has 3-layer resolution:
    1. AcroForm metadata (highest confidence)
    2. Spatial analysis (medium confidence)
    3. None (flags for image OCR)

    Args:
        waiver_key: The waiver identifier
        pdf_path: Path to the PDF file
        force: If True, re-extract even if cache exists

    Returns:
        dict with AcroForm data (from cache or freshly extracted)
    """
    try:
        # Check if cache exists
        if not force:
            cache_check = has_acroform_cache(waiver_key)
            if cache_check.get("exists"):
                # Load and return cached data
                cached = load_acroform_cache(waiver_key)
                cached["from_cache"] = True
                return cached

        # Extract AcroForm data using checkbox resolver
        from .checkbox_tools import resolve_checkboxes

        resolver_result = resolve_checkboxes(pdf_path, waiver_key)

        if resolver_result.get("status") == "error":
            return resolver_result

        # Get the checkbox cache from the resolver
        checkbox_cache = resolver_result.get("cache", {})

        # Build checkbox_values dict: field_name -> 0/1/None
        checkbox_values = {}
        page_hints = {}
        for field_name, field_data in checkbox_cache.items():
            checkbox_values[field_name] = field_data.get("value")
            # Note: page hints aren't in the resolver output, but we store the field names

        # Build cache data structure
        acroform_data = {
            "checkbox_values": checkbox_values,
            "field_names": list(checkbox_values.keys()),
            "page_hints": page_hints,
            "resolution_details": checkbox_cache,  # Full resolver output with method, confidence
            "resolved_acroform": resolver_result.get("resolved_acroform", 0),
            "resolved_spatial": resolver_result.get("resolved_spatial", 0),
            "no_resolution": resolver_result.get("no_resolution", 0),
            "total_checkboxes": resolver_result.get("total_fields", 0),
        }

        # Save to cache
        save_result = save_acroform_cache(waiver_key, pdf_path, acroform_data)

        if save_result.get("status") != "success":
            return save_result

        return {
            "status": "success",
            "waiver_key": waiver_key,
            "from_cache": False,
            "acroform": acroform_data,
            "cache_path": save_result.get("cache_path"),
            "resolved_acroform": acroform_data["resolved_acroform"],
            "resolved_spatial": acroform_data["resolved_spatial"],
            "no_resolution": acroform_data["no_resolution"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
