"""
Services Extraction Agent.

Extracts detailed service-level data from 1915(c) HCBS waiver PDFs.
Produces one row per service per waiver with parent/child hierarchy.

This agent:
1. Takes a list of discovered services (from overview extraction or standalone discovery)
2. Extracts each service ONE AT A TIME for accuracy
3. Returns structured data with parent/child relationships
"""

import os
import json
import time
import random
from contextlib import nullcontext
from datetime import datetime, timezone

from google import genai
from google.genai import types

from .services_prompt import (
    SERVICES_DISCOVERY_PROMPT,
    build_service_extraction_prompt,
    build_batch_extraction_prompt,
)
from ..parsers.hcbs_taxonomy import lookup_curated
from ..shared_libraries.file_utils import log_event, get_output_dir
from ..shared_libraries.model_config import get_model_name, get_location


def checkpoint_path(waiver_key: str, name: str) -> str:
    """Path of a per-waiver checkpoint file, e.g. services_attempt1.jsonl."""
    return os.path.join(get_output_dir(waiver_key), name)


def _append_checkpoint(waiver_key: str | None, name: str, record: dict) -> None:
    """Append one JSON line to a checkpoint file. Never raises."""
    if not waiver_key:
        return
    try:
        with open(checkpoint_path(waiver_key, name), "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"    [CHECKPOINT] write failed: {e}")


def get_checkpointed_services(waiver_key: str) -> dict[str, dict]:
    """Read services_attempt1.jsonl and return {service_name: record} for
    every service that has a successful (non-error) checkpointed
    extraction - the last entry per name wins if extracted more than once.

    Shared by both the CLI's --resume (extract_all_services) and the ADK
    orchestrator's resume support (discover_service_list's resume=True),
    so "skip what's already done" means the same thing in both places.
    Returns {} if no checkpoint exists yet or it's empty/unreadable.
    """
    already_done: dict[str, dict] = {}
    cp_path = checkpoint_path(waiver_key, "services_attempt1.jsonl")
    if not os.path.exists(cp_path):
        return already_done
    try:
        with open(cp_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                name = record.get("parent", {}).get("service_name")
                if name and not record.get("parent", {}).get("error"):
                    already_done[name] = record
    except Exception as e:
        print(f"    [RESUME] failed to read checkpoint, ignoring: {e}")
        return {}
    return already_done


# Constants
CONNECTION_RETRY_MAX = int(os.getenv("GEMINI_CONNECTION_RETRY_MAX", "3"))
CONNECTION_RETRY_BASE_DELAY = 5.0
SERVICE_CALL_DELAY_SECONDS = 10.0  # pause between per-service calls to avoid Vertex 429 RESOURCE_EXHAUSTED


def _is_connection_error(error: Exception) -> bool:
    """Check if an error is a connection-level error that should be retried."""
    error_str = str(error).lower()
    if (getattr(error, "code", None) == 504
            or getattr(error, "status", None) == "DEADLINE_EXCEEDED"):
        return True
    connection_patterns = [
        "disconnected",
        "connection reset",
        "connection refused",
        "connection aborted",
        "remote end closed connection",
        "server disconnected",
        "remoteprotocolerror",
        "connectionerror",
        "timeout",
        "timed out",
        "deadline_exceeded",
        # Transient/rate-limit errors: also worth a retry rather than
        # failing the whole call outright (e.g. the critic's body-text
        # verifier used to treat these as "no evidence" and flag every
        # field it was checking - see services_critic_agent.py).
        "resource_exhausted",
        "429",
        "max retries exceeded",
        "nameresolutionerror",
        "failed to resolve",
    ]
    for pattern in connection_patterns:
        if pattern in error_str:
            return True
    try:
        import httpx
        if isinstance(error, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout)):
            return True
    except ImportError:
        pass
    if isinstance(error, (ConnectionError, ConnectionResetError, ConnectionAbortedError)):
        return True
    return False


def _retry_on_connection_error(func, *args, max_retries=CONNECTION_RETRY_MAX, **kwargs):
    """Retry wrapper for connection-level errors with exponential backoff."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not _is_connection_error(e):
                raise
            if attempt < max_retries:
                delay = CONNECTION_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"[RETRY] Connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"[RETRY] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                print(f"[RETRY] All {max_retries + 1} attempts exhausted. Last error: {e}")
                raise
    raise last_exception


CLIENT_TIMEOUT_MS = int(os.getenv("GEMINI_CLIENT_TIMEOUT_MS", "120000"))  # 2 min default - caps a stalled cache/generate call instead of hanging forever; override via .env to avoid DEADLINE_EXCEEDED/504 on slow requests


def _get_client(model: str | None = None):
    """Get Gemini client. Checks GOOGLE_GENAI_USE_VERTEXAI env var.

    `model` picks the Vertex location via get_location() - some models
    (gemini-3.1-pro-preview) are only available on the global endpoint;
    a regional endpoint returns a bare model-not-found error for those,
    not a helpful one. Omit `model` to get the historical us-central1
    default (or LOCATION env override) unchanged."""
    http_options = types.HttpOptions(timeout=CLIENT_TIMEOUT_MS)
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE"
    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location=get_location(model),
            http_options=http_options,
        )
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key, http_options=http_options)


def _load_pdf_bytes(pdf_path: str) -> bytes:
    """Read PDF file as bytes."""
    with open(pdf_path, "rb") as f:
        return f.read()


def _parse_json_response(raw_text: str) -> dict | None:
    """Parse JSON from model response, handling markdown fences."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Use JSONDecoder to parse only the first JSON object
    decoder = json.JSONDecoder()
    text = text.lstrip()
    try:
        result, _ = decoder.raw_decode(text)
        return result
    except json.JSONDecodeError:
        return json.loads(text)


def _call_model(client, pdf_bytes: bytes, prompt: str, model: str | None = None) -> str:
    """Call the Gemini model with PDF and prompt."""
    def _make_request():
        response = client.models.generate_content(
            model=model or get_model_name(),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_bytes,
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,
            ),
        )
        return response.text.strip()

    return _retry_on_connection_error(_make_request)


class PdfCache:
    """
    Explicit Gemini context cache holding one waiver PDF.

    The PDF is uploaded and prefill-processed ONCE; every subsequent call
    references the cache instead of re-sending ~hundreds of pages.

    Policy (per researcher, 2026-07-29): stay on the cached path as long as
    possible. An expired cache is RECREATED, not downgraded. Inline PDF
    bytes are a last resort permitted only for small PDFs
    (< INLINE_LAST_RESORT_MB); for large PDFs a caching failure raises
    instead of silently burning time/money on inline calls.

    Ownership (per researcher, 2026-09-14): one instance is meant to be
    created ONCE per waiver run and shared across every phase that touches
    the PDF (batch extraction, then re-extraction) - see the optional
    `cache=` parameter on extract_all_services()/run_services_reextraction().
    `__exit__` deletes the underlying cache, so only the outermost `with`
    block for a waiver should own it; a phase handed an existing `cache`
    must not open its own.

    Failure handling is deliberately narrow: only a server-confirmed "this
    cache resource doesn't exist" response (status NOT_FOUND / code 404)
    is treated as real expiry and triggers a recreate. Everything else -
    429s, DEADLINE_EXCEEDED, transport errors, the transient "insufficient
    token count" race Vertex sometimes returns at creation time for large
    multi-page PDFs (confirmed transient: cross-referencing
    analytics_agent/logs shows the identical, unchanged file succeeding on
    other attempts, 2026-09-14) - leaves cache_name untouched. A call that
    fails for one of those reasons falls back to a one-off inline request
    for THAT call only (small PDFs), so a still-valid cache isn't discarded
    over an unrelated, transient error.

    Usage:
        with PdfCache(client, pdf_path, waiver_key) as cache:
            text = cache.call(prompt)
    """

    TTL = "3600s"  # 1 h - a 24-service pass at ~75s/call needs well over 30 min
    MAX_CACHE_ATTEMPTS = 5  # total create/recreate attempts per instance
    # Short, bounded retry specifically for the known-transient Vertex
    # ingestion race (see class docstring) - not a general retry-on-failure.
    TRANSIENT_CREATE_RETRY_DELAY_S = 3
    INLINE_LAST_RESORT_MB = 8  # inline only ever permitted below this size
    # Renew (extend TTL in place, same cache_name) instead of letting a long
    # run hit real expiry - checked before every call, not reactively.
    RENEW_MARGIN_S = 300

    def __init__(
        self,
        client,
        pdf_path: str,
        waiver_key: str | None = None,
        ttl_seconds: int | None = None,
        model: str | None = None,
        cache_only: bool = False,
    ):
        self.cache_only = cache_only
        self.client = client
        self.pdf_path = pdf_path
        self.pdf_bytes = _load_pdf_bytes(pdf_path)
        self.waiver_key = waiver_key
        self.model = model or get_model_name()
        if ttl_seconds is not None:
            self.TTL = f"{int(ttl_seconds)}s"  # instance override of the class default
        self.cache_name: str | None = None
        self.create_time = None
        self.expire_time = None
        self.token_count = None
        self._attempts = 0
        self._last_creation_error: Exception | None = None

    @property
    def _inline_allowed(self) -> bool:
        return not self.cache_only and len(self.pdf_bytes) < self.INLINE_LAST_RESORT_MB * 1024 * 1024

    def _create_cache(self, part: "types.Part"):
        """One unretried create call. Returns the full CachedContent object
        (not just its name) so create_time/expire_time/usage_metadata are
        available to record - see MEMORY point 4."""
        return self.client.caches.create(
            model=self.model,
            config=types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=[part])],
                ttl=self.TTL,
                display_name=self.waiver_key or "waiver-pdf",
            ),
        )

    def _gcs_uri(self) -> str:
        """
        Upload the PDF to GCS (once - reused across runs) and return its URI.
        Vertex requires large cached content to come from GCS rather than
        inline bytes. The bucket is created lazily on first use.
        """
        from google.cloud import storage

        bucket_name = os.getenv(
            "CACHE_GCS_BUCKET",
            f"{os.getenv('PROJECT_ID', 'waiver')}-waiver-pdf-cache",
        )
        import hashlib

        digest = hashlib.sha256(self.pdf_bytes).hexdigest()
        blob_name = f"pdf_cache/{digest}/{os.path.basename(self.pdf_path)}"

        gcs = storage.Client(project=os.getenv("PROJECT_ID"))
        bucket = gcs.lookup_bucket(bucket_name)
        if bucket is None:
            bucket = gcs.create_bucket(
                bucket_name, location=os.getenv("LOCATION", "us-central1")
            )
            print(f"    [CACHE] Created GCS bucket gs://{bucket_name}")
        blob = bucket.blob(blob_name)
        if not blob.exists():
            blob.upload_from_string(
                self.pdf_bytes, content_type="application/pdf"
            )
        return f"gs://{bucket_name}/{blob_name}"

    def _attempt_create(self):
        """One attempt at creating the cache via whichever path applies to
        this PDF's size. Returns the CachedContent object or raises."""
        if self._inline_allowed:
            try:
                return self._create_cache(
                    types.Part(inline_data=types.Blob(
                        mime_type="application/pdf",
                        data=self.pdf_bytes,
                    ))
                )
            except Exception as e:
                if "data size limit" not in str(e).lower():
                    raise
                # PDF too large for inline caching - go through GCS instead.
                uri = self._gcs_uri()
                cache = self._create_cache(
                    types.Part(file_data=types.FileData(
                        file_uri=uri, mime_type="application/pdf",
                    ))
                )
                print(f"    [CACHE] Cached via GCS ({uri})")
                return cache
        else:
            # PDF >= INLINE_LAST_RESORT_MB: skip the inline attempt
            # entirely and go straight through GCS.
            uri = self._gcs_uri()
            cache = self._create_cache(
                types.Part(file_data=types.FileData(
                    file_uri=uri, mime_type="application/pdf",
                ))
            )
            print(f"    [CACHE] Cached via GCS ({uri})")
            return cache

    def _establish_cache(self) -> None:
        """(Re)create the cache, bounded by MAX_CACHE_ATTEMPTS for this
        instance's whole lifetime. Retries once, briefly, on the known
        transient "insufficient token count" signature (see class
        docstring) before counting the attempt as failed. Never raises for
        a small PDF - leaves cache_name as None and records
        _last_creation_error for the caller to decide what to do next.
        """
        while self._attempts < self.MAX_CACHE_ATTEMPTS:
            self._attempts += 1
            try:
                cache = self._attempt_create()
                self.cache_name = cache.name
                self.create_time = cache.create_time
                self.expire_time = cache.expire_time
                self.token_count = (
                    cache.usage_metadata.total_token_count
                    if cache.usage_metadata else None
                )
                self._last_creation_error = None
                if self.waiver_key:
                    log_event(self.waiver_key, {
                        "agent": "services_agent",
                        "action": "pdf_cache_created",
                        "cache_name": self.cache_name,
                        "attempt": self._attempts,
                        "create_time": str(self.create_time),
                        "expire_time": str(self.expire_time),
                        "cached_content_token_count": self.token_count,
                    })
                return
            except Exception as e:
                self._last_creation_error = e
                transient = "minimum token count" in str(e).lower()
                if self.waiver_key:
                    log_event(self.waiver_key, {
                        "agent": "services_agent",
                        "action": "pdf_cache_create_failed",
                        "attempt": self._attempts,
                        "code": getattr(e, "code", None),
                        "status": getattr(e, "status", None),
                        "transient_signature": transient,
                        "error": str(e),
                    })
                if transient and self._attempts < self.MAX_CACHE_ATTEMPTS:
                    print(f"    [CACHE] Create attempt {self._attempts} hit "
                          "the known transient token-count race - retrying "
                          f"in {self.TRANSIENT_CREATE_RETRY_DELAY_S}s...")
                    time.sleep(self.TRANSIENT_CREATE_RETRY_DELAY_S)
                    continue
                break
        self.cache_name = None

    def _maybe_renew(self) -> None:
        """Extend the existing cache's TTL in place (same cache_name) if it's
        within RENEW_MARGIN_S of expiring - checked before every request so
        a long run never hits real expiry in the first place. Uses
        caches.update(), a real, distinct SDK call from caches.create() that
        does not change the resource's identity. If the update itself 404s,
        the cache is genuinely gone; fall through to the normal
        recreate-on-NOT_FOUND path in call() rather than raising here. Any
        other update failure (timeout, 429) is treated like any other
        transient error - logged, cache_name left untouched, the caller
        proceeds against the still-presumed-valid cache."""
        if not self.cache_name or not self.expire_time:
            return
        try:
            remaining = (self.expire_time - datetime.now(timezone.utc)).total_seconds()
        except TypeError:
            return
        if remaining >= self.RENEW_MARGIN_S:
            return
        try:
            updated = self.client.caches.update(
                name=self.cache_name,
                config=types.UpdateCachedContentConfig(ttl=self.TTL),
            )
            self.expire_time = updated.expire_time
            if self.waiver_key:
                log_event(self.waiver_key, {
                    "agent": "services_agent",
                    "action": "pdf_cache_renewed",
                    "cache_name": self.cache_name,
                    "new_expire_time": str(self.expire_time),
                })
        except Exception as e:
            status = getattr(e, "status", None)
            code = getattr(e, "code", None)
            message = str(e).lower()
            cache_missing = (status == "NOT_FOUND" or code == 404) and (
                self.cache_name.lower() in message
                or "cachedcontent" in message
                or "cached content" in message
                or "cached_content" in message
            )
            if self.waiver_key:
                log_event(self.waiver_key, {
                    "agent": "services_agent",
                    "action": "pdf_cache_renew_failed",
                    "cache_name": self.cache_name,
                    "code": code,
                    "status": status,
                    "cache_missing": cache_missing,
                    "error": str(e),
                })
            if cache_missing:
                # Genuinely gone - let the next call() attempt discover this
                # via the normal NOT_FOUND path rather than duplicating the
                # recreate logic here.
                self.cache_name = None
            # Any other renew failure: leave cache_name as-is, transient.

    def __enter__(self):
        self._establish_cache()
        if self.cache_name is None:
            if self._inline_allowed:
                print(f"    [CACHE] Caching unavailable "
                      f"({self._last_creation_error}); small PDF - inline "
                      "permitted as last resort")
            else:
                # Large PDF with no cache: fail loudly instead of silently
                # running the whole waiver on the slow/expensive inline path.
                raise RuntimeError(
                    f"Context caching unavailable for large PDF "
                    f"({len(self.pdf_bytes) / 1048576:.1f} MB) after "
                    f"{self._attempts} attempt(s) - refusing inline "
                    f"fallback: {self._last_creation_error}"
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache_name:
            try:
                self.client.caches.delete(name=self.cache_name)
            except Exception:
                pass
            if self.waiver_key:
                log_event(self.waiver_key, {
                    "agent": "services_agent",
                    "action": "pdf_cache_deleted",
                    "cache_name": self.cache_name,
                })
        return False

    def generate(self, contents: list):
        """Low-level, multi-turn-capable version of call(): takes an
        arbitrary `contents` list (e.g. a truncation-continuation retry
        that appends model/user turns) instead of a single text prompt,
        and returns the full response object rather than just `.text` -
        callers needing usage_metadata or to keep building on the
        conversation (reextraction_agent's text path) use this directly.

        Same renew-before-request and NOT_FOUND-only-recreate protections
        as call(). Unlike call(), this has NO inline-fallback path (a
        multi-turn conversation can't be redirected to the single-prompt
        _call_model helper) - it raises the underlying error on anything
        it can't recover by recreating a confirmed-missing cache, and
        leaves `cache_name` untouched on any other error so the NEXT call
        still tries the (presumably still valid) cache fresh."""
        if not self.cache_name:
            if self._attempts < self.MAX_CACHE_ATTEMPTS:
                self._establish_cache()
            if not self.cache_name:
                raise RuntimeError(
                    f"Context caching unavailable after {self._attempts} "
                    f"attempt(s): {self._last_creation_error}"
                )

        def _make_request():
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=65536,
                    cached_content=self.cache_name,
                ),
            )
            # Real proof of cache reuse for THIS call - creation-time token
            # count only proves the cache was big enough, not that a later
            # call actually hit it.
            usage = getattr(response, "usage_metadata", None)
            if usage is not None and self.waiver_key:
                log_event(self.waiver_key, {
                    "agent": "services_agent",
                    "action": "pdf_cache_generate",
                    "cache_name": self.cache_name,
                    "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
                    "total_token_count": getattr(usage, "total_token_count", None),
                })
            return response

        self._maybe_renew()
        if not self.cache_name:
            self._establish_cache()
            if not self.cache_name:
                raise RuntimeError("Unable to restore PDF cache after renewal")
        cache_miss_retries = 0

        # Replacement requests use the same error handling and bounded
        # creation budget as the original request.
        while True:
            try:
                return _retry_on_connection_error(_make_request)
            except Exception as e:
                code = getattr(e, "code", None)
                status = getattr(e, "status", None)
                # A missing model/project is not a missing PDF cache. Require
                # both the API status and a reference to cached content.
                message = str(e).lower()
                cache_missing = (status == "NOT_FOUND" or code == 404) and (
                    self.cache_name.lower() in message
                    or "cachedcontent" in message
                    or "cached content" in message
                    or "cached_content" in message
                )
                age_s = None
                if self.create_time:
                    try:
                        age_s = (datetime.now(timezone.utc) - self.create_time).total_seconds()
                    except TypeError:
                        age_s = None
                if self.waiver_key:
                    log_event(self.waiver_key, {
                        "agent": "services_agent",
                        "action": "pdf_cache_call_failed",
                        "cache_name": self.cache_name,
                        "cache_age_s": age_s,
                        "code": code,
                        "status": status,
                        "cache_missing": cache_missing,
                        "error": str(e),
                    })

                if cache_missing and self.cache_only and cache_miss_retries < 2:
                    # A transient lookup miss does not invalidate a live cache.
                    cache_miss_retries += 1
                    time.sleep(self.TRANSIENT_CREATE_RETRY_DELAY_S)
                    continue

                if cache_missing:
                    # Server-confirmed the cache is genuinely gone (real expiry
                    # or otherwise) - safe to recreate and stay on the cached
                    # path, keeping the same instance/attempt budget.
                    self.cache_name = None
                    if self._attempts < self.MAX_CACHE_ATTEMPTS:
                        age_desc = f"age {age_s:.0f}s" if age_s is not None else "age unknown"
                        print(f"    [CACHE] Cache not found ({age_desc}) - "
                              f"recreating (attempt {self._attempts + 1}"
                              f"/{self.MAX_CACHE_ATTEMPTS})...")
                        self._establish_cache()
                        if self.cache_name:
                            continue
                # Anything else - a 429/504/timeout already exhausted by
                # _retry_on_connection_error, or recreate attempts exhausted -
                # is not evidence the cache itself is invalid. Don't discard a
                # possibly-still-valid cache_name over it - re-raise as-is;
                # call() decides whether an inline fallback applies, and
                # multi-turn callers of generate() directly must handle this
                # themselves (no single-prompt inline path for those).
                raise

    def call(self, prompt: str) -> str:
        """Single-prompt convenience wrapper over generate(), adding the
        small-PDF inline-fallback path generate() doesn't support (a
        multi-turn conversation can't be redirected to the single-prompt
        _call_model helper, so that path only exists here)."""
        if not self.cache_name:
            if self._attempts < self.MAX_CACHE_ATTEMPTS:
                self._establish_cache()
            if not self.cache_name:
                if self._inline_allowed:
                    return _call_model(self.client, self.pdf_bytes, prompt, model=self.model)
                raise RuntimeError(
                    f"Context caching unavailable for large PDF after "
                    f"{self._attempts} attempt(s) - refusing inline "
                    f"fallback: {self._last_creation_error}"
                )
        try:
            response = self.generate([types.Content(role="user", parts=[types.Part(text=prompt)])])
            return response.text.strip()
        except Exception as e:
            if self._inline_allowed:
                print(f"    [CACHE] Cached call failed ({e}); "
                      "small PDF - inline for this call only")
                return _call_model(self.client, self.pdf_bytes, prompt, model=self.model)
            raise RuntimeError(
                f"Cached call failed for large PDF and inline fallback is "
                f"disabled: {e}"
            )


def discover_services(
    pdf_path: str,
    waiver_key: str | None = None,
    model: str | None = None,
    cache=None,
) -> dict:
    """
    Discover all services in a waiver PDF.

    This is used when we don't have discovered_services from the overview extraction.

    Args:
        pdf_path: Path to the PDF file
        waiver_key: Optional waiver key for logging

    Returns:
        Dict with 'services' list and appendix page ranges
    """
    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_agent",
            "action": "discovery_start",
            "pdf_path": pdf_path,
        })

    try:
        if cache is not None:
            raw_text = cache.call(SERVICES_DISCOVERY_PROMPT)
        else:
            raw_text = _call_model(
                _get_client(model), _load_pdf_bytes(pdf_path),
                SERVICES_DISCOVERY_PROMPT, model=model,
            )
        result = _parse_json_response(raw_text)

        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_agent",
                "action": "discovery_complete",
                "services_found": len(result.get("services", [])),
            })

        return result

    except Exception as e:
        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_agent",
                "action": "discovery_error",
                "error": str(e),
            })
        raise


def extract_single_service(
    pdf_path: str,
    service_name: str,
    state: str,
    waiver_title: str,
    file_name: str,
    appendix_hints: dict | None = None,
    waiver_key: str | None = None,
    cache: "PdfCache | None" = None,
    guidance: str | None = None,
) -> dict:
    """
    Extract detailed data for a single service.

    Args:
        pdf_path: Path to the PDF file
        service_name: The exact service name to extract
        state: The state name
        waiver_title: The waiver title
        file_name: The PDF filename
        appendix_hints: Optional dict with page hints for appendices
        waiver_key: Optional waiver key for logging
        cache: Optional PdfCache so the PDF is uploaded/processed only once
            per waiver instead of once per service
        guidance: Optional free-text steer for this one attempt (see
            build_service_extraction_prompt) - for manually retrying/
            correcting one named service, not the normal discovery loop.

    Returns:
        Dict with 'parent' and 'children' data
    """
    prompt = build_service_extraction_prompt(
        service_name=service_name,
        state=state,
        waiver_title=waiver_title,
        file_name=file_name,
        appendix_hints=appendix_hints,
        guidance=guidance,
    )

    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_agent",
            "action": "extraction_start",
            "service_name": service_name,
        })

    try:
        if cache is not None:
            raw_text = cache.call(prompt)
        else:
            client = _get_client()
            pdf_bytes = _load_pdf_bytes(pdf_path)
            raw_text = _call_model(client, pdf_bytes, prompt)
        result = _parse_json_response(raw_text)

        children_count = len(result.get("children", []))

        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_agent",
                "action": "extraction_complete",
                "service_name": service_name,
                "has_parent": "parent" in result,
                "children_count": children_count,
            })

        return result

    except Exception as e:
        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_agent",
                "action": "extraction_error",
                "service_name": service_name,
                "error": str(e),
            })
        raise


def extract_all_services(
    pdf_path: str,
    services: list[str],
    state: str,
    waiver_title: str,
    file_name: str,
    waiver_key: str | None = None,
    one_at_a_time: bool = True,
    appendix_hints: dict | None = None,
    resume: bool = False,
    cache: "PdfCache | None" = None,
) -> list[dict]:
    """
    Extract data for all services in a waiver.

    Args:
        pdf_path: Path to the PDF file
        services: List of service names to extract
        state: The state name
        waiver_title: The waiver title
        file_name: The PDF filename
        waiver_key: Optional waiver key for logging
        one_at_a_time: If True, extract each service separately (more accurate)
        appendix_hints: Optional dict with page hints for appendices
        resume: If True, skip services already present (without an error)
            in this waiver's services_attempt1.jsonl checkpoint from a
            prior killed/crashed run, extracting only the remainder. New
            results are appended to that same checkpoint instead of
            truncating it. No effect if no checkpoint exists yet.
        cache: Optional PdfCache already owned by the caller (e.g. the
            waiver pipeline), shared across this batch AND the later
            re-extraction pass so the PDF is only ever cached once per
            waiver run. When omitted, a cache scoped to just this call is
            created and deleted here, as before.

    Returns:
        List of service dicts, each with 'parent' and 'children'
    """
    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_agent",
            "action": "batch_extraction_start",
            "services_count": len(services),
            "one_at_a_time": one_at_a_time,
        })

    results = []

    if one_at_a_time:
        # Extract each service separately (recommended for accuracy).
        # The PDF is cached server-side once and reused for every call.
        # Each completed service is appended to a crash-safe checkpoint so
        # a failed run can be rebuilt without re-extracting.
        already_done: dict[str, dict] = {}
        if waiver_key:
            cp_path = checkpoint_path(waiver_key, "services_attempt1.jsonl")
            if resume and os.path.exists(cp_path):
                already_done = get_checkpointed_services(waiver_key)
                if already_done:
                    print(f"    [RESUME] {len(already_done)} service(s) already "
                          "checkpointed, skipping re-extraction for those")
            else:
                # Fresh run (or --resume with nothing to resume from): start
                # the checkpoint file over.
                try:
                    open(cp_path, "w").close()
                except Exception:
                    pass
        # If the caller already owns a cache (shared across this batch and
        # the later re-extraction pass), use it as-is via nullcontext -
        # we must NOT __enter__/__exit__ a cache we don't own. Otherwise
        # fall back to creating one scoped to just this call, as before.
        cache_ctx = (
            nullcontext(cache) if cache is not None
            else PdfCache(_get_client(), pdf_path, waiver_key)
        )
        with cache_ctx as active_cache:
            for i, service_name in enumerate(services, 1):
                if service_name in already_done:
                    print(f"    [{i}/{len(services)}] Skipping (already done): {service_name}")
                    results.append(already_done[service_name])
                    continue
                print(f"    [{i}/{len(services)}] Extracting: {service_name}")
                try:
                    service_data = extract_single_service(
                        pdf_path=pdf_path,
                        service_name=service_name,
                        state=state,
                        waiver_title=waiver_title,
                        file_name=file_name,
                        appendix_hints=appendix_hints,
                        waiver_key=waiver_key,
                        cache=active_cache,
                    )
                    results.append(service_data)
                except Exception as e:
                    print(f"    Error extracting {service_name}: {e}")
                    results.append({
                        "parent": {
                            "row_type": "parent",
                            "service_name": service_name,
                            "error": str(e),
                        },
                        "children": [],
                    })
                _append_checkpoint(waiver_key, "services_attempt1.jsonl", results[-1])
                if i < len(services):
                    time.sleep(SERVICE_CALL_DELAY_SECONDS)
    else:
        # Batch extraction (faster but may lose accuracy for complex waivers)
        client = _get_client()
        pdf_bytes = _load_pdf_bytes(pdf_path)

        prompt = build_batch_extraction_prompt(
            services=services,
            state=state,
            waiver_title=waiver_title,
            file_name=file_name,
        )

        try:
            raw_text = _call_model(client, pdf_bytes, prompt)
            batch_result = _parse_json_response(raw_text)
            results = batch_result.get("services", [])
        except Exception as e:
            if waiver_key:
                log_event(waiver_key, {
                    "agent": "services_agent",
                    "action": "batch_extraction_error",
                    "error": str(e),
                })
            raise

    if waiver_key:
        total_children = sum(len(r.get("children", [])) for r in results)
        log_event(waiver_key, {
            "agent": "services_agent",
            "action": "batch_extraction_complete",
            "services_extracted": len(results),
            "total_children": total_children,
        })

    return results


# HCBS Taxonomy in the reference sheet's vocabulary, with example services
# drawn from the researcher's curated crosswalk. Based on the official
# taxonomy (Peebles & Bohl, MMRR 2014, Exhibit 1; TMSIS COT003188).
HCBS_TAXONOMY_CATEGORIES = [
    ("Case management", "supports coordination, community health worker"),
    ("Round-the-clock services",
     "residential habilitation, group living, shared living, supported living"),
    ("Supported employment",
     "employment support, job coach, job developer, assessment/discovery, benefits counseling, supported employment transportation"),
    ("Day services",
     "day habilitation, prevocational services, adult day health, community experience, community integration, non-vocational habilitation"),
    ("Nursing services", "private duty nursing, skilled nursing, RN/LPN nursing"),
    ("Home delivered meals", "home delivered meals"),
    ("Home-based services",
     "personal care, attendant services, companion, homemaker, chore, community living supports, remote supports, in-home therapies (occupational/physical/speech/massage), overnight health and safety support"),
    ("Caregiver support services",
     "respite (in-home/out-of-home/per-diem), family home care training, caregiver counseling"),
    ("Support partic direct",
     "individual directed goods and services, fiscal intermediary, financial management services, information and assistance for participant direction"),
    ("Participant training", "participant training, positive behavior support"),
    ("Tech mods equip",
     "assistive technology, environmental accessibility adaptations, specialized medical equipment and supplies, PERS, vehicle modifications, durable medical equipment, pest control"),
    ("Non-medical transportation", "enhanced/community transportation"),
    ("Community transition svcs",
     "community transition services, housing stabilization"),
    ("Other", "crisis intervention, counseling, interpreter - LAST RESORT ONLY"),
]


def infer_taxonomy_fallback(
    services_data: list[dict],
    waiver_key: str | None = None,
    model: str | None = None,
) -> int:
    """
    Classify services whose HCBS Taxonomy table is empty in the PDF into an
    official CMS taxonomy category (the same crosswalk the reference sheet's
    researchers applied manually).

    Only touches parents with hcbs_taxonomy None. The inferred value is the
    official category label; the evidence dict records that it was inferred,
    not read from the document.

    Returns the number of services classified.
    """
    pending = []
    for idx, service in enumerate(services_data):
        parent = service.get("parent", {})
        if parent.get("service_name") and parent.get("hcbs_taxonomy") is None:
            pending.append((idx, parent["service_name"]))
    if not pending:
        return 0

    category_lines = "\n".join(
        f"{i+1}. {name} (e.g. {examples})"
        for i, (name, examples) in enumerate(HCBS_TAXONOMY_CATEGORIES)
    )
    items = [{"id": i, "service_name": name} for i, (_, name) in enumerate(pending)]
    prompt = f"""You classify Medicaid 1915(c) HCBS waiver services into the official
CMS HCBS Taxonomy categories (Truven/Mathematica, MMRR 2014).

## Categories (in taxonomy order, with example services)
{category_lines}

## Rules
1. If a service could fit more than one category, assign the category that
   comes FIRST in the list above (this is the official taxonomy tie-break).
2. Use "Other" ONLY when no other category can reasonably apply. A service
   with any equipment/supply/modification character belongs in
   "Tech mods equip"; anything delivered in the home belongs in
   "Home-based services" before it belongs in "Other".

## Services to classify
{json.dumps(items, indent=1)}

Return ONLY a JSON array: [{{"id": 0, "category": "<exact category name from the list>"}}]
Use the category names EXACTLY as written above.
"""

    valid = {name for name, _ in HCBS_TAXONOMY_CATEGORIES}
    try:
        client = _get_client(model)
        response = client.models.generate_content(
            model=model or get_model_name(),
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
            ),
        )
        verdicts = _parse_json_response(response.text.strip())
        if isinstance(verdicts, dict):
            verdicts = verdicts.get("classifications", [])
        by_id = {v.get("id"): v.get("category") for v in verdicts}
    except Exception as e:
        if waiver_key:
            log_event(waiver_key, {
                "agent": "services_agent",
                "action": "taxonomy_inference_error",
                "error": str(e),
            })
        return 0

    classified = 0
    for i, (idx, name) in enumerate(pending):
        category = by_id.get(i)
        if category not in valid:
            continue
        parent = services_data[idx]["parent"]
        parent["hcbs_taxonomy"] = category
        evidence = dict(parent.get("evidence") or {})
        evidence["hcbs_taxonomy"] = (
            "INFERRED from CMS HCBS Taxonomy crosswalk (MMRR 2014 Exhibit 1) - "
            "the PDF's taxonomy table is empty for this service"
        )
        parent["evidence"] = evidence
        classified += 1

    if waiver_key:
        log_event(waiver_key, {
            "agent": "services_agent",
            "action": "taxonomy_inferred",
            "services_classified": classified,
        })
    return classified


def flatten_services_to_rows(
    services_data: list[dict],
    waiver_key: str,
    state: str,
    waiver_title: str,
    file_name: str,
) -> list[dict]:
    """
    Flatten service data into rows suitable for CSV/SQLite export.

    Converts parent/child structure to flat rows where:
    - Parent rows have all fields populated
    - Child rows have empty state/waiver/file_name and inherit from parent

    Args:
        services_data: List of service dicts with 'parent' and 'children'
        waiver_key: The normalized waiver key
        state: The state name
        waiver_title: The waiver title
        file_name: The PDF filename

    Returns:
        List of flat row dicts ready for storage
    """
    rows = []
    row_index = 0

    for service in services_data:
        parent = service.get("parent", {})

        # Parent row
        parent_row = {
            "waiver_key": waiver_key,
            "state": state,
            "waiver_title": waiver_title,
            "file_name": file_name,
            "row_type": "parent",
            "parent_service": None,
            "service_name": parent.get("service_name"),
            "hcbs_taxonomy": parent.get("hcbs_taxonomy"),
            "service_type": parent.get("service_type"),
            "delivery_method": parent.get("service_delivery_method"),
            "provider_eligibility": parent.get("provider_eligibility"),
            "provider_specifications": parent.get("provider_specifications"),
            "self_direction_authority": parent.get("self_direction_authority"),
            "information_assistance": parent.get("information_assistance"),
            "row_index": row_index,
        }
        for year in range(1, 6):
            for suffix in ("capitation", "unit", "users", "avg_units_per_user",
                           "avg_cost_per_unit", "component_cost", "total_cost"):
                key = f"year_{year}_{suffix}"
                parent_row[key] = parent.get(key)
        rows.append(parent_row)
        row_index += 1

        # Child rows
        parent_service_name = parent.get("service_name")
        for child in service.get("children", []):
            child_row = {
                "waiver_key": waiver_key,
                "state": None,  # Empty for child rows
                "waiver_title": None,
                "file_name": None,
                "row_type": "child",
                "parent_service": parent_service_name,
                "service_name": child.get("service_name"),
                # Child taxonomy: curated crosswalk on the child's own name
                # first, else inherit the parent's resolved taxonomy.
                "hcbs_taxonomy": (
                    lookup_curated(child.get("service_name"))
                    or parent.get("hcbs_taxonomy")
                ),
                "service_type": None,  # Empty for child rows
                "delivery_method": None,
                "provider_eligibility": None,
                "provider_specifications": None,
                "self_direction_authority": None,
                "information_assistance": None,
                "row_index": row_index,
            }
            for year in range(1, 6):
                child_row[f"year_{year}_capitation"] = None
                for suffix in ("unit", "users", "avg_units_per_user",
                               "avg_cost_per_unit", "component_cost"):
                    key = f"year_{year}_{suffix}"
                    child_row[key] = child.get(key)
                child_row[f"year_{year}_total_cost"] = child.get(
                    f"year_{year}_total_cost", "N/A"
                )
            rows.append(child_row)
            row_index += 1

    return rows
