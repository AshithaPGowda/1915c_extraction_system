"""
ChromaDB storage for critic corrections.

Stores failing field corrections as embeddings for semantic search
and pattern analysis of extraction errors.

Key: For Attempt 2, embeds the VERBATIM paragraph text from evidence
so that future waivers with similar text will match via RAG.
"""

import os
from typing import Any


# ChromaDB path
from ..shared_libraries.paths import CHROMA_DIR
CHROMA_PATH = str(CHROMA_DIR)

COLLECTION_NAME = "critic_corrections"


def _get_collection():
    """
    Get ChromaDB collection, creating it if needed.

    Raises:
        ImportError: If chromadb is not installed
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError(
            "ChromaDB is not installed. Install it with:\n"
            "  pip install chromadb\n"
            "Or add it to your project dependencies."
        )

    # Create persistent client
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Critic corrections from waiver extraction pipeline"}
    )

    return collection


def save_corrections(
    waiver_key: str,
    all_attempt_critiques: list[dict],
    extraction_type: str = "overview"
) -> int:
    """
    Save critic corrections to ChromaDB.

    Embeds each failing field from all attempts as a separate document.
    For Attempt 2, uses the VERBATIM evidence text as the document content
    so that RAG can match similar paragraph text from future waivers.

    Args:
        waiver_key: Normalized waiver key
        all_attempt_critiques: List of critique dicts from every attempt
        extraction_type: Type of extraction (e.g., "overview")

    Returns:
        Number of documents added
    """
    collection = _get_collection()

    # Track which fields eventually passed
    final_critique = all_attempt_critiques[-1] if all_attempt_critiques else {}
    final_passing_fields = {
        f["field"] for f in final_critique.get("passing", [])
    }

    documents = []
    metadatas = []
    ids = []

    for attempt_num, critique in enumerate(all_attempt_critiques, start=1):
        for failing in critique.get("failing", []):
            field_path = failing["field"]
            extracted_val = failing.get("extracted")
            expected_val = failing.get("expected")
            reason = failing.get("reason") or failing.get("critic_reason", "")
            evidence = failing.get("extracted_evidence")

            # Did this field eventually pass?
            resolved = field_path in final_passing_fields

            # Build document text
            # For Attempt 2 (re-extraction), prioritize verbatim evidence for better RAG
            if attempt_num >= 2 and evidence:
                # Use verbatim evidence as primary content for embedding
                # This allows similar paragraph text from other waivers to match
                doc_text = (
                    f"VERBATIM FROM PDF: {evidence}\n\n"
                    f"Field: {field_path}\n"
                    f"Extracted value: {extracted_val}\n"
                    f"Expected value: {expected_val}\n"
                    f"Critic reason: {reason}"
                )
            else:
                # Attempt 1 or no evidence: use standard format
                doc_text = (
                    f"Field: {field_path}\n"
                    f"Extracted: {extracted_val}\n"
                    f"Expected: {expected_val}\n"
                    f"Reason: {reason}"
                )

            # Build metadata
            metadata = {
                "waiver_key": waiver_key,
                "field_path": field_path,
                "attempt_number": attempt_num,
                "resolved": resolved,
                "extraction_type": extraction_type,
                "extracted_value": str(extracted_val)[:500] if extracted_val else "",
                "expected_value": str(expected_val)[:500] if expected_val else "",
                "has_verbatim_evidence": bool(evidence),
            }

            # Add evidence to metadata if available (truncated for storage)
            if evidence:
                metadata["verbatim_evidence"] = str(evidence)[:1000]

            # Unique ID per waiver/field/attempt
            doc_id = f"{waiver_key}_{extraction_type}_{field_path}_{attempt_num}"
            # Sanitize ID (ChromaDB has restrictions)
            doc_id = doc_id.replace(" ", "_").replace(".", "_")[:512]

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Also index low_confidence fields with their evidence
        for low_conf in critique.get("low_confidence", []):
            field_path = low_conf["field"]
            extracted_val = low_conf.get("extracted")
            expected_val = low_conf.get("expected")
            reason = low_conf.get("reason") or low_conf.get("critic_reason", "")
            evidence = low_conf.get("extracted_evidence")

            resolved = field_path in final_passing_fields

            if attempt_num >= 2 and evidence:
                doc_text = (
                    f"VERBATIM FROM PDF (LOW CONFIDENCE): {evidence}\n\n"
                    f"Field: {field_path}\n"
                    f"Extracted value: {extracted_val}\n"
                    f"Expected value: {expected_val}\n"
                    f"Critic reason: {reason}"
                )
            else:
                doc_text = (
                    f"Field (low confidence): {field_path}\n"
                    f"Extracted: {extracted_val}\n"
                    f"Expected: {expected_val}\n"
                    f"Reason: {reason}"
                )

            metadata = {
                "waiver_key": waiver_key,
                "field_path": field_path,
                "attempt_number": attempt_num,
                "resolved": resolved,
                "extraction_type": extraction_type,
                "confidence": "low",
                "extracted_value": str(extracted_val)[:500] if extracted_val else "",
                "expected_value": str(expected_val)[:500] if expected_val else "",
                "has_verbatim_evidence": bool(evidence),
            }

            if evidence:
                metadata["verbatim_evidence"] = str(evidence)[:1000]

            doc_id = f"{waiver_key}_{extraction_type}_{field_path}_lowconf_{attempt_num}"
            doc_id = doc_id.replace(" ", "_").replace(".", "_")[:512]

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)

    if not documents:
        return 0

    # Upsert documents (overwrites if IDs exist)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return len(documents)


def query_similar_corrections(
    query_text: str,
    n_results: int = 10,
    filter_resolved: bool | None = None,
    filter_field: str | None = None
) -> list[dict]:
    """
    Query for similar corrections using semantic search.

    Useful for finding past corrections with similar verbatim text,
    which can inform extraction strategy for new waivers.

    Args:
        query_text: Text to search for similar corrections (e.g., paragraph from PDF)
        n_results: Maximum number of results
        filter_resolved: If set, filter by resolved status
        filter_field: If set, filter by field path

    Returns:
        List of matching correction documents with metadata
    """
    collection = _get_collection()

    where_filter = {}
    if filter_resolved is not None:
        where_filter["resolved"] = filter_resolved
    if filter_field is not None:
        where_filter["field_path"] = filter_field

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter if where_filter else None,
    )

    # Format results
    corrections = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            corrections.append({
                "document": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

    return corrections


def query_by_field(
    field_path: str,
    n_results: int = 20
) -> list[dict]:
    """
    Query all corrections for a specific field path.

    Useful for understanding common extraction errors for a field.

    Args:
        field_path: The field path to query
        n_results: Maximum number of results

    Returns:
        List of correction documents for that field
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[field_path],
        n_results=n_results,
        where={"field_path": field_path},
    )

    corrections = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            corrections.append({
                "document": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

    return corrections


def get_collection_stats() -> dict:
    """Get statistics about the corrections collection."""
    collection = _get_collection()

    return {
        "name": COLLECTION_NAME,
        "count": collection.count(),
    }
