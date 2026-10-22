"""Read-only MongoDB query execution.

Executes only validated find() filters against a MongoDB collection and returns
the matching documents plus execution metadata.  No write operations, no
arbitrary JavaScript, and a hard result limit are enforced.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from bson import ObjectId
from pymongo.collection import Collection

from backend.database.mongodb import get_database


MAX_LIMIT = 50


def _sanitize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a MongoDB document into a JSON-serializable dict.

    ObjectId values are converted to strings so the frontend can render them.
    """
    sanitized: Dict[str, Any] = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def execute_filter(
    filter_dict: Dict[str, Any],
    *,
    collection_name: str = "students",
    limit: int = MAX_LIMIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a read-only find() filter and return (results, metadata).

    Parameters
    ----------
    filter_dict:
        A validated MongoDB filter dict.
    collection_name:
        Target collection (default ``students``).
    limit:
        Maximum number of documents to return (capped at ``MAX_LIMIT``).

    Returns
    -------
    (results, metadata) where results is a list of sanitized documents and
    metadata contains count, limit, collection, and execution time in ms.
    """
    if limit <= 0:
        limit = MAX_LIMIT
    limit = min(limit, MAX_LIMIT)

    db = get_database()
    collection: Collection = db[collection_name]

    start = time.perf_counter()
    cursor = collection.find(filter_dict).limit(limit)
    results = [_sanitize_document(doc) for doc in cursor]
    elapsed_ms = (time.perf_counter() - start) * 1000

    metadata = {
        "collection": collection_name,
        "filter": filter_dict,
        "limit": limit,
        "returned": len(results),
        "execution_time_ms": round(elapsed_ms, 2),
    }
    return results, metadata


def test_filter_syntax(filter_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Best-effort syntax check without executing against the database.

    Confirms the filter is a non-empty dict.  Returns (True, "") when sane.
    """
    if not isinstance(filter_dict, dict) or not filter_dict:
        return False, "Filter is empty or not an object."
    return True, ""
