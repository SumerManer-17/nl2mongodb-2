"""Validation and normalization for LLM-generated MongoDB filters.

The validator guarantees that only safe, read-only MongoDB find() filters are
executed against the database.  It is intentionally strict: unknown fields or
operators cause rejection with a clear error rather than silent execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Operators permitted inside a field condition (e.g. {"gpa": {"$gt": 8}}).
_FIELD_OPERATORS = {"$gt", "$gte", "$lt", "$lte", "$eq", "$ne", "$in", "$nin", "$exists"}

# Logical operators permitted at the top level (and nested inside $and/$or).
_LOGICAL_OPERATORS = {"$and", "$or"}

# Write / dangerous operations that must never be generated.
_FORBIDDEN_KEYS = {
    "$set", "$unset", "$inc", "$push", "$pull", "$addToSet",
    "insert", "update", "delete", "drop", "remove", "replace",
    "$where", "$eval", "$func",
}


class FilterValidationError(ValueError):
    """Raised when a generated filter is invalid or unsafe."""


def _normalize_operator_key(key: str) -> str:
    """Map common non-dollar operator spellings to MongoDB operators.

    ``gt`` -> ``$gt``, ``gte`` -> ``$gte``, etc.  Keys already starting with
    ``$`` are returned unchanged.  Unknown keys are returned unchanged so the
    validator can reject them explicitly.
    """
    if key.startswith("$"):
        return key

    mapping = {
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "eq": "$eq",
        "ne": "$ne",
        "in": "$in",
        "nin": "$nin",
        "and": "$and",
        "or": "$or",
    }
    return mapping.get(key, key)


def _resolve_field_path(raw_field: str, schema_fields: Dict[str, Any]) -> str:
    """Resolve a raw field name to a canonical schema field path.

    Matching is case-insensitive so model output like ``GPA`` maps to ``gpa``.
    Returns the canonical path from the schema, or raises if no match exists.
    """
    if raw_field in schema_fields:
        return raw_field

    # Array-element paths like "skills[]" resolve to the array field "skills"
    # because MongoDB queries arrays directly on the field name.
    if raw_field.endswith("[]"):
        base = raw_field[:-2]
        if base in schema_fields:
            return base
        lowered = base.lower()
        for path in schema_fields:
            if path.lower() == lowered:
                return path

    lowered = raw_field.lower()
    for path in schema_fields:
        if path.lower() == lowered:
            return path

    # Handle dotted paths field-by-field (e.g. "Academic.GPA" -> "academic.gpa").
    if "." in raw_field:
        parts = raw_field.split(".")
        resolved: List[str] = []
        for part in parts:
            match = None
            for path in schema_fields:
                if path.lower() == part.lower():
                    match = path
                    break
            if match is None:
                raise FilterValidationError(
                    f"Unknown field path '{raw_field}' (segment '{part}' not in schema)."
                )
            resolved.append(match)
        return ".".join(resolved)

    raise FilterValidationError(
        f"Unknown field '{raw_field}'. Supported fields: {sorted(schema_fields)}"
    )


def normalize_filter(
    raw: Any, schema_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize and validate a raw parsed filter dict.

    Steps:
      1. Ensure the top level is a dict.
      2. Unwrap a single ``{"filter": {...}}`` wrapper the model often emits.
      3. Reject forbidden / write operations.
      4. Normalize operator keys (``gt`` -> ``$gt``).
      5. Resolve field paths against the schema (case-insensitive).
      6. Recursively validate ``$and`` / ``$or`` operands.

    Returns a clean filter dict safe to pass to ``collection.find()``.
    """
    if not isinstance(raw, dict):
        raise FilterValidationError("Generated filter is not a JSON object.")

    filter_dict = raw

    # The model frequently wraps the real filter in {"filter": {...}}.
    if len(filter_dict) == 1 and "filter" in filter_dict:
        inner = filter_dict["filter"]
        if isinstance(inner, dict):
            filter_dict = inner

    # Reject any top-level forbidden keys.
    for key in filter_dict:
        if key in _FORBIDDEN_KEYS:
            raise FilterValidationError(
                f"Forbidden operation '{key}' is not allowed in a read-only filter."
            )
        if key not in _FIELD_OPERATORS and key not in _LOGICAL_OPERATORS and not key.startswith("$"):
            # Treat as a field path (validated below).
            pass

    normalized: Dict[str, Any] = {}

    for raw_field, value in filter_dict.items():
        field = _resolve_field_path(raw_field, schema_fields)

        if field in _LOGICAL_OPERATORS:
            # $and / $or: value must be a list of filter dicts.
            if not isinstance(value, list):
                raise FilterValidationError(
                    f"'{field}' expects a list of filter objects."
                )
            normalized[field] = [
                normalize_filter(item, schema_fields) for item in value
            ]
            continue

        if isinstance(value, dict):
            # Field condition: { "$gt": 8 } or { "gt": 8 }.
            cond: Dict[str, Any] = {}
            for op_key, op_val in value.items():
                op = _normalize_operator_key(op_key)
                if op in _FORBIDDEN_KEYS:
                    raise FilterValidationError(
                        f"Forbidden operator '{op}' is not allowed."
                    )
                if op not in _FIELD_OPERATORS and op not in _LOGICAL_OPERATORS:
                    raise FilterValidationError(
                        f"Unsupported operator '{op_key}' on field '{field}'. "
                        f"Allowed: {sorted(_FIELD_OPERATORS)}."
                    )
                cond[op] = op_val
            normalized[field] = cond
        else:
            # Equality: { "city": "Pune" } -> { "city": "$eq": "Pune" }?  No --
            # MongoDB equality is just { "city": "Pune" }. Keep as-is.
            normalized[field] = value

    return normalized


def validate_filter(
    filter_dict: Dict[str, Any], schema_fields: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate a filter dict against the schema.

    Returns ``(True, [])`` if valid, otherwise ``(False, errors)``.
    This is a thin wrapper around :func:`normalize_filter` that collects
    errors instead of raising.
    """
    errors: List[str] = []
    try:
        normalize_filter(filter_dict, schema_fields)
    except FilterValidationError as exc:
        errors.append(str(exc))
    return (len(errors) == 0), errors
