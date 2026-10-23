"""Schema-aware MongoDB query generation using the local LLM.

Pipeline:

    natural-language question
    -> retrieve relevant schema context from ChromaDB
    -> construct a prompt
    -> call the local Ollama model
    -> parse / normalize JSON
    -> validate against the live schema

This module reuses the existing :mod:`backend.rag.schema_retriever` retriever
and does not create a second RAG implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from backend.database.mongodb import get_database
from backend.rag.schema_retriever import load_latest_schema, retrieve_schema
from backend.query.query_validator import normalize_filter, FilterValidationError
from backend.query import llm_client
from backend.query.marathi_preprocessor import preprocess_question


class QueryGenerationError(RuntimeError):
    """Raised when a query cannot be generated or parsed."""


def _load_schema_fields() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the latest schema snapshot.

    Returns ``(full_schema, fields_by_path)`` where ``fields_by_path`` maps
    field path -> field schema dict for validation.
    """
    schema = load_latest_schema()
    fields_by_path: Dict[str, Any] = {}
    for field in schema.get("fields", []):
        fields_by_path[field["path"]] = field
    return schema, fields_by_path


def _retrieve_context(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """Retrieve the most relevant schema fields for a question."""
    results = retrieve_schema(question, top_k=top_k)
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        context.append(
            {
                "field_path": metadata.get("field_path"),
                "types": metadata.get("types"),
                "document": document,
                "distance": round(float(distance), 4),
            }
        )
    return context


def _build_prompt(question: str, schema: Dict[str, Any], context: List[Dict[str, Any]]) -> str:
    """Construct the prompt sent to the LLM."""
    lines: List[str] = []
    lines.append("You are a MongoDB query generation assistant.")
    lines.append("")
    lines.append("TASK: Convert the user's natural-language question into a MongoDB find() filter.")
    lines.append("Return ONLY a valid JSON object representing the filter. No markdown, no explanation, no extra text.")
    lines.append("")
    lines.append(f"Collection: {schema.get('collection')}")
    lines.append(f"Schema version: latest snapshot in schema_versions/")
    lines.append("")
    lines.append("RELEVANT SCHEMA FIELDS (use only these field paths):")
    for item in context:
        lines.append(
            f"  - {item['field_path']} (types: {item['types']}) - {item['document']}"
        )
    lines.append("")
    lines.append("RULES:")
    lines.append("  1. Use only field paths listed above.")
    lines.append("  2. Only include fields directly referenced by the question. Do NOT add extra fields.")
    lines.append("  3. Do NOT invent names, values, or conditions the user did not mention.")
    lines.append("  4. For numeric comparisons use MongoDB operators: $gt, $gte, $lt, $lte.")
    lines.append("  5. For equality use the value directly, e.g. {\"city\": \"Pune\"}.")
    lines.append("  6. For array membership (e.g. skills contains Python) use {\"skills\": \"Python\"}.")
    lines.append('  7. To check a field exists (e.g. has an email) use {"email": {"$exists": true}}.')
    lines.append("  8. For nested fields use dotted paths, e.g. {\"academic.gpa\": {\"$gt\": 8}}.")
    lines.append('  9. For case-insensitive text search (e.g. student likes painting) use {"hobby": {"$regex": "^painting$", "$options": "i"}}.')
    lines.append(" 10. Output a single raw JSON object. Do not wrap it in {\"filter\": ...}.")
    lines.append(" 11. Do not use $where, $eval, or any write/update/delete operators.")
    lines.append("")
    lines.append("EXAMPLES:")
    lines.append('  Q: Find students from Pune')
    lines.append('  A: {"city": "Pune"}')
    lines.append('  Q: Find students whose GPA is greater than 8')
    lines.append('  A: {"gpa": {"$gt": 8}}')
    lines.append('  Q: Find students from CSE department')
    lines.append('  A: {"department": "CSE"}')
    lines.append('  Q: Find students who know Python')
    lines.append('  A: {"skills": "Python"}')
    lines.append('  Q: Student who knows java')
    lines.append('  A: {"skills": "Java"}')
    lines.append('  Q: Find students with email addresses')
    lines.append('  A: {"email": {"$exists": true}}')
    lines.append('  Q: Student like to paint')
    lines.append('  A: {"hobby": {"$regex": "^paint", "$options": "i"}}')
    lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("FILTER:")
    return "\n".join(lines)


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Any:
    """Extract the first valid JSON object from text.

    Handles:
      - trailing/leading commentary
      - markdown fences (already stripped but defensive)
      - a {"filter": {...}} wrapper (unwrapped by the caller/validator)
    """
    text = _strip_markdown(text)
    # Try the whole string first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to finding the first balanced { ... } object.
    start = text.find("{")
    if start == -1:
        raise QueryGenerationError("No JSON object found in model response.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    raise QueryGenerationError("Model response did not contain a valid JSON object.")


def _translate_if_marathi(question: str) -> str:
    """Translate Marathi (Devanagari) input to English before generation."""
    return preprocess_question(question)


def generate_filter(question: str) -> Dict[str, Any]:
    """Generate a validated MongoDB filter for a natural-language question.

    Returns the normalized filter dict.  Raises :class:`QueryGenerationError`
    if the model is unavailable, returns malformed output, or the filter fails
    schema validation.
    """
    question = _translate_if_marathi(question)

    if not llm_client.is_available():
        raise QueryGenerationError(
            "Ollama is not reachable. Start Ollama (`ollama serve`) and confirm "
            f"the model `{llm_client.MODEL_NAME}` is installed (`ollama list`)."
        )

    schema, fields_by_path = _load_schema_fields()
    context = _retrieve_context(question)

    prompt = _build_prompt(question, schema, context)

    try:
        raw = llm_client.generate_response(prompt, use_json_mode=True)
    except llm_client.OllamaError as exc:
        raise QueryGenerationError(str(exc)) from exc

    try:
        parsed = _extract_json_object(raw)
    except QueryGenerationError as exc:
        raise QueryGenerationError(
            f"Failed to parse model response as JSON: {exc}\n"
            f"Raw response: {raw[:300]}"
        ) from exc

    try:
        filter_dict = normalize_filter(parsed, fields_by_path)
    except FilterValidationError as exc:
        raise QueryGenerationError(
            f"Generated filter failed validation: {exc}\n"
            f"Raw model response: {raw[:300]}"
        ) from exc

    return filter_dict


def generate_filter_with_metadata(question: str) -> Dict[str, Any]:
    """Generate a filter and return it with debugging metadata.

    Returns a dict with keys:
      filter, question, model, schema_version, retrieved_fields, raw_response.
    """
    question = _translate_if_marathi(question)

    if not llm_client.is_available():
        raise QueryGenerationError(
            "Ollama is not reachable. Start Ollama (`ollama serve`) and confirm "
            f"the model `{llm_client.MODEL_NAME}` is installed (`ollama list`)."
        )

    schema, fields_by_path = _load_schema_fields()
    context = _retrieve_context(question)

    prompt = _build_prompt(question, schema, context)

    raw = ""
    try:
        raw = llm_client.generate_response(prompt, use_json_mode=True)
    except llm_client.OllamaError as exc:
        raise QueryGenerationError(str(exc)) from exc

    try:
        parsed = _extract_json_object(raw)
        filter_dict = normalize_filter(parsed, fields_by_path)
    except (QueryGenerationError, FilterValidationError) as exc:
        return {
            "filter": None,
            "error": str(exc),
            "question": question,
            "model": llm_client.MODEL_NAME,
            "schema_version": "latest",
            "retrieved_fields": [c["field_path"] for c in context],
            "raw_response": raw[:500],
        }

    return {
        "filter": filter_dict,
        "error": None,
        "question": question,
        "model": llm_client.MODEL_NAME,
        "schema_version": "latest",
        "retrieved_fields": [c["field_path"] for c in context],
        "raw_response": raw,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for q in ["Find students from Pune", "Find students whose GPA is greater than 8"]:
        try:
            result = generate_filter_with_metadata(q)
            print(f"Q: {q}")
            print("  filter:", result.get("filter"))
            if result.get("error"):
                print("  error:", result["error"])
        except QueryGenerationError as exc:
            print(f"Q: {q} -> ERROR: {exc}")
