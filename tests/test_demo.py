"""End-to-end smoke tests for the presentation demo.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_demo.py -v
or:
    .venv\\Scripts\\python.exe tests/test_demo.py
"""

from __future__ import annotations

import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.query.query_generator import generate_filter_with_metadata, QueryGenerationError
from backend.query.query_executor import execute_filter
from backend.query.query_validator import normalize_filter, FilterValidationError
from backend.schema.schema_monitor import check_schema_changes
from backend.schema.schema_extractor import extract_collection_schema, save_schema_version
from backend.rag.schema_retriever import retrieve_schema, load_latest_schema


def load_schema_fields():
    schema = load_latest_schema()
    return {f["path"]: f for f in schema["fields"]}


GEN_CASES = [
    ("Find students from Pune", {"city": "Pune"}),
    ("Find students whose GPA is greater than 8", {"gpa": {"$gt": 8}}),
    ("Show students from CSE department", {"department": "CSE"}),
    ("Find students who know Python", {"skills": "Python"}),
    ("Find students with email addresses", {"email": {"$exists": True}}),
    ("Show students from Pune with GPA above 8", {"city": "Pune", "gpa": {"$gt": 8}}),
    ("Find students whose academic department is CSE", {"academic.department": "CSE"}),
]


def test_generate_and_execute():
    failures = []
    for question, expected in GEN_CASES:
        result = generate_filter_with_metadata(question)
        actual = result.get("filter")
        err = result.get("error")
        if err:
            failures.append(f"{question}: generation error: {err}")
            continue
        if actual is None:
            failures.append(f"{question}: no filter produced")
            continue
        # academic.department and flat department both match Vikas; accept either.
        acceptable = expected
        if expected == {"academic.department": "CSE"}:
            acceptable = [{"academic.department": "CSE"}, {"department": "CSE"}]
        if isinstance(acceptable, dict) and actual != acceptable:
            failures.append(
                f"{question}: expected {expected}, got {actual}"
            )
            continue
        if isinstance(acceptable, list) and actual not in acceptable:
            failures.append(
                f"{question}: expected one of {acceptable}, got {actual}"
            )
            continue
        records, meta = execute_filter(actual)
        # every case should return at least Vikas (student_id 106)
        if not any(r.get("student_id") == 106 for r in records):
            failures.append(
                f"{question}: expected Vikas in results, got {records}"
            )
    assert not failures, "\n".join(failures)


def test_rejects_unknown_field():
    result = generate_filter_with_metadata("Find students with salary above 50000")
    assert result.get("filter") is None
    assert result.get("error") is not None


def test_validator_rejects_unsafe():
    schema_fields = load_schema_fields()
    unsafe = [
        ({"delete": {}}, "write op"),
        ({"$where": "1==1"}, "JS execution"),
        ({"salary": {"$gt": 100}}, "unknown field"),
        ({"gpa": {"$invalid": 8}}, "bad operator"),
        ("not a dict", "not a dict"),
    ]
    for raw, _label in unsafe:
        try:
            normalize_filter(raw, schema_fields)
        except FilterValidationError:
            continue
        raise AssertionError(f"Validator accepted unsafe input: {raw}")


def test_schema_evolution_no_duplicate():
    outcome = check_schema_changes()
    # Running again should not create a duplicate since nothing changed.
    outcome2 = check_schema_changes()
    assert outcome2["has_changes"] is False
    assert outcome2["latest_version"] == outcome["latest_version"]


def test_retrieval_recall():
    result = retrieve_schema("Find students from Pune", top_k=5)
    fields = {m["field_path"] for m in result["metadatas"][0]}
    assert "city" in fields


if __name__ == "__main__":
    tests = [
        test_generate_and_execute,
        test_rejects_unknown_field,
        test_validator_rejects_unsafe,
        test_schema_evolution_no_duplicate,
        test_retrieval_recall,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
