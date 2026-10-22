"""Streamlit frontend for the Schema-Adaptive NL-to-MongoDB demo.

Run from the project root:

    .venv\\Scripts\\python.exe -m streamlit run frontend/app.py

The app imports the backend query modules directly (in-process), so only one
server needs to be running.  Ollama and MongoDB must be started separately.
"""

from __future__ import annotations

import os
import sys
import json
import contextlib
import io
from typing import Any, Dict, List, Tuple

# Ensure the project root (parent of frontend/) is on the import path so
# ``backend.*`` modules resolve whether streamlit is launched from the root
# or from the frontend/ folder.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

from backend.query import llm_client  # noqa: E402
from backend.query.query_generator import generate_filter_with_metadata, QueryGenerationError  # noqa: E402
from backend.query.query_executor import execute_filter  # noqa: E402
from backend.database.mongodb import get_database, test_connection  # noqa: E402
from backend.rag.schema_retriever import (  # noqa: E402
    load_latest_schema,
    index_schema,
    retrieve_schema,
)
from backend.schema.schema_extractor import (  # noqa: E402
    extract_collection_schema,
    save_schema_version,
)
from backend.schema.schema_monitor import get_latest_snapshot  # noqa: E402
from backend.schema.schema_comparator import load_schema, compare_schemas  # noqa: E402
from backend.evaluation.test_schema_retrieval import (  # noqa: E402
    TEST_CASES as RETRIEVAL_TEST_CASES,
    evaluate_recall_at_k,
)

COLLECTION_NAME = "students"
SCHEMA_DIR = os.path.join(ROOT, "schema_versions")


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------
def check_mongodb() -> Tuple[bool, str]:
    """Return (ok, message) for the MongoDB connection."""
    try:
        db = get_database()
        db.command("ping")
        return True, f"Connected to MongoDB ({db.name})."
    except Exception as exc:  # pragma: no cover - connectivity depends on env
        return False, f"MongoDB unavailable: {exc}"


def check_ollama() -> Tuple[bool, str]:
    """Return (ok, message) for the Ollama connection."""
    try:
        if llm_client.is_available():
            return True, f"Ollama reachable (model: {llm_client.MODEL_NAME})."
        return False, "Ollama /api/tags did not respond."
    except Exception as exc:  # pragma: no cover - connectivity depends on env
        return False, f"Ollama unavailable: {exc}"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def list_snapshots() -> List[Dict[str, Any]]:
    """List available schema snapshot files with version numbers."""
    snapshots: List[Dict[str, Any]] = []
    if not os.path.isdir(SCHEMA_DIR):
        return snapshots
    for name in os.listdir(SCHEMA_DIR):
        if name.startswith("students_v") and name.endswith(".json"):
            try:
                version = int(name.split("_v")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            snapshots.append({"version": version, "file": name})
    snapshots.sort(key=lambda item: item["version"])
    return snapshots


def check_schema_changes() -> Dict[str, Any]:
    """Compare the live DB schema against the latest snapshot.

    Reuses the existing schema monitor/comparator/extractor modules so the
    behaviour matches the command-line tool.  Returns a dict describing the
    latest version, detected changes, and any newly created version.
    """
    latest_path, latest_version = get_latest_snapshot()

    if latest_path is None:
        current_schema = extract_collection_schema(COLLECTION_NAME)
        save_schema_version(current_schema, "v1")
        return {
            "latest_version": "v1",
            "changes": None,
            "has_changes": False,
            "created": "v1",
            "message": "No previous snapshot. Created students_v1.json.",
        }

    old_schema = load_schema(str(latest_path))
    current_schema = extract_collection_schema(COLLECTION_NAME)
    changes = compare_schemas(old_schema, current_schema.model_dump())
    has_changes = any([
        changes["added_fields"],
        changes["removed_fields"],
        changes["changed_types"],
    ])

    result: Dict[str, Any] = {
        "latest_version": f"v{latest_version}",
        "changes": changes,
        "has_changes": has_changes,
    }

    if not has_changes:
        result["message"] = "No schema changes detected."
        return result

    next_version = latest_version + 1
    save_schema_version(current_schema, f"v{next_version}")
    result["new_version"] = f"v{next_version}"
    result["has_changes"] = True
    result["message"] = (
        f"Schema changes detected. Created students_v{next_version}.json."
    )
    return result


# ---------------------------------------------------------------------------
# Page: Query Playground
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "Find students from Pune",
    "Find students whose GPA is greater than 8",
    "Show students from CSE department",
    "Find students who know Python",
    "Find students with email addresses",
    "Show students from Pune with GPA above 8",
]


def render_query_playground() -> None:
    st.header("🔍 Query Playground")
    st.caption(
        "Enter a natural-language question. The system retrieves relevant schema "
        "fields from ChromaDB, generates a MongoDB filter with the local Ollama "
        "model, validates it, and executes it against the real database."
    )

    # Connectivity status chips.
    mongo_ok, mongo_msg = check_mongodb()
    ollama_ok, ollama_msg = check_ollama()
    col1, col2 = st.columns(2)
    with col1:
        if mongo_ok:
            st.success("✅ " + mongo_msg)
        else:
            st.error("❌ " + mongo_msg)
    with col2:
        if ollama_ok:
            st.success("✅ " + ollama_msg)
        else:
            st.error("❌ " + ollama_msg)

    st.divider()

    # Example query buttons.
    st.subheader("Example queries")
    cols = st.columns(3)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 3]:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state["nl_question"] = example

    # Question input.
    question = st.text_area(
        "Your question",
        value=st.session_state.get("nl_question", ""),
        height=100,
        placeholder="e.g. Find students whose GPA is greater than 8",
    )
    st.session_state["nl_question"] = question

    col_generate, col_execute = st.columns([1, 1])
    generate = col_generate.button("Generate Query", type="primary", use_container_width=True)
    execute = col_execute.button("Execute Query", use_container_width=True)

    # --- Generate ---
    if generate:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not ollama_ok:
            st.error("Ollama is not available. Start Ollama and try again.")
        elif not mongo_ok:
            st.error("MongoDB is not available. Start MongoDB and try again.")
        else:
            with st.spinner("Retrieving schema context and generating query..."):
                try:
                    result = generate_filter_with_metadata(question)
                    st.session_state["last_result"] = result
                    st.session_state["last_error"] = None
                    st.session_state["last_records"] = None
                    st.session_state["last_exec_meta"] = None
                except QueryGenerationError as exc:
                    st.session_state["last_result"] = None
                    st.session_state["last_error"] = str(exc)
                    st.session_state["last_records"] = None
                    st.session_state["last_exec_meta"] = None

    # --- Execute ---
    if execute:
        result = st.session_state.get("last_result")
        if not result or result.get("filter") is None:
            st.warning("Generate a query first.")
        elif not mongo_ok:
            st.error("MongoDB is not available. Start MongoDB and try again.")
        else:
            with st.spinner("Executing validated read-only query..."):
                try:
                    records, meta = execute_filter(result["filter"])
                    st.session_state["last_records"] = records
                    st.session_state["last_exec_meta"] = meta
                except Exception as exc:  # pragma: no cover - defensive
                    st.session_state["last_records"] = None
                    st.session_state["last_exec_meta"] = None
                    st.error(f"Query execution failed: {exc}")

    # --- Display results ---
    result = st.session_state.get("last_result")
    error = st.session_state.get("last_error")

    if error:
        st.error(error)
    elif result:
        st.subheader("Generated MongoDB filter")
        st.code(json.dumps(result.get("filter"), indent=2), language="json")

        # Validation status.
        filter_obj = result.get("filter")
        if filter_obj:
            st.success("✅ Validation passed — filter uses schema fields and approved operators only.")
        else:
            st.error("❌ Validation failed — no filter produced.")

        with st.expander("Retrieved schema context", expanded=False):
            fields = result.get("retrieved_fields", [])
            if fields:
                st.write("Relevant field paths:")
                st.write(", ".join(fields))
            else:
                st.write("No schema fields retrieved.")
            st.write("Model:", result.get("model"))
            st.write("Schema version:", result.get("schema_version"))
            if result.get("raw_response"):
                st.caption("Raw model response (debug):")
                st.code(result["raw_response"], language="text")

    # --- Execution output ---
    records = st.session_state.get("last_records")
    exec_meta = st.session_state.get("last_exec_meta")

    if exec_meta:
        st.subheader("Matching records")
        st.caption(
            f"Collection: {exec_meta['collection']} | "
            f"Returned: {exec_meta['returned']} | "
            f"Limit: {exec_meta['limit']} | "
            f"Time: {exec_meta['execution_time_ms']} ms"
        )
        if records:
            st.dataframe(records, use_container_width=True, hide_index=True)
        else:
            st.info("No matching records found.")
    elif st.session_state.get("last_exec_meta") is None and execute and st.session_state.get("last_result"):
        pass  # handled above


# ---------------------------------------------------------------------------
# Page: Schema Explorer
# ---------------------------------------------------------------------------
def render_schema_explorer() -> None:
    st.header("🗂️ Schema Explorer")
    st.caption("Inspect the current extracted schema for the students collection.")

    mongo_ok, mongo_msg = check_mongodb()
    if not mongo_ok:
        st.error("❌ " + mongo_msg)
        return

    if st.button("Refresh Schema", type="primary"):
        with st.spinner("Extracting schema and re-indexing into ChromaDB..."):
            try:
                index_schema()
                st.success("Schema re-indexed into ChromaDB.")
            except Exception as exc:
                st.error(f"Schema refresh failed: {exc}")

    st.divider()

    try:
        schema = load_latest_schema()
    except FileNotFoundError:
        st.warning("No schema snapshot found. Run 'Refresh Schema' or 'Check for Changes' first.")
        return

    col_v, col_coll = st.columns(2)
    with col_v:
        files = list_snapshots()
        latest = files[-1]["version"] if files else "?"
        st.metric("Latest schema version", f"v{latest}")
    with col_coll:
        st.metric("Collection", schema.get("collection", COLLECTION_NAME))

    st.subheader("Field paths, types, and sample values")
    rows = []
    for field in schema.get("fields", []):
        samples = field.get("sample_values") or []
        rows.append({
            "field_path": field["path"],
            "types": ", ".join(field.get("types", [])),
            "sample_values": ", ".join(str(s) for s in samples[:5]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Schema Evolution
# ---------------------------------------------------------------------------
def render_schema_evolution() -> None:
    st.header("📈 Schema Evolution")
    st.caption(
        "Compare the live database schema against the latest saved snapshot. "
        "A new snapshot is created only when a real change is detected."
    )

    mongo_ok, mongo_msg = check_mongodb()
    if not mongo_ok:
        st.error("❌ " + mongo_msg)
        return

    st.subheader("Available schema snapshots")
    snapshots = list_snapshots()
    if snapshots:
        st.dataframe(snapshots, use_container_width=True, hide_index=True)
    else:
        st.info("No schema snapshots found.")

    st.divider()

    if st.button("Check for Changes", type="primary"):
        with st.spinner("Comparing live schema to latest snapshot..."):
            try:
                outcome = check_schema_changes()
                st.session_state["schema_outcome"] = outcome
            except Exception as exc:
                st.error(f"Schema change check failed: {exc}")

    outcome = st.session_state.get("schema_outcome")
    if outcome:
        st.subheader("Change detection result")
        if outcome.get("created"):
            st.info(outcome["message"])
        elif outcome.get("has_changes"):
            st.success(outcome["message"])
            changes = outcome["changes"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Added fields**")
                st.write(changes["added_fields"] or "—")
            with c2:
                st.write("**Removed fields**")
                st.write(changes["removed_fields"] or "—")
            with c3:
                st.write("**Changed types**")
                st.write(changes["changed_types"] or "—")
            st.write("New snapshot:", outcome.get("new_version"))
        else:
            st.info(outcome.get("message", "No schema changes detected."))


# ---------------------------------------------------------------------------
# Page: Evaluation
# ---------------------------------------------------------------------------
def render_evaluation() -> None:
    st.header("📊 Evaluation")
    st.caption(
        "Recall@5 measures schema retrieval quality.  Exact-match and execution "
        "accuracy measure generated MongoDB filters.  These are distinct metrics — "
        "a high retrieval recall does not imply perfect query generation."
    )

    st.subheader("Schema retrieval — Recall@5 (English)")
    st.write(
        "Six manually labeled queries checked against the retrieved schema fields "
        "(top-5 from ChromaDB)."
    )

    if st.button("Run retrieval evaluation", type="primary"):
        with st.spinner("Running Recall@5 evaluation..."):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                evaluate_recall_at_k(RETRIEVAL_TEST_CASES, k=5)
            output = buf.getvalue()
        st.text(output)

    st.divider()

    st.subheader("Query generation — exact-match & execution (sample)")
    st.caption(
        "Runs a fixed set of English questions through the generator and checks "
        "whether the produced filter matches the expected filter exactly, and "
        "whether it executes successfully against MongoDB."
    )

    GEN_CASES = [
        {
            "question": "Find students from Pune",
            "expected": {"city": "Pune"},
        },
        {
            "question": "Find students whose GPA is greater than 8",
            "expected": {"gpa": {"$gt": 8}},
        },
        {
            "question": "Show students from CSE department",
            "expected": {"department": "CSE"},
        },
        {
            "question": "Find students who know Python",
            "expected": {"skills": "Python"},
        },
        {
            "question": "Find students with email addresses",
            "expected": {"email": {"$exists": True}},
        },
    ]

    if st.button("Run query-generation evaluation", type="primary"):
        if not llm_client.is_available():
            st.error("Ollama is not available. Start Ollama and try again.")
        else:
            exact_hits = 0
            exec_hits = 0
            rows: List[Dict[str, Any]] = []
            mongo_ok, _ = check_mongodb()
            for case in GEN_CASES:
                q = case["question"]
                expected = case["expected"]
                try:
                    result = generate_filter_with_metadata(q)
                    actual = result.get("filter")
                    exact = (actual == expected)
                    if exact:
                        exact_hits += 1
                    executed = False
                    record_count = None
                    if actual and mongo_ok:
                        try:
                            records, _ = execute_filter(actual)
                            executed = True
                            record_count = len(records)
                            if executed:
                                exec_hits += 1
                        except Exception:
                            executed = False
                    rows.append({
                        "question": q,
                        "expected": json.dumps(expected),
                        "actual": json.dumps(actual) if actual else None,
                        "exact_match": "✅" if exact else "❌",
                        "executed": "✅" if executed else "❌",
                        "records": record_count,
                        "error": result.get("error"),
                    })
                except QueryGenerationError as exc:
                    rows.append({
                        "question": q,
                        "expected": json.dumps(expected),
                        "actual": None,
                        "exact_match": "❌",
                        "executed": "❌",
                        "records": None,
                        "error": str(exc)[:120],
                    })

            n = len(GEN_CASES)
            st.metric("Exact-match accuracy", f"{exact_hits}/{n} ({exact_hits/n:.0%})")
            st.metric("Execution success", f"{exec_hits}/{n} ({exec_hits/n:.0%})")
            st.caption(
                "Small sample set — directional only, not a general accuracy claim."
            )
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="NL-to-MongoDB Query Generator",
        page_icon="🍃",
        layout="wide",
    )

    st.title("🍃 Schema-Adaptive NL-to-MongoDB Query Generator")
    st.caption(
        "Natural-language questions -> schema-aware RAG -> local LLM (Ollama) -> "
        "validated read-only MongoDB query -> real results."
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Query Playground", "Schema Explorer", "Schema Evolution", "Evaluation"],
        index=0,
    )

    if page == "Query Playground":
        render_query_playground()
    elif page == "Schema Explorer":
        render_schema_explorer()
    elif page == "Schema Evolution":
        render_schema_evolution()
    elif page == "Evaluation":
        render_evaluation()

    st.sidebar.divider()
    st.sidebar.subheader("Status")
    mongo_ok, mongo_msg = check_mongodb()
    ollama_ok, ollama_msg = check_ollama()
    st.sidebar.write("MongoDB:", "✅" if mongo_ok else "❌", mongo_msg)
    st.sidebar.write("Ollama:", "✅" if ollama_ok else "❌", ollama_msg)
    st.sidebar.caption("Model: " + llm_client.MODEL_NAME)
    st.sidebar.caption(
        "The bundled qwen2.5:0.5b model is small; a larger local model gives "
        "more reliable generation. Set OLLAMA_MODEL to switch."
    )


if __name__ == "__main__":
    main()
