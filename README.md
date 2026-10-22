# Schema-Adaptive Multilingual NL-to-MongoDB Query Generation (RAG)

A presentation-ready demo that turns natural-language questions into validated,
read-only MongoDB queries using a local LLM (Ollama) and schema-aware RAG
(ChromaDB). Built for a college final-year project demonstration.

## What the demo does

1. You type a natural-language question (e.g. "Find students whose GPA is
   greater than 8").
2. The system retrieves the relevant schema fields from ChromaDB.
3. A local Ollama model generates a MongoDB `find()` filter.
4. The filter is validated against the live schema (fields, operators, safety).
5. The validated filter is executed against the real MongoDB collection.
6. Matching records are displayed in a table.

## Prerequisites

- MongoDB running locally on `mongodb://localhost:27017` (database `collegeDB`,
  collection `students`).
- Ollama running locally on `http://localhost:11434` with at least one model
  installed (run `ollama list` to check).
- Python 3.12 (a `venv/` is already provided in this repo).

## First-time setup

From the project root (`nl2mongodb-2`):

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (streamlit was added for the frontend)
pip install -r requirements.txt

# 3. Confirm your environment variables
#    .env already contains MONGODB_URI and MONGODB_DATABASE.
#    Optionally set OLLAMA_MODEL to a larger local model, e.g.:
#      $env:OLLAMA_MODEL="qwen2.5:3b"
```

Never commit `.env`, `venv/`, `__pycache__/`, or `chroma_db/` (they are in
`.gitignore`).

## Start the application

MongoDB and Ollama must be running first, then launch the Streamlit frontend:

```powershell
# Terminal A: make sure MongoDB is running (mongod)
# Terminal B: make sure Ollama is running (ollama serve) and the model is pulled
ollama list
ollama pull qwen2.5:0.5b   # only if not already listed

# Terminal C: launch the app
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Demo sequence (examiner walkthrough)

1. **Query Playground** — pick an example question or type your own:
   - `Find students from Pune`
   - `Find students whose GPA is greater than 8`
   - `Show students from CSE department`
   - `Find students who know Python`
   - `Find students with email addresses`
   - `Show students from Pune with GPA above 8`
2. Click **Generate Query** — see the retrieved schema fields and the generated
   MongoDB filter (formatted JSON) plus a validation status.
3. Click **Execute Query** — see the actual matching records from MongoDB.
4. **Schema Explorer** — view the current collection, field paths, types, and
   sample values; click **Refresh Schema** to re-index into ChromaDB.
5. **Schema Evolution** — view saved snapshots (`students_v1.json` …), click
   **Check for Changes** to compare the live schema with the latest snapshot. A
   new snapshot is created only when a real change is detected.
6. **Evaluation** — view Recall@5 for schema retrieval and a small query
   generation evaluation (exact-match + execution).

## Project structure

```
backend/
  main.py
  database/mongodb.py      # MongoDB connection (get_database, test_connection)
  database/seed.py         # sample student data (NOT run on startup)
  schema/schema_models.py  # pydantic schema models
  schema/schema_extractor.py  # extract + save schema snapshots
  schema/schema_comparator.py # compare two snapshots
  schema/schema_monitor.py    # live vs latest snapshot; new version on change
  rag/schema_retriever.py     # ChromaDB RAG over schema fields
  query/llm_client.py         # local Ollama client (configurable model)
  query/query_generator.py    # RAG -> prompt -> LLM -> parse -> validate
  query/query_validator.py    # filter validation / normalization / safety
  query/query_executor.py     # read-only find() execution
  evaluation/test_schema_retrieval.py # Recall@5 evaluation
frontend/
  app.py                     # Streamlit app (Query Playground, Explorer, Evolution, Evaluation)
schema_versions/             # saved schema snapshots (students_vN.json)
chroma_db/                   # ChromaDB persistence
tests/
  test_demo.py               # end-to-end smoke tests
.env                         # MongoDB URI / database (not committed)
```

## Components in plain language

- **MongoDB connection** — connects to the local `collegeDB` database using
  settings from `.env`. Exposes `get_database()` and `test_connection()`.
- **Schema extractor** — inspects MongoDB documents, detects field paths and
  data types (including nested objects like `academic.gpa` and arrays like
  `skills`), collects sample values, and saves snapshots as JSON.
- **Schema comparator / monitor** — compares the latest saved snapshot against
  the live schema and creates a new version only when fields or types actually
  change (no duplicates).
- **Schema retriever (RAG)** — loads the latest snapshot, embeds field
  descriptions with `all-MiniLM-L6-v2`, stores them in ChromaDB, and retrieves
  the fields most relevant to a question.
- **LLM client** — calls the local Ollama chat API. The model is configurable
  via `OLLAMA_MODEL` (default `qwen2.5:0.5b`).
- **Query generator** — builds a schema-aware prompt, calls Ollama, parses the
  JSON filter, normalizes common operator spellings, and validates it.
- **Query validator** — ensures the output is a JSON object, uses only schema
  fields, permits only safe operators (`$gt`, `$gte`, `$lt`, `$lte`, `$eq`,
  `$ne`, `$in`, `$nin`, `$exists`, `$and`, `$or`), and rejects write/JS
  operations.
- **Query executor** — runs `collection.find(filter)` (read-only, capped at 50
  docs) and returns matching records plus execution metadata.
- **Streamlit frontend** — Query Playground, Schema Explorer, Schema Evolution,
  and Evaluation pages.

## Known limitations

- The bundled `qwen2.5:0.5b` model is small and can occasionally produce
  malformed output; the parser/validator handles this by reporting a clear
  error rather than executing a bad query. For more reliable generation, use a
  larger local model (e.g. `qwen2.5:3b`) and set `OLLAMA_MODEL`.
- Query generation is currently English-only. Multilingual (Hindi/Marathi and
  code-mixed) support is a future milestone.
- The demo generates `find()` filters only — no aggregation pipelines or write
  operations.
- Do not run `backend/database/seed.py` during the presentation: it calls
  `delete_many({})` and would wipe the sample data.

## Running the tests

```powershell
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe tests/test_demo.py
.\venv\Scripts\python.exe -m backend.evaluation.test_schema_retrieval
```

## Git

Work only inside this copied repository. The new GitHub remote should be
configured only after you provide its URL; do not push to the original
repository.
