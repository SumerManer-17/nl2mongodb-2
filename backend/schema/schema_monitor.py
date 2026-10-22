
import re
from pathlib import Path

from backend.schema.schema_extractor import (
    extract_collection_schema,
    save_schema_version,
)
from backend.schema.schema_comparator import (
    load_schema,
    compare_schemas,
)


SCHEMA_DIR = Path("schema_versions")
COLLECTION_NAME = "students"


def get_latest_snapshot():
    pattern = re.compile(rf"{re.escape(COLLECTION_NAME)}_v(\d+)\.json$")
    snapshots = []

    for path in SCHEMA_DIR.glob(f"{COLLECTION_NAME}_v*.json"):
        match = pattern.match(path.name)

        if match:
            snapshots.append((int(match.group(1)), path))

    if not snapshots:
        return None, 0

    version, path = max(snapshots, key=lambda item: item[0])
    return path, version


def check_schema_changes():
    """Compare the live DB schema against the latest snapshot.

    Returns a dict describing the latest version, detected changes, and any
    newly created version.  Never returns ``None`` so callers can rely on the
    result.
    """
    result = {
        "latest_version": None,
        "changes": None,
        "has_changes": False,
        "created": None,
        "new_version": None,
        "message": "",
    }

    latest_path, latest_version = get_latest_snapshot()

    if latest_path is None:
        print("No previous snapshot found. Creating the first snapshot.")
        current_schema = extract_collection_schema(COLLECTION_NAME)
        save_schema_version(current_schema, "v1")
        result["latest_version"] = "v1"
        result["created"] = "v1"
        result["has_changes"] = False
        result["message"] = "No previous snapshot. Created students_v1.json."
        return result

    result["latest_version"] = f"v{latest_version}"

    old_schema = load_schema(str(latest_path))
    current_schema = extract_collection_schema(COLLECTION_NAME)

    changes = compare_schemas(
        old_schema,
        current_schema.model_dump(),
    )

    result["changes"] = changes

    has_changes = any([
        changes["added_fields"],
        changes["removed_fields"],
        changes["changed_types"],
    ])

    result["has_changes"] = has_changes

    if not has_changes:
        print("No schema changes detected.")
        result["message"] = "No schema changes detected."
        return result

    print("Schema changes detected:")
    print(changes)

    next_version = latest_version + 1
    save_schema_version(current_schema, f"v{next_version}")

    result["new_version"] = f"v{next_version}"
    result["message"] = (
        f"Schema changes detected. Created students_v{next_version}.json."
    )
    return result


if __name__ == "__main__":
    check_schema_changes()
