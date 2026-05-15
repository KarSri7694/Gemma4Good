from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import DatabaseManager, DEFAULT_DATABASE


def initialize_database(db_path: Path | None = None) -> None:
    if db_path is None:
        DEFAULT_DATABASE.ensure_database()
        return
    DatabaseManager(db_path=db_path, schema_path=ROOT / "db" / "schema.sql").ensure_database()


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized SQLite database at {DEFAULT_DATABASE.db_path}")
