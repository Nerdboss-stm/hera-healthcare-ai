"""Initialize PostgreSQL tables on first deploy."""

import os
import sys

try:
    import psycopg2
except ImportError:
    print("psycopg2 not installed, skipping DB init")
    sys.exit(0)


def init_db():
    """Create tables from schema.sql if they don't exist."""
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "user": os.getenv("DB_USER", "hera"),
        "password": os.getenv("DB_PASSWORD", "hera123"),
        "dbname": os.getenv("DB_NAME", "hera_ai"),
    }

    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "database", "schema.sql"
    )

    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()

        if os.path.exists(schema_path):
            with open(schema_path) as f:
                cursor.execute(f.read())
            print("DB tables initialized from schema.sql")
        else:
            print(f"schema.sql not found at {schema_path}, skipping")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB init skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    init_db()
