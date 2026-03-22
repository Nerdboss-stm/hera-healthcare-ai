import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5433)),
    "user": os.getenv("DB_USER", "hera"),
    "password": os.getenv("DB_PASSWORD", "hera123"),
    "dbname": os.getenv("DB_NAME", "hera_ai"),
}

# Paths relative to project root
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATA_NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "clinical_notes")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
