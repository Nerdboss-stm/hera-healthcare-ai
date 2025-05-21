import psycopg2
from datetime import datetime
from clinical_summarizer_api.app.config import DB_CONFIG

def log_to_db(note: str, summary: str, status: str = "SUCCESS"):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO summaries (note, summary, status, timestamp) VALUES (%s, %s, %s, %s)",
            (note, summary, status, datetime.now())
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Logging failed: {e}")

