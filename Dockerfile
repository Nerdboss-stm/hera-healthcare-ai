FROM python:3.10-slim-bullseye

WORKDIR /app

# Install PyTorch CPU-only first (much smaller, no CUDA bloat)
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir "uvicorn[standard]" prometheus_client psycopg2-binary

COPY config/ ./config/
COPY serving/ ./serving/
COPY agents/ ./agents/
COPY rag/ ./rag/
COPY ner/ ./ner/
COPY evaluation/ ./evaluation/
COPY fhir_layer/ ./fhir_layer/
COPY data_engineering/ ./data_engineering/
COPY risk_prediction/ ./risk_prediction/
COPY clinical_summarizer/ ./clinical_summarizer/
COPY data/processed/ ./data/processed/
COPY data/clinical_notes/ ./data/clinical_notes/
COPY database/ ./database/
COPY scripts/ ./scripts/
COPY model/ ./model/

EXPOSE 8000

CMD python scripts/init_db.py && uvicorn serving.api:app --host 0.0.0.0 --port ${PORT:-8000}
