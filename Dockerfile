FROM --platform=linux/amd64 python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir "uvicorn[standard]" prometheus_client psycopg2-binary

COPY config/ ./config/
COPY serving/ ./serving/
COPY risk_prediction/ ./risk_prediction/
COPY clinical_summarizer/ ./clinical_summarizer/
COPY data/processed/ ./data/processed/
COPY data/clinical_notes/ ./data/clinical_notes/
COPY database/ ./database/

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
