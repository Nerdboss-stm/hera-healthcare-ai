FROM --platform=linux/amd64 python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir "uvicorn[standard]" prometheus_client psycopg2-binary

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
