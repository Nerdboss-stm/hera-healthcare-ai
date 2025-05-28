# Use amd64-compatible image for M1/M2 Macs
FROM --platform=linux/amd64 python:3.10-slim-bullseye

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first for caching efficiency
COPY week3_day13/requirements.txt ./requirements.txt

# Install project dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir "uvicorn[standard]" prometheus_client psycopg2-binary

# Copy rest of the project into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "clinical_summarizer_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

