FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install base packages with exact versions
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    protobuf==3.20.3 \
    && pip install --no-cache-dir \
    pandas==1.5.3 \
    mlflow==1.30.0 \
    psycopg2-binary==2.9.6 \
    boto3==1.26.59\
    pymysql

CMD mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB} \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0