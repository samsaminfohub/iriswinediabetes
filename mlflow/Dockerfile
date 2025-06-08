FROM python:3.9-slim

WORKDIR /mlflow

# Install MLflow and PostgreSQL dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary \
    protobuf==3.20.3

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# Expose port
EXPOSE 5000

# Default command (will be overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0"]