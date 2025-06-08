FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install base packages with exact versions
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    protobuf==3.20.3 \
    altair==4.2.2 \
    pandas==1.5.3

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]