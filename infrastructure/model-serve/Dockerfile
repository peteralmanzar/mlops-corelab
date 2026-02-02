# Model Serving API
# Loads champion models from MLflow and serves predictions via REST API

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy the serving application
COPY serving/ /app/serving/

# Copy custom modules needed to unpickle sklearn models
COPY data_transform.py /app/data_transform.py
COPY model_builder.py /app/model_builder.py

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV CHAMPION_ALIAS=champion
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
