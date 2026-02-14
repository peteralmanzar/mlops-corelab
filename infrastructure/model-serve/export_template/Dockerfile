FROM python:3.10-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Copy custom modules for unpickling sklearn models
COPY lib/ /app/lib/

# Copy model artifacts
COPY model/ /app/model/

# Copy config
COPY config.json /app/config.json

# Set Python path so sklearn can find custom transformer classes during unpickling
ENV PYTHONPATH="/app:/app/lib:${PYTHONPATH}"

# Environment variables
ENV CONFIG_PATH=/app/config.json
ENV MODEL_DIR=/app/model
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
