# Custom Airflow image with ML dependencies
# Provides scikit-learn, TensorFlow, MLflow, and other ML packages for the pipeline DAGs

ARG AIRFLOW_VERSION=3.1.6
ARG PYTHON_VERSION=3.10

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

# Install Python packages for ML pipeline
RUN pip install --no-cache-dir \
    scikit-learn \
    mlflow>=2.10.0 \
    matplotlib \
    seaborn \
    pandas \
    joblib \
    tensorflow
