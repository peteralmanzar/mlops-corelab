# Custom Airflow image with ML dependencies
# Provides scikit-learn, TensorFlow, MLflow, and other ML packages for the pipeline DAGs

ARG AIRFLOW_VERSION=3.1.6
ARG PYTHON_VERSION=3.10

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

# Install Python packages for ML pipeline as airflow user
# Unset PIP_CONSTRAINT to bypass Airflow's hash-checked constraints
RUN PIP_CONSTRAINT="" pip install --no-cache-dir --user \
    scikit-learn \
    "mlflow>=2.10.0" \
    matplotlib \
    seaborn \
    pandas \
    joblib \
    tensorflow
