FROM jupyter/datascience-notebook:python-3.10

USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy pinned requirements and install
COPY requirements.txt /tmp/requirements.txt

USER ${NB_UID}

# Install Python libraries from requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt