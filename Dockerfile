# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY src/ ./src/
# ENV PYTHONPATH=/app/src

# NEW --------

# churn-mlops/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY setup.py /app/
COPY src/ ./src/

# Install code so 'mlops' package is available
RUN pip install -e .

ENV PYTHONPATH=/app/src
