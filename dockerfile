# Base Python image
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Ensure model is included (may be excluded by .dockerignore)
COPY src/serving/model /app/src/serving/model

# Flatten MLflow artifacts for inference
COPY src/serving/model/m-350da676ed7b42818b2c80b74d5678cb/artifacts /app/model

# Enable clean imports and real-time logs
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Start FastAPI app
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]