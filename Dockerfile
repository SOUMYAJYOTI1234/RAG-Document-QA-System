FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p data vector_store logs

# Expose ports: 8000 = FastAPI, 8501 = Streamlit
EXPOSE 8000 8501

# Default: start FastAPI (docker-compose overrides per service)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
