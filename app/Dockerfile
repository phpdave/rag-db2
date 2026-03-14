FROM python:3.11-slim

WORKDIR /app

# System deps: gcc for psycopg2, wget for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create PDF cache directory
RUN mkdir -p /app/pdfs

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
