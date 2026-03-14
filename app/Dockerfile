FROM python:3.11-slim

WORKDIR /app

# System deps + IBM i Access ODBC Driver (for IBM i / AS400 — NOT ibm_db which is LUW/z/OS)
# Repo uses trusted=yes (no GPG key needed): https://public.dhe.ibm.com/software/ibmi/products/odbc/debs/dists/1.1.0/
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    wget \
    curl \
    apt-transport-https \
    unixodbc \
    unixodbc-dev \
    && echo "deb [trusted=yes] https://public.dhe.ibm.com/software/ibmi/products/odbc/debs 1.1.0 main" \
       > /etc/apt/sources.list.d/ibm-iaccess.list \
    && apt-get update && apt-get install -y --no-install-recommends ibm-iaccess \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — PyTorch CUDA index takes priority so we get the
# GPU-enabled wheel; PyPI is the fallback for everything else.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

# Copy application code
COPY . .

# Create PDF cache directory
RUN mkdir -p /app/pdfs

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
