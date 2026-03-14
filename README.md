# Db2 for i SQL Reference — RAG Setup

Ask natural-language questions against the IBM Db2 for i 7.5 SQL Reference
(or any other PDF) using pgvector + Claude.

## Stack

| Component | Technology |
|-----------|-----------|
| Vector store | PostgreSQL 16 + pgvector |
| Embeddings | `BAAI/bge-large-en-v1.5` (local, CPU/GPU) |
| Generation | Anthropic Claude (via API) |
| API | FastAPI |
| PDF parsing | pdfplumber |
| Search | Hybrid: cosine similarity + PostgreSQL FTS |

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- An Anthropic API key

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Start the stack

```bash
docker compose up -d
```

This starts:
- **postgres** — pgvector database (port 5432)
- **app** — FastAPI server (port 8000)

The embedding model (~1.3 GB) downloads on first start and is cached in a
Docker volume so subsequent restarts are fast.

### 4. Ingest the PDF

The IBM i SQL Reference is ~1,739 pages. Ingestion takes **15–30 minutes**
on a modern CPU (embedding is the bottleneck).

```bash
# Trigger ingestion via the API
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.ibm.com/docs/es/ssw_ibm_i_75/pdf/rbafzpdf.pdf"}'

# Poll for status
curl "http://localhost:8000/ingest/status?source=https://www.ibm.com/docs/es/ssw_ibm_i_75/pdf/rbafzpdf.pdf"
```

Or run ingestion directly in the container (useful for watching progress):

```bash
docker exec -it rag-app python ingest.py
```

To ingest a **local PDF** instead:

```bash
# Copy your PDF into the container volume
docker cp myfile.pdf rag-app:/app/pdfs/myfile.pdf

# Trigger ingestion
curl -X POST http://localhost:8000/ingest/file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/app/pdfs/myfile.pdf"}'
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What columns does QSYS2.SYSTABLESTAT have?"}'
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I get the on-disk size of a table including indexes?"}'
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between DATA_SIZE and VARIABLE_LENGTH_SIZE?"}'
```

The response includes:
- `answer` — Claude's answer grounded in the docs
- `sources` — the chunks used, with page numbers and scores
- `usage` — token counts

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/sources` | List ingested documents |
| `POST` | `/ingest/url` | Ingest PDF from URL (background) |
| `POST` | `/ingest/file` | Ingest local PDF file (background) |
| `GET` | `/ingest/status?source=...` | Check ingestion status |
| `POST` | `/query` | Ask a question |

Interactive docs: http://localhost:8000/docs

---

## Tuning

### Chunk size

Smaller chunks (200–300 words) give more precise retrieval but less context
per chunk. Larger chunks (500–600) give more context but can dilute relevance.
Edit `CHUNK_SIZE` in `.env` and re-ingest with `--force`.

### top_k

The `/query` endpoint accepts a `top_k` parameter (default 8). Increase it for
broader questions, decrease it for narrow lookups. More chunks = more tokens
sent to Claude.

### Embedding model

`BAAI/bge-large-en-v1.5` is the default (1024-dim, ~1.3 GB).
For a lighter setup use `BAAI/bge-base-en-v1.5` (768-dim, ~440 MB) —
update `EMBEDDING_MODEL` in `.env` **before** ingesting (dimensions must match).

### GPU acceleration

If you have an NVIDIA GPU, install the `nvidia-container-toolkit` and add this
to the `app` service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Troubleshooting

**Ingestion is slow**
That's expected — 1,739 pages with 400-word chunks = ~4,000 chunks, each
requiring an embedding forward pass. On CPU, ~15–30 minutes total.

**`ANTHROPIC_API_KEY` not set error**
Make sure your `.env` file exists and Docker Compose is picking it up:
`docker compose --env-file .env up -d`

**pgvector extension missing**
The `pgvector/pgvector:pg16` image includes the extension. If you use a
plain `postgres` image it won't be there.

**Re-ingest after changing chunk size**
```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "...", "force": true}'
```
