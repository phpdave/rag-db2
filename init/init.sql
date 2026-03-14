-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main document chunks table
-- 1024 dimensions matches BAAI/bge-large-en-v1.5
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,           -- PDF filename or URL
    page_num    INTEGER,                 -- Page number in the PDF
    chunk_index INTEGER NOT NULL,        -- Position within the document
    content     TEXT NOT NULL,           -- Raw text of this chunk
    embedding   vector(1024),            -- Embedding vector
    metadata    JSONB DEFAULT '{}'::jsonb, -- Any extra metadata
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for fast approximate nearest-neighbour search
-- lists = sqrt(num_rows) is a good starting point; tune after ingestion
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text index on content for hybrid search fallback
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin (to_tsvector('english', content));

-- Track which PDFs have been ingested so re-runs are idempotent
CREATE TABLE IF NOT EXISTS ingested_sources (
    id          SERIAL PRIMARY KEY,
    source      TEXT UNIQUE NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);
