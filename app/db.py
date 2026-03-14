"""
db.py — PostgreSQL / pgvector connection helpers.
"""

import os
import psycopg2
from pgvector.psycopg2 import register_vector

_DATABASE_URL = os.environ["DATABASE_URL"]


def get_conn():
    """Return a new psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(_DATABASE_URL)
    register_vector(conn)
    return conn


def delete_chunks(source: str) -> None:
    """Delete all chunks and the ingested_sources record for a source."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            cur.execute("DELETE FROM ingested_sources WHERE source = %s", (source,))
        conn.commit()


def is_already_ingested(source: str) -> bool:
    """Return True if this source has already been ingested."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ingested_sources WHERE source = %s",
                (source,),
            )
            return cur.fetchone() is not None


def mark_ingested(source: str, chunk_count: int) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingested_sources (source, chunk_count)
                VALUES (%s, %s)
                ON CONFLICT (source) DO UPDATE SET
                    chunk_count = EXCLUDED.chunk_count,
                    ingested_at = NOW()
                """,
                (source, chunk_count),
            )
        conn.commit()


def insert_chunks(rows: list[dict]) -> None:
    """
    Bulk-insert chunk rows.
    Each row: {source, page_num, chunk_index, content, embedding, metadata}
    """
    if not rows:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (source, page_num, chunk_index, content, embedding, metadata)
                VALUES
                    (%(source)s, %(page_num)s, %(chunk_index)s,
                     %(content)s, %(embedding)s, %(metadata)s)
                """,
                rows,
            )
        conn.commit()


def similarity_search(
    embedding,
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Return the top-k chunks closest to `embedding` using cosine similarity.
    Optionally filter by source document.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if source_filter:
                cur.execute(
                    """
                    SELECT id, source, page_num, chunk_index, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE source = %s AND LENGTH(content) > 100
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, source_filter, embedding, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source, page_num, chunk_index, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE LENGTH(content) > 100
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def fulltext_search(query: str, top_k: int = 8) -> list[dict]:
    """
    Keyword-based fallback using PostgreSQL full-text search.
    Useful when the semantic query misses exact function / column names.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, page_num, chunk_index, content, metadata,
                       ts_rank(to_tsvector('english', content),
                               plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE LENGTH(content) > 100
                  AND to_tsvector('english', content)
                      @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_sources() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source, chunk_count, ingested_at FROM ingested_sources ORDER BY ingested_at DESC"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
