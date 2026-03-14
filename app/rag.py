"""
rag.py — Retrieval-augmented generation using pgvector + Claude.

Hybrid search: cosine similarity (semantic) + BM25-style full-text (keyword).
Results are re-ranked by a weighted combination before being sent to Claude.
"""

import os
from dataclasses import dataclass

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

import db

# ──────────────────────────────────────────────────────────────────────────────
# Singleton model loader (avoid reloading on every request)
# ──────────────────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        _model = SentenceTransformer(model_name)
    return _model


def embed_query(text: str) -> list[float]:
    """
    BGE models recommend prepending a query instruction for retrieval tasks.
    For passage retrieval the instruction is: 'Represent this sentence for
    searching relevant passages: '
    """
    instruction = "Represent this sentence for searching relevant passages: "
    model = _get_model()
    vec = model.encode(instruction + text, normalize_embeddings=True)
    return vec.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid search + re-ranking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id:    int
    source:      str
    page_num:    int
    content:     str
    score:       float
    score_type:  str   # "semantic" | "fulltext" | "hybrid"


def _is_index_chunk(content: str) -> bool:
    """
    Detect index/TOC pages: they have a high ratio of bare page numbers
    ('word 1234' entries) relative to total words.
    """
    import re
    numbers = re.findall(r'\b\d{3,4}\b', content)
    words   = content.split()
    return len(numbers) / max(len(words), 1) > 0.12


def hybrid_search(
    query: str,
    top_k: int = 8,
    semantic_weight: float = 0.7,
    fulltext_weight: float = 0.3,
    min_score: float = 0.4,
) -> list[SearchResult]:
    """
    Combine semantic (vector) and full-text search results.
    Scores are normalised to [0, 1] then merged with weighted sum.
    Index-like chunks and low-scoring results are filtered out.
    """
    embedding = embed_query(query)

    sem_rows = db.similarity_search(embedding, top_k=top_k * 4)
    ft_rows  = db.fulltext_search(query,         top_k=top_k * 4)

    # Build score maps keyed by chunk id
    sem_scores: dict[int, float] = {}
    ft_scores:  dict[int, float] = {}
    all_rows:   dict[int, dict]  = {}

    # Normalise semantic scores (already cosine, range ≈ 0–1)
    if sem_rows:
        max_s = max(r["score"] for r in sem_rows) or 1.0
        for r in sem_rows:
            if not _is_index_chunk(r["content"]):
                sem_scores[r["id"]] = r["score"] / max_s
                all_rows[r["id"]]   = r

    # Normalise full-text scores (ts_rank, range 0–1)
    if ft_rows:
        max_f = max(r["score"] for r in ft_rows) or 1.0
        for r in ft_rows:
            if not _is_index_chunk(r["content"]):
                ft_scores[r["id"]] = r["score"] / max_f
                all_rows[r["id"]]  = r

    # Merge
    combined: dict[int, float] = {}
    for cid in set(list(sem_scores.keys()) + list(ft_scores.keys())):
        combined[cid] = (
            semantic_weight * sem_scores.get(cid, 0.0)
            + fulltext_weight * ft_scores.get(cid, 0.0)
        )

    # Filter low-scoring results, sort, take top_k
    top_ids = sorted(
        (cid for cid, score in combined.items() if score >= min_score),
        key=combined.get,
        reverse=True,
    )[:top_k]

    results = []
    for cid in top_ids:
        row = all_rows[cid]
        results.append(SearchResult(
            chunk_id   = cid,
            source     = row["source"],
            page_num   = row["page_num"],
            content    = row["content"],
            score      = combined[cid],
            score_type = "hybrid",
        ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert on IBM Db2 for i (AS/400) SQL. You answer questions \
accurately and concisely based ONLY on the reference documentation excerpts \
provided below. When quoting syntax or function signatures, use code blocks. \
If the answer is not covered by the excerpts, say so clearly rather than \
guessing. When relevant, cite the page number from the documentation.\
"""

_SYSTEM_PROMPT_EXECUTE = """\
You are an expert on IBM Db2 for i (AS/400) SQL with access to a live IBM i system.
Use the documentation excerpts below as a reference when helpful, but you are NOT
limited to them — you can write SQL for any table or schema the user mentions.

Rules:
- Always produce ONE complete, executable SELECT statement in a ```sql code block.
- Use the exact table/schema names the user provides.
- For catalog queries, prefer QSYS2 views (QSYS2.SYSTABLES, QSYS2.SYSCOLUMNS, etc.).
- Do not add FETCH FIRST clauses — the system adds a row limit automatically.
- Be concise. Show the SQL first, then a brief explanation.\
"""


def build_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[Excerpt {i} — page {r.page_num}, score {r.score:.3f}]\n"
            f"{r.content}"
        )
    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer:    str
    sources:   list[SearchResult]
    prompt_tokens:    int
    response_tokens:  int
    sql_results: dict | None = None


def ask(
    question: str,
    top_k: int = 8,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2048,
    execute_sql: bool = False,
) -> RAGResponse:
    """
    Full RAG pipeline: embed → retrieve → generate.
    """
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = hybrid_search(question, top_k=top_k)

    if not results:
        return RAGResponse(
            answer  = "No relevant documentation found for your query.",
            sources = [],
            prompt_tokens   = 0,
            response_tokens = 0,
        )

    context      = build_context(results)
    user_message = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}"
    )

    system = _SYSTEM_PROMPT_EXECUTE if execute_sql else _SYSTEM_PROMPT

    response = client.messages.create(
        model      = model,
        max_tokens = max_tokens,
        system     = system,
        messages   = [{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    sql_results = None
    if execute_sql:
        import ibmi
        sql = ibmi.extract_all_sql(answer)
        if sql:
            sql_results = ibmi.run_all_statements(sql)

    return RAGResponse(
        answer          = answer,
        sources         = results,
        prompt_tokens   = response.usage.input_tokens,
        response_tokens = response.usage.output_tokens,
        sql_results     = sql_results,
    )
