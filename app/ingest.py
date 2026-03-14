"""
ingest.py — Download a PDF, chunk it, embed it, and store in pgvector.

Usage (inside container):
    python ingest.py                          # uses PDF_URL env var
    python ingest.py --url https://...        # override URL
    python ingest.py --file /app/pdfs/doc.pdf # local file

The script is idempotent: re-running it for the same source is a no-op
unless you pass --force.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import db

# ──────────────────────────────────────────────────────────────────────────────
# Config (overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", 400))      # tokens (approx words)
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", 50))
BATCH_SIZE      = 32   # embeddings per forward pass
PDF_DIR         = Path("/app/pdfs")

# ──────────────────────────────────────────────────────────────────────────────
# PDF download
# ──────────────────────────────────────────────────────────────────────────────

def download_pdf(url: str) -> Path:
    """Download a PDF to the local cache and return its path."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # Derive a stable filename from a hash of the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    name     = url.split("/")[-1] or "document.pdf"
    dest     = PDF_DIR / f"{url_hash}_{name}"

    if dest.exists():
        print(f"[ingest] PDF already cached at {dest}")
        return dest

    print(f"[ingest] Downloading {url} …")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,*/*",
    }
    r = requests.get(url, stream=True, timeout=120, headers=headers)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[ingest] Saved to {dest}")
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# PDF → text extraction
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_RE  = re.compile(r'^\s*(?:IBM\s+i\s*:\s*)?Db2\s+for\s+i\s+SQL\s+Reference\s*$', re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$')


def _clean_text(text: str) -> str:
    """Strip running headers, footers, and bare page numbers from extracted text."""
    lines = [
        line for line in text.splitlines()
        if not _HEADER_RE.match(line) and not _PAGE_NUM_RE.match(line)
    ]
    return "\n".join(lines).strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Return list of {page_num, text} dicts.
    pdfplumber handles multi-column layouts better than pypdf.
    Headers/footers are stripped before returning.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if not text:
                continue
            text = _clean_text(text)
            if len(text.split()) >= 20:   # skip near-empty pages
                pages.append({"page_num": i + 1, "text": text})
    print(f"[ingest] Extracted text from {len(pages)} pages")
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def chunk_page(page_num: int, text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Split a page's text into overlapping chunks that respect sentence boundaries.
    Sentences are identified by splitting on '. ', '.\n', '!\n', '?\n'.
    """
    # Split into sentences while keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks  = []
    idx     = 0
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > chunk_size and current:
            chunks.append({
                "page_num":    page_num,
                "chunk_index": idx,
                "content":     " ".join(current),
            })
            idx += 1
            # Overlap: keep sentences from the tail that fit within overlap words
            overlap_sents: list[str] = []
            overlap_words = 0
            for s in reversed(current):
                w = len(s.split())
                if overlap_words + w > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_words += w
            current = overlap_sents
            current_words = overlap_words
        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append({
            "page_num":    page_num,
            "chunk_index": idx,
            "content":     " ".join(current),
        })

    return chunks


def build_chunks(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Convert all pages to chunks, adding section detection metadata."""
    all_chunks = []
    global_idx = 0

    for page in pages:
        page_chunks = chunk_page(page["page_num"], page["text"], chunk_size, overlap)
        for c in page_chunks:
            c["chunk_index"] = global_idx
            c["metadata"]    = detect_section(c["content"])
            global_idx      += 1
        all_chunks.extend(page_chunks)

    print(f"[ingest] Built {len(all_chunks)} chunks "
          f"(size={chunk_size} words, overlap={overlap})")
    return all_chunks


# Simple heuristic section detector for the IBM i SQL Reference structure
_SECTION_PATTERNS = [
    (r"^Chapter\s+\d+\.",                "chapter"),
    (r"^Appendix\s+[A-Z]\.",             "appendix"),
    (r"^(CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE|GRANT|REVOKE|CALL)\b", "statement"),
    (r"^[A-Z_]{2,}\s*\(",               "function"),
    (r"^SYS[A-Z]+\b",                   "catalog_view"),
]

def detect_section(text: str) -> dict:
    first_line = text.split("\n")[0].strip()
    for pattern, label in _SECTION_PATTERNS:
        if re.match(pattern, first_line, re.IGNORECASE):
            return {"section_type": label, "heading": first_line[:80]}
    return {"section_type": "body", "heading": ""}


# ──────────────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> SentenceTransformer:
    print(f"[ingest] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[ingest] Model loaded  (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Add an 'embedding' key to each chunk dict."""
    texts = [c["content"] for c in chunks]

    print(f"[ingest] Embedding {len(texts)} chunks in batches of {batch_size} …")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        # BGE models benefit from a query prefix; for passage encoding use empty prefix
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def ingest(source: str, pdf_path: Path, force: bool = False) -> None:
    if db.is_already_ingested(source):
        if not force:
            print(f"[ingest] '{source}' already ingested. Pass --force to re-ingest.")
            return
        print(f"[ingest] --force: deleting existing chunks for '{source}' …")
        db.delete_chunks(source)

    # 1. Extract text
    pages = extract_pages(pdf_path)
    if not pages:
        print("[ingest] ERROR: No text extracted from PDF.")
        sys.exit(1)

    # 2. Chunk
    chunks = build_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Embed
    model  = load_model(EMBEDDING_MODEL)
    chunks = embed_chunks(model, chunks)

    # 4. Persist
    print("[ingest] Writing chunks to database …")
    rows = [
        {
            "source":      source,
            "page_num":    c["page_num"],
            "chunk_index": c["chunk_index"],
            "content":     c["content"],
            "embedding":   c["embedding"],
            "metadata":    json.dumps(c["metadata"]),
        }
        for c in chunks
    ]

    # Insert in batches of 500 to avoid huge transactions
    batch = 500
    for i in tqdm(range(0, len(rows), batch), desc="Inserting"):
        db.insert_chunks(rows[i : i + batch])

    db.mark_ingested(source, len(rows))
    print(f"[ingest] Done. {len(rows)} chunks stored for source '{source}'.")


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--url",  default=os.environ.get("PDF_URL"), help="URL of PDF to download")
    group.add_argument("--file", help="Path to a local PDF file")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already done")
    args = parser.parse_args()

    if args.file:
        pdf_path = Path(args.file)
        source   = pdf_path.name
        if not pdf_path.exists():
            print(f"ERROR: File not found: {pdf_path}")
            sys.exit(1)
    elif args.url:
        pdf_path = download_pdf(args.url)
        source   = args.url
    else:
        print("ERROR: Provide --url or --file, or set PDF_URL environment variable.")
        sys.exit(1)

    ingest(source, pdf_path, force=args.force)


if __name__ == "__main__":
    main()
