#!/usr/bin/env python3
"""
Dwengo Learning Content — Production-Ready RAG Web Service
=========================================================

Serve a retrieval‑augmented chat API over HTTP for Dwengo's markdown learning
content (learning objects + paths). Designed to run in real time on an NVIDIA A100.

Highlights
----------
- FastAPI app with typed request/response models
- Multilingual retrieval using `intfloat/multilingual-e5-large-instruct` (E5)
- FAISS vector index persisted to disk (+ metadata sidecar)
- Repo sync & automatic (re)index on startup; `/v1/reindex` endpoint
- Optional API key via `X-API-Key`
- CORS support, `/health` and `/version` endpoints
- Token streaming via `text/event-stream` (SSE‑style) or non‑streaming JSON
- Single worker recommended for GPU (avoid multiple model copies)

Run
---
```bash
pip install --upgrade fastapi uvicorn[standard] gitpython pyyaml python-frontmatter \
  sentence-transformers faiss-cpu transformers accelerate torch rich

# Environment (adjust as needed)
export DWENGO_REPO_URL=https://github.com/dwengovzw/learning_content
export DATA_DIR=./learning_content
export INDEX_PATH=./dwengo_faiss.index
export META_PATH=./dwengo_faiss.meta.json
export EMB_MODEL=intfloat/multilingual-e5-large-instruct
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SYNC_REPO=true
export API_KEY=changeme   # optional; omit to disable
export CORS_ORIGINS=*     # or comma-separated list

# Start (single worker for GPU)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```
"""

from __future__ import annotations

import os
import io
import re
import json
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import faiss  # type: ignore
import frontmatter  # type: ignore
from git import Repo  # type: ignore
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import torch
from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # type: ignore

# -----------------------------
# Config
# -----------------------------
APP_VERSION = "1.0.0"
LOG = logging.getLogger("dwengo_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

REPO_URL = os.getenv("DWENGO_REPO_URL", "https://github.com/dwengovzw/learning_content")
DATA_DIR = Path(os.getenv("DATA_DIR", "./learning_content"))
# Additional content source: Jupyter notebooks repo
NOTEBOOKS_REPO_URL = os.getenv("DWENGO_NOTEBOOKS_REPO_URL", "https://github.com/dwengovzw/PythonNotebooks")
NOTEBOOKS_DIR = Path(os.getenv("DWENGO_NOTEBOOKS_DIR", "./python_notebooks"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./dwengo_faiss2.index"))
META_PATH = Path(os.getenv("META_PATH", "./dwengo_faiss2.meta.json"))
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
#LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-3-12b-it")
SYNC_REPO = os.getenv("SYNC_REPO", "true").lower() == "true"
API_KEY = os.getenv("API_KEY")  # optional
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

SYSTEM_PROMPT = (
    "Je bent de helpende onderwijsassistent van Dwengo vzw. Dwengo is een organisatie die gratis lesmateriaal maakt over AI, robotica, computationeel denken, STEM en physical computing."
    "Dwengo richt zich op leerkrachten en leerlingen in het secundair onderwijs in Vlaanderen en Nederland."
    "Beantwoord in het Nederlands wanneer de vraag in het Nederlands is. "
    "Gebruik de contextfragmenten uit het Dwengo-leermateriaal om nauwkeurige, brongebonden antwoorden te geven. "
    "Neem geen links op in je antwoorden. Verwijs niet naar bronnen."
    "Als je het antwoord niet met zekerheid weet vanuit de context, zeg dan eerlijk dat je het niet zeker weet."
    "Vervang eventuele markdown in je antwoord door html. Geef codefragmenten weern in een codeblok met de juiste taal."
)
INSTRUCT_EMBED_PREFIX = "Instruct: Represent the query for retrieving supporting passages: "
INSTRUCT_PASSAGE_PREFIX = "Passage: "

MD_EXTS = {".md", ".markdown"}

API_KEY_HEADER_KEY = "DWENGO-API-KEY"
DWENGO_WEBSITE_CONTENT_TYPE = "DWENGO_WEBSITE_CONTENT"
DWENGO_PYTHON_NOTEBOOK_TYPE = "DWENGO_PYTHON_NOTEBOOK"

# Notebook ingestion controls
NB_INCLUDE_CODE = os.getenv("DWENGO_NB_INCLUDE_CODE", "false").lower() == "true"
NB_MAX_BYTES = int(os.getenv("DWENGO_NB_MAX_BYTES", "4000000"))  # 4 MB
NB_MAX_CODE_CHARS = int(os.getenv("DWENGO_NB_MAX_CODE_CHARS", "2000"))
NB_MAX_TEXT_CHARS = int(os.getenv("DWENGO_NB_MAX_TEXT_CHARS", "120000"))

# -----------------------------
# Security
# -----------------------------
# class APIKeyAuth:
#     def __init__(self, header_name: str = API_KEY_HEADER_KEY):
#         self.header_name = header_name

#     async def __call__(self, request: Request):
#         if not API_KEY:
#             return  # auth disabled
#         key = request.headers.get(self.header_name)
#         print(f"Received API key: {key}")
#         if key != API_KEY:
#             raise HTTPException(status_code=401, detail="Invalid API key")
        
async def require_api_key(request: Request):
    if not API_KEY:
        return
    key = request.headers.get(API_KEY_HEADER_KEY)
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

#require_api_key = APIKeyAuth()

# -----------------------------
# Data IO & indexing
# -----------------------------

def clone_or_update_repo(repo_url: str, dest_dir: Path) -> None:
    print(f"[repo] ensure {dest_dir} from {repo_url}")
    if dest_dir.exists() and (dest_dir / ".git").exists():
        LOG.info("Updating repo at %s", dest_dir)
        print(f"[repo] update {dest_dir}")
        repo = Repo(str(dest_dir))
        repo.remotes.origin.fetch()
        repo.git.reset("--hard", "origin/HEAD")
    else:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        LOG.info("Cloning repo to %s", dest_dir)
        print(f"[repo] clone into {dest_dir}")
        Repo.clone_from(repo_url, str(dest_dir))


def read_markdown_files(root: Path) -> List[Dict[str, Any]]:
    print(f"[ingest-md] scanning {root}")
    items: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in MD_EXTS and p.is_file():
            try:
                post = frontmatter.load(str(p))
                metadata = post.metadata or {}
                text = str(post.content)
                items.append({
                    "path": str(p),
                    "text": text,
                    "hruid": metadata.get("hruid"),
                    "language": metadata.get("language"),
                    "title": metadata.get("title"),
                    "estimated_time": metadata.get("estimated_time"),
                    "target_ages": metadata.get("target_ages"),
                    "_source_type": DWENGO_WEBSITE_CONTENT_TYPE,
                })
            except Exception as e:
                LOG.warning("Could not parse %s: %s", p, e)
    print(f"[ingest-md] parsed {len(items)} files")
    return items

def read_notebook_files(root: Path) -> List[Dict[str, Any]]:
    """Read notebooks listed in PythonNotebooks.json (in repo root),
    extract textual content and return items ready for embedding.

    For each JSON entry, use the entry key as hruid, Name as title, Description as description,
    and include each .ipynb from the Files array. Non-notebook files are ignored.
    """
    items: List[Dict[str, Any]] = []
    index_path = root / "PythonNotebooks.json"
    print(f"[ingest-nb] index: {index_path}")
    if not index_path.exists():
        LOG.warning("Notebook index JSON not found at %s; skipping notebook ingestion.", index_path)
        return items

    try:
        notebook_index = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(notebook_index, dict):
            print(f"[ingest-nb] entries: {len(notebook_index)}")
    except Exception as e:
        LOG.error("Failed to parse %s: %s", index_path, e)
        return items

    def detect_language(entry_key: str, base_path: Optional[str], rel_path: str) -> str:
        if entry_key.endswith("_en") or (base_path and base_path.startswith("en/")) or rel_path.startswith("en/"):
            return "en"
        return "nl"

    total_listed = 0
    total_parsed = 0
    for entry_key, entry in (notebook_index.items() if isinstance(notebook_index, dict) else []):
        print(f"[ingest-nb] entry: {entry_key}")
        try:
            files = entry.get("Files", []) if isinstance(entry, dict) else []
            base_path = entry.get("BasePath") if isinstance(entry, dict) else None
            title = entry.get("Name") if isinstance(entry, dict) else None
            description = entry.get("Description") if isinstance(entry, dict) else None
            print(f"[ingest-nb]  description: {description}")
            for rel in files:
                if not isinstance(rel, str) or not rel.endswith(".ipynb"):
                    continue
                p = root / rel
                total_listed += 1
                if not p.exists() or not p.is_file():
                    LOG.warning("Listed notebook not found: %s (entry %s)", p, entry_key)
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    
                    cells = data.get("cells", [])
                    parts: List[str] = []
                    for cell in cells:
                        ctype = cell.get("cell_type")
                        src = cell.get("source", [])
                        src_text = "".join(src) if isinstance(src, list) else str(src)
                        if ctype == "markdown":
                            parts.append(src_text)
                        elif ctype == "code":
                            if src_text.strip():
                                parts.append(f"\n```python\n{src_text}\n```\n")
                    text = "\n\n".join([t for t in parts if t.strip()])
                    print(f"[ingest-nb]  text: {text[:100]}... (truncated)")
                    lang = detect_language(entry_key, base_path, rel)
                    print(f"[ingest-nb]  parsed {p} (lang={lang}, chunks TBD)")
                    items.append({
                        "path": str(p),
                        "text": text,
                        "hruid": entry_key,  # use JSON entry key as id
                        "language": lang,
                        "title": title or p.stem,
                        "description": description,
                        "_source_type": DWENGO_PYTHON_NOTEBOOK_TYPE,
                    })
                    total_parsed += 1
                    print(f"[ingest-nb]  parsed {p} total_parsed {total_parsed} (lang={lang}, chunks TBD)")
                except Exception as e:
                    LOG.warning("Could not parse notebook %s (entry %s): %s", p, entry_key, e)
                    print(f"[ingest-nb]  could not parse {p} (entry {entry_key}): {e}")
        except Exception as e:
            LOG.warning("Skipping entry %s due to error: %s", entry_key, e)
            print(f"[ingest-nb]  skipping entry {entry_key} due to error: {e}")
    print(f"[ingest-nb] listed {total_listed}, parsed {total_parsed}, items {len(items)}")
    return items

def read_learning_paths(root: Path) -> List[Dict[str, Any]]:
    paths = []
    for p in root.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "nodes" in data and "hruid" in data:
                paths.append({"path": str(p), **data})
        except Exception:
            # Not a learning path JSON, skip
            pass
    return paths


def split_markdown(text: str, max_tokens: int = 400) -> List[str]:
    """Greedy, header-aware splitter. Keeps headers with following text.
    Not token-accurate, but good enough. max_tokens approximates words.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split by top-level headers first
    parts = re.split(r"\n(?=#+\s)" , text)

    chunks = []
            for rel in files:
        words = part.split()
        if not words:
            continue
        cur = []
        count = 0
        for w in words:
            cur.append(w)
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                if NB_MAX_BYTES and size and size > NB_MAX_BYTES:
                    LOG.warning("Skipping large notebook (%d bytes): %s", size, p)
                    continue
            count += 1
            if count >= max_tokens:
                chunks.append(" ".join(cur))
                cur = []
                    total_chars = 0
                count = 0
        if cur:
            chunks.append(" ".join(cur))

    # Fallback if no headers
                            if src_text:
                                if NB_MAX_TEXT_CHARS and (total_chars + len(src_text) > NB_MAX_TEXT_CHARS):
                                    remain = max(0, NB_MAX_TEXT_CHARS - total_chars)
                                    parts.append(src_text[:remain])
                                    total_chars += remain
                                    break
                                parts.append(src_text)
                                total_chars += len(src_text)
        words = text.split()
                            if NB_INCLUDE_CODE and src_text.strip():
                                code_snip = src_text[:NB_MAX_CODE_CHARS] if NB_MAX_CODE_CHARS else src_text
                                block = f"\n```python\n{code_snip}\n```\n"
                                if NB_MAX_TEXT_CHARS and (total_chars + len(block) > NB_MAX_TEXT_CHARS):
                                    remain = max(0, NB_MAX_TEXT_CHARS - total_chars)
                                    parts.append(block[:remain])
                                    total_chars += remain
                                    break
                                parts.append(block)
                                total_chars += len(block)
            cur.append(w)
            if len(cur) >= max_tokens:
                chunks.append(" ".join(cur)); cur = []
        if cur:
            chunks.append(" ".join(cur))

    return chunks


class VectorStore:
    def __init__(self, dim: int):
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.docs: List[Dict[str, Any]] = []  # each has: text, title, path, language

    def add(self, embs: np.ndarray, metas: List[Dict[str, Any]]):
        assert embs.shape[0] == len(metas)
        # faiss typings vary; at runtime this accepts a (n,d) float32 array
        self.index.add(embs.astype(np.float32))  # type: ignore[arg-type]
        self.docs.extend(metas)

    def save(self, index_path: Path, meta_path: Path):
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps(self.docs, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(index_path: Path, meta_path: Path) -> "VectorStore":
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vs = VectorStore(index.d)
        vs.index = index
        vs.docs = meta
        return vs

# -----------------------------
# App & global state
# -----------------------------
app = FastAPI(title="Dwengo RAG Service", version=APP_VERSION)

# Disable cors since it is handled by the reverse proxy (nginx)
#LOG.info(f"cors: {CORS_ORIGINS}")
#print(f"cors: {CORS_ORIGINS}")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

_embedder: Optional[SentenceTransformer] = None
_vstore: Optional[VectorStore] = None
_tokenizer = None
_model = None
_generate_lock = asyncio.Lock()  # serialize access to GPU model

# -----------------------------
# Schemas
# -----------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    k: int = Field(6, ge=1, le=20)
    max_new_tokens: int = Field(384, ge=16, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False
    history: Optional[List[Tuple[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    retrieved: List[Dict[str, Any]]

class ReindexRequest(BaseModel):
    force: bool = True

# -----------------------------
# Helpers
# -----------------------------
async def ensure_ready():
    if any(x is None for x in (_embedder, _vstore, _tokenizer, _model)):
        raise HTTPException(status_code=503, detail="Service not initialized yet")


def embed_passages(passages: List[str]) -> np.ndarray:
    assert _embedder is not None
    print(f"[embed] encoding {len(passages)} passages …")
    embs = _embedder.encode(passages, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    print(f"[embed] done: shape={embs.shape}")
    return embs.astype(np.float32)


def retrieve(query: str, k: int) -> List[Dict[str, Any]]:
    assert _embedder is not None and _vstore is not None
    q = INSTRUCT_EMBED_PREFIX + query
    print(f"[retrieve] k={k}, query='{query[:80]}{'…' if len(query)>80 else ''}'")
    q_emb = _embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = _vstore.index.search(q_emb, k)  # type: ignore[call-arg]
    hits: List[Dict[str, Any]] = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc = dict(_vstore.docs[idx])
        doc["score"] = float(score)
        hits.append(doc)
    if hits:
        print(f"[retrieve] hits={len(hits)}, top_score={hits[0]['score']:.4f}")
    else:
        print("[retrieve] no hits")
    return hits


def build_prompt(query: str, contexts: List[Dict[str, Any]], history: Optional[List[Tuple[str, str]]] = None) -> str:
    print(f"[prompt] contexts={len(contexts)}, history={'yes' if history else 'no'}")
    ctx_blocks = []
    for c in contexts:
        header = f"Bron: {Path(c.get('path',''))}\nTitel: {c.get('title')}\nTaal: {c.get('language')}".strip()
        text = c.get("text", "").strip()
        ctx_blocks.append(f"[{header}]\n{text}")
    joined_ctx = "\n\n---\n\n".join(ctx_blocks)

    hist_txt = ""
    if history:
        last = history[-4:]
        hist_txt = "\n".join([f"<|user|> {u}\n<|assistant|> {a}" for u, a in last])

    prompt = (
        (hist_txt + "\n") if hist_txt else ""
        ) + f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n" \
        f"<|user|>\nVraag: {query}\n\nHier zijn relevante fragmenten. Gebruik ze als betrouwbare bron.\n\n{joined_ctx}\n\nGeef een beknopt antwoord citeer geen bronnen.\n</|user|>\n"
    print(f"[prompt] length={len(prompt)}")
    return prompt


async def generate_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float, stream: bool, source_links: List[str]):
    assert _model is not None and _tokenizer is not None
    print(f"[gen] stream={stream}, max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}")
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
    )

    async with _generate_lock:
        loop = asyncio.get_event_loop()
        thread = asyncio.to_thread(_model.generate, **gen_kwargs)  # type: ignore[arg-type]

        def stream_tokens():
            for token in streamer:
                yield token

        gen_task = loop.create_task(thread)
        lp_links_html =  f"<br>Bronnen:<br>{"<br>".join(source_links)}"

        if stream:
            async def agen():
                token_count = 0
                for tok in stream_tokens():
                    token_count += 1
                    yield f"data: {tok}\n\n"
                await gen_task
                if source_links:
                    yield f"data: {lp_links_html}\n\n"
                    print(f"Streamed source links")
                print(f"[gen] streamed tokens={token_count}, appended_links={bool(lp_links_html)}")
            return StreamingResponse(agen(), media_type="text/event-stream")
        else:
             # Run blocking iteration off the event loop
            out = await asyncio.to_thread(stream_tokens)
            await gen_task
            response = f"{''.join(out)}{lp_links_html}"
            print(f"[gen] full response length={len(response)}, appended_links={bool(lp_links_html)}")
            return response

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    ready = all(x is not None for x in (_embedder, _vstore, _tokenizer, _model))
    return {"status": "ok" if ready else "initializing", "version": APP_VERSION}


@app.get("/version")
async def version():
    return {"version": APP_VERSION, "models": {"embedding": EMB_MODEL, "llm": LLM_MODEL}}


@app.post("/v1/reindex")
async def reindex(data: ReindexRequest, _=Depends(require_api_key)):
    print(f"Reindex request: {data}")
    await ensure_ready()
    global _vstore
    items_md = read_markdown_files(DATA_DIR)
    items_nb = read_notebook_files(NOTEBOOKS_DIR)
    print(f"[reindex] md_items={len(items_md)}, nb_items={len(items_nb)}")
    items = items_md + items_nb
    paths = read_learning_paths(DATA_DIR)
    
    LOG.info("Reindexing %d markdown files and %d learning paths from %s", len(items), len(paths), DATA_DIR)
    
    all_texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    
    print(f"[reindex] chunking {len(items)} items …")
    for it in items:
        chunks = split_markdown(it.get("text", ""), max_tokens=400)
        print(f"[reindex] {it.get('path')} -> {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            is_notebook = str(it.get("path", "")).endswith(".ipynb") or it.get("_source_type") == DWENGO_PYTHON_NOTEBOOK_TYPE
            meta = {
                "text": ch,
                "title": it.get("title"),
                "path": it.get("path"),
                "hruid": it.get("hruid"),
                "language": it.get("language"),
                "description": it.get("description"),
                "type": DWENGO_PYTHON_NOTEBOOK_TYPE if is_notebook else DWENGO_WEBSITE_CONTENT_TYPE,
                "chunk_id": i,
            }
            
            # Attach learning path titles for website content only
            if not is_notebook:
                lp_indexes = []
                if it.get("hruid"):
                    for lp in paths:
                        for node in lp.get("nodes", []):
                            if node.get("learningobject_hruid") == it["hruid"]:
                                lp_indexes.append({
                                    "title": lp.get("title"),
                                    "hruid": lp.get("hruid"),
                                    "language": lp.get("language"),
                                })
                if lp_indexes:
                    meta["learning_paths"] = lp_indexes
            
            metas.append(meta)
            all_texts.append(INSTRUCT_PASSAGE_PREFIX + ch)

    print(f"[reindex] embedding {len(all_texts)} chunks …")
    embs = embed_passages(all_texts)

    store = VectorStore(embs.shape[1])
    store.add(embs, metas)
    store.save(INDEX_PATH, META_PATH)
    print(f"[reindex] saved index={INDEX_PATH}, meta={META_PATH}")
    _vstore = store
    return {"status": "reindexed", "chunks": len(all_texts)}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, _=Depends(require_api_key)):
    LOG.info(f'Received request: {body}')
    print(f'Received request: {body}')
    await ensure_ready()
    print(f"Ready to process chat request")
    hits = retrieve(body.query, body.k)
    print(f"Retrieved {len(hits)} hits")
    prompt = build_prompt(body.query, hits, history=body.history)
    print(f"Built prompt of length {len(prompt)}")
    
    # Add sources 
    lp_links = []
    for i, h in enumerate(hits, 1):
        type = h.get("type")
        if type == DWENGO_WEBSITE_CONTENT_TYPE:
            learning_paths = h.get("learning_paths", [])
            for lp in learning_paths:
                id = lp.get("hruid", "?")
                lp_lang = lp.get("language", "?")
                lp_tt = lp.get("title", "?")
                lp_links.append(f"<a href='https://dwengo.org/learning-path.html?hruid={id}&language={lp_lang}&te=true' target='_blank'>{lp_tt}</a>")
        elif type == DWENGO_PYTHON_NOTEBOOK_TYPE:
            id = h.get("hruid", "?")
            lp_links.append(f"<a href='https://kiks.ilabt.imec.be/hub/tmplogin?id={id}' target='_blank'>{h.get('title','?')}</a>")
    # remove duplicates from lp_links
    lp_links = list(dict.fromkeys(lp_links))
    print(f"[chat] lp_links={len(lp_links)}")
    for l in lp_links:
        print(f"[chat-link]  {l}")
            
    
    if body.stream:
        print("Starting streaming response")
        resp = await generate_text(prompt, body.max_new_tokens, body.temperature, body.top_p, stream=True, source_links=lp_links)
        return resp  # StreamingResponse
    else:
        print("Starting non-streaming response")
        text = await generate_text(prompt, body.max_new_tokens, body.temperature, body.top_p, stream=False, source_links=lp_links)
        return ChatResponse(response=cast(str, text), retrieved=[{k: h.get(k) for k in ("path","title","language","score","chunk_id")} for h in hits])

# -----------------------------
# Lifespan
# -----------------------------
@app.on_event("startup")
async def on_startup():
    global _embedder, _vstore, _tokenizer, _model

    if SYNC_REPO:
        try:
            clone_or_update_repo(REPO_URL, DATA_DIR)
        except Exception as e:
            LOG.error("Repository sync failed (content): %s", e)
        try:
            clone_or_update_repo(NOTEBOOKS_REPO_URL, NOTEBOOKS_DIR)
        except Exception as e:
            LOG.error("Repository sync failed (notebooks): %s", e)

    LOG.info("Loading embedding model: %s", EMB_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_device = "cpu"
    _embedder = SentenceTransformer(EMB_MODEL, device=emb_device)

    if INDEX_PATH.exists() and META_PATH.exists():
        LOG.info("Loading FAISS index from disk")
        _vstore = VectorStore.load(INDEX_PATH, META_PATH)
    else:
        LOG.info("No index found. Building initial index from content…")
        items_md = read_markdown_files(DATA_DIR)
        items_nb = read_notebook_files(NOTEBOOKS_DIR)
        items = items_md + items_nb
        print(f"[startup] md_items={len(items_md)}, nb_items={len(items_nb)}")
        paths = read_learning_paths(DATA_DIR)
        all_texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for it in items:
            chunks = split_markdown(it.get("text", ""), max_tokens=400)
            for i, ch in enumerate(chunks):
                all_texts.append(INSTRUCT_PASSAGE_PREFIX + ch)
                is_notebook = str(it.get("path", "")).endswith(".ipynb") or it.get("_source_type") == DWENGO_PYTHON_NOTEBOOK_TYPE
                meta: Dict[str, Any] = {
                    "text": ch,
                    "title": it.get("title"),
                    "path": it.get("path"),
                    "language": it.get("language"),
                    "hruid": it.get("hruid"),
                    "type": DWENGO_PYTHON_NOTEBOOK_TYPE if is_notebook else DWENGO_WEBSITE_CONTENT_TYPE,
                    "chunk_id": i,
                }
                # Attach learning path info for website content only
                if not is_notebook and it.get("hruid"):
                    lp_indexes = []
                    for lp in paths:
                        for node in lp.get("nodes", []):
                            if node.get("learningobject_hruid") == it["hruid"]:
                                lp_indexes.append({
                                    "title": lp.get("title"),
                                    "hruid": lp.get("hruid"),
                                    "language": lp.get("language"),
                                })
                    if lp_indexes:
                        meta["learning_paths"] = lp_indexes
                metas.append(meta)
        if all_texts:
            embs = embed_passages(all_texts)
            _vstore = VectorStore(embs.shape[1])
            _vstore.add(embs, metas)
            _vstore.save(INDEX_PATH, META_PATH)
        else:
            dim = _embedder.get_sentence_embedding_dimension()
            _vstore = VectorStore(int(dim) if dim is not None else 768)
            _vstore.save(INDEX_PATH, META_PATH)

    LOG.info("Loading LLM: %s", LLM_MODEL)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.bfloat16
    print(f"[startup] loading model {LLM_MODEL} on device {device} with dtype {torch_dtype}")
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _ = _tokenizer("hello", return_tensors="pt").to(_model.device)


@app.on_event("shutdown")
async def on_shutdown():
    LOG.info("Shutting down Dwengo RAG service")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8064, workers=4, proxy_headers=True, forwarded_allow_ips="*")
