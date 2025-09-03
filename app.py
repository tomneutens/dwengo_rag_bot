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
from typing import Any, Dict, List, Optional, Tuple

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
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./dwengo_faiss.index"))
META_PATH = Path(os.getenv("META_PATH", "./dwengo_faiss.meta.json"))
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large-instruct")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SYNC_REPO = os.getenv("SYNC_REPO", "false").lower() == "true"
API_KEY = os.getenv("API_KEY")  # optional
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

SYSTEM_PROMPT = (
    "Je bent een helpende onderwijsassistent. Beantwoord in het Nederlands wanneer de vraag in het Nederlands is. "
    "Gebruik de contextfragmenten uit het Dwengo-leermateriaal om nauwkeurige, brongebonden antwoorden te geven. "
    "Neem geen links op in je antwoorden. Verwijs niet naar bronnen."
    "Als je het antwoord niet met zekerheid weet vanuit de context, zeg dan eerlijk dat je het niet zeker weet."
)
INSTRUCT_EMBED_PREFIX = "Instruct: Represent the query for retrieving supporting passages: "
INSTRUCT_PASSAGE_PREFIX = "Passage: "

MD_EXTS = {".md", ".markdown"}

API_KEY_HEADER_KEY = "DWENGO-API-KEY"
DWENGO_WEBSITE_CONTENT_TYPE = "DWENGO_WEBSITE_CONTENT"
DWENGO_PYTHON_NOTEBOOK_TYPE = "DWENGO_PYTHON_NOTEBOOK"

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
    if dest_dir.exists() and (dest_dir / ".git").exists():
        LOG.info("Updating repo at %s", dest_dir)
        repo = Repo(str(dest_dir))
        repo.remotes.origin.fetch()
        repo.git.reset("--hard", "origin/HEAD")
    else:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        LOG.info("Cloning repo to %s", dest_dir)
        Repo.clone_from(repo_url, str(dest_dir))


def read_markdown_files(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in MD_EXTS and p.is_file():
            try:
                post = frontmatter.load(p)
                metadata = post.metadata or {}
                text = str(post.content)
                items.append({
                    "path": str(p),
                    "text": text,
                    "hruid": metadata.get("hruid"),
                    "language": metadata.get("language"),
                    "title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "keywords": metadata.get("keywords"),
                    "estimated_time": metadata.get("estimated_time"),
                    "target_ages": metadata.get("target_ages"),
                    "teacher_exclusive": metadata.get("teacher_exclusive"),
                    "raw": text,
                })
            except Exception as e:
                LOG.warning("Could not parse %s: %s", p, e)
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
    for part in parts:
        words = part.split()
        if not words:
            continue
        cur = []
        count = 0
        for w in words:
            cur.append(w)
            count += 1
            if count >= max_tokens:
                chunks.append(" ".join(cur))
                cur = []
                count = 0
        if cur:
            chunks.append(" ".join(cur))

    # Fallback if no headers
    if not chunks:
        words = text.split()
        cur = []
        for w in words:
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
        self.index.add(embs.astype(np.float32))
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    embs = _embedder.encode(passages, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embs.astype(np.float32)


def retrieve(query: str, k: int) -> List[Dict[str, Any]]:
    assert _embedder is not None and _vstore is not None
    q = INSTRUCT_EMBED_PREFIX + query
    q_emb = _embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = _vstore.index.search(q_emb, k)
    hits: List[Dict[str, Any]] = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc = dict(_vstore.docs[idx])
        doc["score"] = float(score)
        hits.append(doc)
    return hits


def build_prompt(query: str, contexts: List[Dict[str, Any]], history: Optional[List[Tuple[str, str]]] = None) -> str:
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
        f"<|user|>\nVraag: {query}\n\nHier zijn relevante fragmenten. Gebruik ze als betrouwbare bron.\n\n{joined_ctx}\n\nGeef een beknopt antwoord en citeer de bron(nen) met pad + titel.\n</|user|>\n"
    return prompt


async def generate_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float, stream: bool, source_links: List[str]):
    assert _model is not None and _tokenizer is not None
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
        thread = asyncio.to_thread(_model.generate, **gen_kwargs)

        def stream_tokens():
            for token in streamer:
                yield token

        gen_task = loop.create_task(thread)
        lp_links_html = f"<br>{"<br>".join(source_links)}"

        if stream:
            async def agen():
                for tok in stream_tokens():
                    yield f"data: {tok}\n\n"
                await gen_task
                if source_links:
                    yield f"data: {lp_links_html}\n\n"
            return StreamingResponse(agen(), media_type="text/event-stream")
        else:
            # Run blocking iteration off the event loop
            out = await asyncio.to_thread(stream_tokens)
            await gen_task
            response = f"{''.join(out)}{lp_links_html}"
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
    await ensure_ready()
    global _vstore
    items = read_markdown_files(DATA_DIR)
    paths = read_learning_paths(DATA_DIR)
    
    LOG.info("Reindexing %d markdown files and %d learning paths from %s", len(items), len(paths), DATA_DIR)
    
    all_texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    
    for it in items:
        chunks = split_markdown(it.get("text", ""), max_tokens=400)
        for i, ch in enumerate(chunks):
            meta = {
                "title": it.get("title"),
                "path": it.get("path"),
                "hruid": it.get("hruid"),
                "language": it.get("language"),
                "description": it.get("description"),
                "type": DWENGO_WEBSITE_CONTENT_TYPE,
                "chunk_id": i,
            }
            
            # Attach learning path titles if this LO appears in any path
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

    embs = embed_passages(all_texts)

    store = VectorStore(embs.shape[1])
    store.add(embs, metas)
    store.save(INDEX_PATH, META_PATH)
    _vstore = store
    return {"status": "reindexed", "chunks": len(all_texts)}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, _=Depends(require_api_key)):
    await ensure_ready()
    hits = retrieve(body.query, body.k)
    prompt = build_prompt(body.query, hits, history=body.history)
    
    # Add sources 
    lp_links = []
    for i, h in enumerate(hits, 1):
        learning_paths = h.get("learning_paths", [])
        for lp in learning_paths:
            lp_hruid = lp.get("hruid", "?")
            lp_lang = lp.get("language", "?")
            lp_tt = lp.get("title", "?")
            lp_links.append(f"<a href='https://dwengo.org/learning-path.html?hruid={lp_hruid}&language={lp_lang}&te=true'>{lp_tt}</a>")
    # remove duplicates from lp_links
    lp_links = list(dict.fromkeys(lp_links))
            
    
    if body.stream:
        resp = await generate_text(prompt, body.max_new_tokens, body.temperature, body.top_p, stream=True, source_links=lp_links)
        return resp  # StreamingResponse
    else:
        text = await generate_text(prompt, body.max_new_tokens, body.temperature, body.top_p, stream=False, source_links=lp_links)
        return ChatResponse(response=text, retrieved=[{k: h.get(k) for k in ("path","title","language","score","chunk_id")} for h in hits])

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
            LOG.error("Repository sync failed: %s", e)

    LOG.info("Loading embedding model: %s", EMB_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _embedder = SentenceTransformer(EMB_MODEL, device=device)

    if INDEX_PATH.exists() and META_PATH.exists():
        LOG.info("Loading FAISS index from disk")
        _vstore = VectorStore.load(INDEX_PATH, META_PATH)
    else:
        LOG.info("No index found. Building initial index from content…")
        items = read_markdown_files(DATA_DIR)
        all_texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for it in items:
            chunks = split_markdown(it.get("text", ""), max_words=400)
            for i, ch in enumerate(chunks):
                all_texts.append(INSTRUCT_PASSAGE_PREFIX + ch)
                metas.append({
                    "text": ch,
                    "title": it.get("title"),
                    "path": it.get("path"),
                    "language": it.get("language"),
                    "chunk_id": i,
                })
        if all_texts:
            embs = embed_passages(all_texts)
            _vstore = VectorStore(embs.shape[1])
            _vstore.add(embs, metas)
            _vstore.save(INDEX_PATH, META_PATH)
        else:
            dim = _embedder.get_sentence_embedding_dimension()
            _vstore = VectorStore(dim)
            _vstore.save(INDEX_PATH, META_PATH)

    LOG.info("Loading LLM: %s", LLM_MODEL)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
