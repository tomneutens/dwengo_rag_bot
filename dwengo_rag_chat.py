#!/usr/bin/env python3
"""
Dwengo Learning Content — Local RAG Chat (Dutch/English)
=======================================================

What this script does
---------------------
1) Clones/updates the Dwengo learning content repository.
2) Parses learning *objects* (Markdown + YAML front matter) and learning *paths* (JSON).
3) Chunks Markdown intelligently and builds a FAISS vector index using multilingual embeddings.
4) Runs a local, streaming chat loop that does retrieval-augmented generation (RAG).
5) Uses an efficient open-source LLM that runs in real time on an NVIDIA A100 GPU.

Default models
--------------
- Retriever embeddings: "intfloat/multilingual-e5-large-instruct" (fast, strong multilingual, incl. Dutch)
- Generator (LLM):      "Qwen/Qwen2.5-7B-Instruct" (great quality/speed; multilingual)

You can switch the LLM to another HF model (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct") if you have access.

Quick start
-----------
python3 dwengo_rag_chat.py \
  --repo https://github.com/dwengovzw/learning_content \
  --data_dir ./learning_content \
  --index_path ./dwengo_faiss \
  --k 6

Type your question and press Enter. Use :reset to clear history, :rebuild to reindex, and :quit to exit.

Dependencies (install once)
---------------------------
pip install --upgrade gitpython pyyaml python-frontmatter sentence-transformers faiss-cpu \
  transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121 \
  tiktoken rich

If you have FAISS-GPU available, install faiss-gpu instead of faiss-cpu.

Notes
-----
- The script auto-detects CUDA and places models on GPU if available.
- On an A100, Qwen2.5-7B-Instruct streams tokens comfortably in real time for chat.
- All retrieval and generation happens locally. No internet calls at runtime (except the first time models are downloaded by HF).
"""

import argparse
import os
import sys
import json
import re
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import frontmatter
import yaml
from git import Repo

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

console = Console()

# -----------------------------
# Config & helpers
# -----------------------------

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"  # Dutch-friendly
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # real-time capable on A100

MD_EXTS = {".md", ".markdown"}
JSON_EXTS = {".json"}

SYSTEM_PROMPT = (
    "Je bent een helpende onderwijsassistent. Beantwoord in het Nederlands wanneer de vraag in het Nederlands is. "
    "Gebruik de contextfragmenten uit het Dwengo-leermateriaal om nauwkeurige, brongebonden antwoorden te geven. "
    "Neem geen links op in je antwoorden. Verwijs niet naar bronnen."
    "Als je het antwoord niet met zekerheid weet vanuit de context, zeg dan eerlijk dat je het niet zeker weet."
)

INSTRUCT_EMBED_PREFIX = "Instruct: Represent the query for retrieving supporting passages: "
INSTRUCT_PASSAGE_PREFIX = "Passage: "

# -----------------------------
# Git operations
# -----------------------------

def clone_or_update_repo(repo_url: str, dest_dir: Path) -> None:
    if dest_dir.exists() and (dest_dir / ".git").exists():
        console.print(f"[bold green]Updating repo[/bold green] at {dest_dir} …")
        repo = Repo(str(dest_dir))
        origin = repo.remotes.origin
        origin.fetch()
        repo.git.reset("--hard", "origin/HEAD")
    else:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        console.print(f"[bold green]Cloning repo[/bold green] into {dest_dir} …")
        Repo.clone_from(repo_url, str(dest_dir))

# -----------------------------
# Parsing learning objects & paths
# -----------------------------

def read_markdown_files(root: Path) -> List[Dict[str, Any]]:
    items = []
    for p in root.rglob("*"):
        if p.suffix.lower() in MD_EXTS and p.is_file():
            try:
                post = frontmatter.load(p)
                metadata = post.metadata or {}
                text = str(post.content)
                items.append({
                    "path": str(p),
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
                console.print(f"[yellow]Warning:[/yellow] Could not parse {p}: {e}")
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

# -----------------------------
# Chunking (Markdown-aware but framework-light)
# -----------------------------

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

# -----------------------------
# Vector store (FAISS)
# -----------------------------

def build_or_load_faiss(index_path: Path, dim: int) -> faiss.IndexFlatIP:
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        return index
    index = faiss.IndexFlatIP(dim)
    return index

# -----------------------------
# RAG pipeline
# -----------------------------

class RAGPipeline:
    def __init__(
        self,
        data_dir: Path,
        index_path: Path,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        k: int = 6,
        rebuild: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.index_path = index_path
        self.k = k

        # Embeddings
        console.print("[bold]Loading embedding model[/bold] …")
        self.embedder = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        # Docs storage parallel to FAISS
        self.doc_store: List[Dict[str, Any]] = []

        # FAISS index
        self.index = build_or_load_faiss(index_path, self.embed_dim)

        # LLM
        console.print("[bold]Loading LLM[/bold] …")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # If index does not exist or rebuild requested, (re)index now.
        if rebuild or not index_path.exists():
            self.reindex()
        else:
            # Load doc_store metadata
            meta_path = index_path.with_suffix(".meta.json")
            if meta_path.exists():
                self.doc_store = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                console.print("[yellow]No metadata found for FAISS index; rebuilding…[/yellow]")
                self.reindex()

    def reindex(self) -> None:
        console.print("[bold green]Indexing learning content…[/bold green]")
        md_items = read_markdown_files(self.data_dir)
        paths = read_learning_paths(self.data_dir)
        path_by_hruid = {p.get("hruid"): p for p in paths}

        all_chunks = []
        chunk_meta = []

        for item in md_items:
            text = item["raw"] or ""
            chunks = split_markdown(text, max_tokens=400)
            for i, ch in enumerate(chunks):
                meta = {
                    "source_path": item.get("path"),
                    "hruid": item.get("hruid"),
                    "title": item.get("title"),
                    "language": item.get("language"),
                    "description": item.get("description"),
                    "chunk_id": i,
                }
                # Attach learning path titles if this LO appears in any path
                lp_indexes = []
                if item.get("hruid"):
                    for lp in paths:
                        for node in lp.get("nodes", []):
                            if node.get("learningobject_hruid") == item["hruid"]:
                                lp_indexes.append({
                                    "title": lp.get("title"),
                                    "hruid": lp.get("hruid"),
                                    "language": lp.get("language"),
                                })
                    if lp_indexes:
                        meta["learning_paths"] = lp_indexes

                all_chunks.append(ch)
                chunk_meta.append(meta)

        # Encode passages
        console.print(f"Encoding {len(all_chunks)} chunks for retrieval…")
        # E5 expects "Passage: " prefix for documents
        passages = [INSTRUCT_PASSAGE_PREFIX + c for c in all_chunks]
        embs = self.embedder.encode(passages, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embs.astype(np.float32))

        # Persist
        faiss.write_index(self.index, str(self.index_path))
        meta_path = self.index_path.with_suffix(".meta.json")
        meta_payload = []
        for ch, m in zip(all_chunks, chunk_meta):
            meta_payload.append({"text": ch, **m})
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.doc_store = meta_payload
        console.print("[bold green]Index built and saved.[/bold green]")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.k
        # E5 instruct uses this query prefix
        q = INSTRUCT_EMBED_PREFIX + query
        q_emb = self.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype(np.float32), top_k)
        hits = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            rec = dict(self.doc_store[idx])
            rec["score"] = float(score)
            hits.append(rec)
        return hits

    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        ctx_blocks = []
        for c in contexts:
            header = f"Bron: {Path(c.get('source_path', ''))}\nTitel: {c.get('title')}\nTaal: {c.get('language')}\n".strip()
            text = c.get("text", "").strip()
            ctx_blocks.append(f"[{header}]\n{text}")
        joined_ctx = "\n\n---\n\n".join(ctx_blocks)

        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n"
            f"<|user|>\nVraag: {query}\n\nHier zijn relevante fragmenten uit het leermateriaal. Gebruik deze als betrouwbare bron.\n\n{joined_ctx}\n\nGeef een beknopt en duidelijk antwoord met concrete verwijzingen naar de bron(nen) (pad + titel).\n</|user|>\n"
        )
        return prompt

    def stream_generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()

# -----------------------------
# CLI chat
# -----------------------------

def run_cli(args):
    data_dir = Path(args.data_dir)

    # Step 1: clone/update repo
    if args.sync_repo:
        clone_or_update_repo(args.repo, data_dir)

    # Step 2: build pipeline
    rag = RAGPipeline(
        data_dir=data_dir,
        index_path=Path(args.index_path),
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        k=args.k,
        rebuild=args.rebuild,
    )

    history: List[Tuple[str, str]] = []  # (user, assistant)

    console.print(Panel.fit("Dwengo RAG Chat — type je vraag. (:reset, :rebuild, :quit)", title="Welkom"))

    while True:
        try:
            user_in = console.input("[bold cyan]Jij[/bold cyan]: ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nTot later!")
            break

        if not user_in.strip():
            continue
        if user_in.strip() == ":quit":
            break
        if user_in.strip() == ":reset":
            history.clear()
            console.print("[green]Gesprek gewist.[/green]\n")
            continue
        if user_in.strip() == ":rebuild":
            rag.reindex()
            console.print("[green]Index herbouwd.[/green]")
            continue

        # Step 3: retrieve
        hits = rag.retrieve(user_in, top_k=args.k)

        # Step 4: prompt with context + chat history (compact)
        hist_txt = "\n".join([f"<|user|> {u}\n<|assistant|> {a}" for u, a in history[-4:]])
        prompt = rag.build_prompt(user_in, hits)
        if hist_txt:
            prompt = hist_txt + "\n" + prompt

        console.print("[bold magenta]Assistent[/bold magenta]:", end=" ")
        with Live(auto_refresh=False) as live:
            buf = []
            for token in rag.stream_generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p):
                buf.append(token)
                if token.endswith("\n") or len(buf) % 8 == 0:
                    live.update(Markdown("".join(buf)))
                    live.refresh()
            live.update(Markdown("".join(buf)))
            live.refresh()
        answer = "".join(buf)

        # Show top sources inline (paths + titles)
        console.print("\n[dim]Bronnen:[/dim]")
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
        for i, link in enumerate(lp_links, 1):
            console.print(f"[dim]{i}. {link}[/dim]")

        history.append((user_in, answer))


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Dwengo Learning Content — Local RAG Chat")
    ap.add_argument("--repo", type=str, default="https://github.com/dwengovzw/learning_content", help="Git repo URL for content")
    ap.add_argument("--data_dir", type=str, default="./learning_content", help="Local path for the repo")
    ap.add_argument("--index_path", type=str, default="./dwengo_faiss", help="Path to store FAISS index (file)")

    ap.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="HF embedding model")
    ap.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help="HF causal LM model")

    ap.add_argument("--k", type=int, default=6, help="Top-k passages to retrieve")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--sync_repo", action="store_true", help="Clone or update the repo before indexing")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild of the FAISS index")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(args)
