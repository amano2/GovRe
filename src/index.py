"""FAISS Vector Index Builder and Retriever using sentence-transformers (all-MiniLM-L6-v2).
Builds per-domain indexes for both original and edited policy sets.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    POLICY_DIRS,
    EDITED_POLICY_DIRS,
    EMBEDDING_MODEL_NAME,
    RESULTS_DIR,
    TOP_K_RETRIEVAL,
)

# Global singleton for embedding model to avoid reloading
_EMBEDDER: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def chunk_markdown_document(filepath: Path, domain: str) -> List[Dict[str, Any]]:
    """
    Parses a markdown document into semantic section chunks based on headers.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = filepath.name
    # Split on headers (# , ## , ### )
    sections = re.split(r'\n(?=#{1,3}\s)', content)
    chunks = []

    for idx, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue
        lines = sec.split("\n")
        header_line = lines[0].strip() if lines else "General"
        header = header_line.lstrip("#").strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else header

        chunk_id = f"{domain}_{filename.replace('.md', '')}_chunk_{idx}"
        chunk_text = f"Policy Document: {filename}\nSection: {header}\nContent:\n{sec}"

        chunks.append({
            "chunk_id": chunk_id,
            "domain": domain,
            "source_file": filename,
            "header": header,
            "text": chunk_text,
            "raw_section": sec,
        })

    return chunks


class DomainIndex:
    """FAISS index manager for a specific policy domain."""

    def __init__(self, domain: str, is_edited: bool = False):
        self.domain = domain
        self.is_edited = is_edited
        self.policy_dir = EDITED_POLICY_DIRS[domain] if is_edited else POLICY_DIRS[domain]
        suffix = "_edited" if is_edited else "_orig"
        self.index_path = RESULTS_DIR / f"faiss_{domain}{suffix}.index"
        self.meta_path = RESULTS_DIR / f"meta_{domain}{suffix}.json"
        self.chunks: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Builds index from policy files or loads existing prebuilt index."""
        if not force_rebuild and self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            return

        # Build index
        embedder = get_embedder()
        self.chunks = []

        if not self.policy_dir.exists():
            raise FileNotFoundError(f"Policy directory not found: {self.policy_dir}")

        for md_file in sorted(self.policy_dir.glob("*.md")):
            file_chunks = chunk_markdown_document(md_file, self.domain)
            self.chunks.extend(file_chunks)

        if not self.chunks:
            raise ValueError(f"No markdown chunks extracted for domain {self.domain} in {self.policy_dir}")

        texts = [c["text"] for c in self.chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]

        # Inner Product with normalized embeddings = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        # Save to disk
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Retrieves top-k most relevant chunks using Hybrid Dense + Lexical Keyword scoring."""
        if self.index is None:
            self.build_or_load()

        embedder = get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # Retrieve candidate pool from dense index
        candidate_k = min(len(self.chunks), max(top_k * 3, 10))
        scores, indices = self.index.search(q_emb.astype("float32"), candidate_k)

        # Tokenize query for lexical scoring
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "with", "by", "from"}
        q_tokens = [w.lower() for w in re.findall(r'[a-zA-Z0-9_\$]+', query) if len(w) >= 2 and w.lower() not in stop_words]
        q_tokens_set = set(q_tokens)

        candidates = []
        for dense_score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                text_lower = chunk["text"].lower()
                header_lower = chunk.get("header", "").lower()

                # Lexical overlap score
                if q_tokens_set:
                    body_matches = sum(1 for t in q_tokens_set if t in text_lower)
                    header_matches = sum(1 for t in q_tokens_set if t in header_lower)
                    lexical_score = (body_matches + 1.5 * header_matches) / max(len(q_tokens_set), 1)
                    lexical_score = min(lexical_score, 1.0)
                else:
                    lexical_score = 0.0

                # Hybrid fusion: 65% Dense Cosine Similarity + 35% Lexical Keyword Match
                # Dense cosine similarity with normalized embeddings is in [-1, 1], typically [0.2, 0.8]
                norm_dense = max(0.0, float(dense_score))
                hybrid_score = 0.65 * norm_dense + 0.35 * lexical_score

                chunk["dense_score"] = float(dense_score)
                chunk["lexical_score"] = round(lexical_score, 4)
                chunk["similarity_score"] = round(hybrid_score, 4)
                candidates.append(chunk)

        # Sort by combined hybrid score descending
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


def build_all_indexes(force_rebuild: bool = False, build_edited: bool = True) -> None:
    """Builds FAISS indexes for all domains (original and edited)."""
    for dom in ["hr", "finance", "it_access"]:
        idx_orig = DomainIndex(dom, is_edited=False)
        idx_orig.build_or_load(force_rebuild=force_rebuild)
        print(f"Built original index for [{dom}] with {len(idx_orig.chunks)} chunks.")

        if build_edited and EDITED_POLICY_DIRS[dom].exists() and list(EDITED_POLICY_DIRS[dom].glob("*.md")):
            idx_edit = DomainIndex(dom, is_edited=True)
            idx_edit.build_or_load(force_rebuild=force_rebuild)
            print(f"Built edited index for [{dom}] with {len(idx_edit.chunks)} chunks.")


if __name__ == "__main__":
    build_all_indexes(force_rebuild=True, build_edited=False)
