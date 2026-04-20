"""
Retrieval Strategies -- Week 4 (RAG)

Beyond top-k cosine: techniques that consistently beat naive dense retrieval
in 2026 production systems.

  * BM25Retriever         - Sparse keyword retrieval (rank-bm25)
  * HybridRetriever       - BM25 + dense, fused with Reciprocal Rank Fusion
  * mmr_search            - Maximal Marginal Relevance (diversity)
  * hyde_retrieve         - Hypothetical Document Embeddings (HyDE)
  * rewrite_query         - LLM-driven query expansion / rewriting
  * reciprocal_rank_fusion- Standalone RRF for arbitrary ranked lists
"""

import re
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 (Best Match 25) sparse retriever via rank_bm25.
    Operates on the same chunk dicts as our vector stores.
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install: pip install rank-bm25")
        self.chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  ✓ BM25 indexed {len(chunks)} chunks")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        import numpy as np
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {**self.chunks[i], "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(top_idx)
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Standard RRF: score(d) = sum over lists of 1 / (k + rank(d)).
    Robust to score scale differences (only ranks matter).

    Args:
        ranked_lists:  Each inner list is a ranked list of chunk dicts.
                       Chunks are de-duped by `(source, chunk_id)` if available,
                       else by `text`.
        k:             RRF smoothing constant (LangChain default = 60)
        top_k:         How many fused results to return
    """
    scores: Dict[str, float] = {}
    objects: Dict[str, Dict[str, Any]] = {}

    def chunk_key(chunk):
        meta = chunk.get("metadata", {})
        if "source" in meta and "chunk_id" in meta:
            return f"{meta['source']}::{meta['chunk_id']}"
        return chunk["text"][:200]

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            objects[key] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for rank, (key, score) in enumerate(fused, start=1):
        c = dict(objects[key])
        c["score"] = score
        c["rank"] = rank
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combine a dense vector store (any of our *Store classes) with BM25,
    fuse with Reciprocal Rank Fusion. The standard 2026 production setup.
    """

    def __init__(self, vector_store, embedding_model, bm25: BM25Retriever):
        self.vs = vector_store
        self.em = embedding_model
        self.bm25 = bm25

    def search(self, query: str, k: int = 5, dense_k: Optional[int] = None,
               sparse_k: Optional[int] = None, rrf_k: int = 60):
        dense_k = dense_k or k * 4
        sparse_k = sparse_k or k * 4
        q_vec = self.em.encode_query(query) if hasattr(self.em, "encode_query") else self.em.encode(query)
        dense_results = self.vs.search(q_vec, k=dense_k)
        sparse_results = self.bm25.search(query, k=sparse_k)
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results], k=rrf_k, top_k=k,
        )
        print(f"  Hybrid: dense={len(dense_results)} sparse={len(sparse_results)} -> fused={len(fused)}")
        return fused


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance (diversity)
# ---------------------------------------------------------------------------

def mmr_search(
    vector_store,
    embedding_model,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    MMR (Carbonell & Goldstein, 1998).

    score = lambda * sim(q, doc) - (1-lambda) * max sim(doc, selected)

    Use lambda=1.0 for pure relevance, lambda=0.0 for pure diversity.
    """
    import numpy as np
    q_vec = embedding_model.encode_query(query) if hasattr(embedding_model, "encode_query") else embedding_model.encode(query)
    candidates = vector_store.search(q_vec, k=fetch_k)
    if not candidates:
        return []

    cand_vecs = embedding_model.encode([c["text"] for c in candidates])
    cand_vecs = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    sim_q = cand_vecs @ q_norm

    selected, remaining = [], list(range(len(candidates)))
    while len(selected) < min(k, len(candidates)) and remaining:
        if not selected:
            best = max(remaining, key=lambda i: sim_q[i])
        else:
            sel_vecs = cand_vecs[selected]
            best, best_score = None, -1e9
            for i in remaining:
                sim_max = max(float(cand_vecs[i] @ sv) for sv in sel_vecs)
                score = lambda_mult * sim_q[i] - (1 - lambda_mult) * sim_max
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)

    return [{**candidates[i], "rank": rank + 1} for rank, i in enumerate(selected)]


# ---------------------------------------------------------------------------
# HyDE: Hypothetical Document Embeddings
# ---------------------------------------------------------------------------

HYDE_PROMPT = """Please write a detailed, factual passage that would directly
answer the question below. Write as if you were the source document, not
as a conversational AI. Do NOT say "I don't know" — write a plausible answer
in 3-5 sentences.

Question: {question}

Passage:"""


def hyde_retrieve(
    vector_store,
    embedding_model,
    llm_client,
    query: str,
    k: int = 5,
    num_hypotheses: int = 1,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    HyDE retrieval: ask an LLM to generate a hypothetical answer, embed THAT,
    and search. Bridges the question/answer asymmetry in embedding space.

    With num_hypotheses > 1, embeds each and averages.
    """
    import numpy as np
    hypos = []
    for _ in range(num_hypotheses):
        resp = llm_client.generate(
            prompt=HYDE_PROMPT.format(question=query),
            model=model,
            max_tokens=300,
            temperature=0.7,
        )
        if "error" not in resp:
            hypos.append(resp["content"].strip())
    if not hypos:
        # fallback to plain query
        return vector_store.search(embedding_model.encode_query(query) if hasattr(embedding_model, "encode_query") else embedding_model.encode(query), k=k)

    print(f"  HyDE generated {len(hypos)} hypothetical answer(s)")
    vecs = embedding_model.encode(hypos)
    avg = np.mean(vecs, axis=0)
    return vector_store.search(avg, k=k)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """Rewrite the user's question to maximize retrieval
recall against a vector database of documents. Output {n} alternative
phrasings, one per line, no numbering, no quotes. Each should be self-
contained and use different keywords.

Question: {question}"""


def rewrite_query(llm_client, query: str, n: int = 3,
                  model: Optional[str] = None) -> List[str]:
    """LLM-driven multi-query rewriting."""
    resp = llm_client.generate(
        prompt=QUERY_REWRITE_PROMPT.format(question=query, n=n),
        model=model,
        max_tokens=400,
        temperature=0.4,
    )
    if "error" in resp:
        return [query]
    rewrites = [ln.strip("- *") for ln in resp["content"].splitlines() if ln.strip()]
    rewrites = [r for r in rewrites if r and len(r) > 5][:n]
    return [query] + rewrites
