"""
Rerankers -- Week 4 (RAG)

After first-stage retrieval (e.g. dense top-50), a reranker re-scores the
candidates with a heavier model that conditions on the query AND each
document jointly (cross-encoder). +33-40% accuracy for ~120ms latency.

Backends (best 2026 picks):
  * CrossEncoderReranker   - sentence-transformers/cross-encoder
                             ('BAAI/bge-reranker-v2-m3' is current SOTA OSS)
  * FlashRankReranker      - Lightweight, fast (ms/doc) — best starter
  * CohereReranker         - Cohere Rerank 4.0 (API, top quality)

All rerankers expose:
    rerank(query, candidates, top_k) -> reranked list with new scores
"""

from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Cross-encoder (sentence-transformers)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Recommended models (2026):
      - 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (fast, English)
      - 'BAAI/bge-reranker-v2-m3'               (SOTA OSS, multilingual)
      - 'BAAI/bge-reranker-base'                (balanced)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")
        print(f"  Loading reranker {model_name}...")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        print("  ✓ Reranker ready")

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = []
        for rank, (c, s) in enumerate(ranked[:top_k], start=1):
            new = dict(c)
            new["score"] = float(s)
            new["rerank_score"] = float(s)
            new["rank"] = rank
            out.append(new)
        return out


# ---------------------------------------------------------------------------
# FlashRank (lightweight, the fastest path)
# ---------------------------------------------------------------------------

class FlashRankReranker:
    """
    FlashRank — a tiny C++ inference engine wrapping ONNX cross-encoders.
    Lowest-friction starter reranker; runs on CPU without torch.

    Models: 'ms-marco-TinyBERT-L-2-v2' (default), 'ms-marco-MiniLM-L-12-v2',
            'rank-T5-flan' (largest, best).
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2",
                 cache_dir: Optional[str] = None):
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError("Install: pip install flashrank")
        kwargs = {"model_name": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.ranker = Ranker(**kwargs)
        self.model_name = model_name
        print(f"  ✓ FlashRank ready ({model_name})")

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        from flashrank import RerankRequest
        if not candidates:
            return []
        passages = [{"id": i, "text": c["text"], "meta": c.get("metadata", {})}
                    for i, c in enumerate(candidates)]
        req = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(req)
        out = []
        for rank, item in enumerate(ranked[:top_k], start=1):
            orig = candidates[item["id"]]
            new = dict(orig)
            new["score"] = float(item["score"])
            new["rerank_score"] = float(item["score"])
            new["rank"] = rank
            out.append(new)
        return out


# ---------------------------------------------------------------------------
# Cohere Rerank (API)
# ---------------------------------------------------------------------------

class CohereReranker:
    """
    Cohere Rerank 4.0 (API). Top accuracy, $$ per call.
    Set COHERE_API_KEY in .env.
    """

    def __init__(self, model: str = "rerank-v3.5"):
        try:
            import cohere
        except ImportError:
            raise ImportError("Install: pip install cohere")
        self.client = cohere.ClientV2()
        self.model = model
        print(f"  ✓ Cohere reranker ready ({model})")

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        r = self.client.rerank(
            model=self.model,
            query=query,
            documents=[c["text"] for c in candidates],
            top_n=top_k,
        )
        out = []
        for rank, result in enumerate(r.results, start=1):
            orig = candidates[result.index]
            new = dict(orig)
            new["score"] = float(result.relevance_score)
            new["rerank_score"] = float(result.relevance_score)
            new["rank"] = rank
            out.append(new)
        return out
