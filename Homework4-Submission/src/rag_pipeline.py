"""
End-to-End RAG Pipeline -- Week 4

Glue layer that wires everything together: load → chunk → embed → index →
retrieve → (rerank) → generate. Used in Notebook 08 for the
Resume/Portfolio agent capstone.
"""

from typing import Dict, List, Optional, Any


GROUNDED_QA_PROMPT = """You are an assistant that answers questions ONLY
using the retrieved context below. If the answer is not in the context,
say "I don't know — that information isn't in my sources."

Always cite the source(s) you used in this format: [source_name].

Retrieved context:
---
{context}
---

Question: {question}

Answer (with citations):"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for the prompt with [source] tags."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("metadata", {}).get("source", f"chunk_{i}")
        parts.append(f"[{src}] {c['text']}")
    return "\n\n".join(parts)


class RAGPipeline:
    """
    Production-style RAG pipeline.

    Components:
      loader        - callable(path) -> document dict
      chunker       - callable(text) -> List[str]
      embedding     - EmbeddingModel
      vector_store  - any *Store
      retriever     - object with .search(query, k) (defaults to vector_store)
      reranker      - optional, has .rerank(query, candidates, top_k)
      llm_client    - LLMClient for final generation
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_client,
        chunker=None,
        retriever=None,
        reranker=None,
        retrieve_k: int = 8,
        rerank_k: int = 4,
    ):
        self.em = embedding_model
        self.vs = vector_store
        self.llm = llm_client
        self.chunker = chunker
        self.retriever = retriever or vector_store
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

    # ---- Indexing -------------------------------------------------------

    def index_documents(self, docs: List[Dict[str, Any]]):
        """Chunk each doc, embed, add to vector store."""
        all_chunks = []
        for doc in docs:
            text = doc["text"]
            chunks = self.chunker(text) if self.chunker else [text]
            source = doc.get("source", "unknown")
            for i, c in enumerate(chunks):
                all_chunks.append({
                    "text": c,
                    "metadata": {"source": source, "chunk_id": i},
                })
        print(f"\n  Embedding {len(all_chunks)} chunks...")
        vecs = self.em.encode([c["text"] for c in all_chunks], batch_size=32)
        self.vs.add(all_chunks, vecs)
        return all_chunks

    # ---- Retrieval ------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "encode_query") or self.retriever is self.vs:
            q_vec = self.em.encode_query(query) if hasattr(self.em, "encode_query") else self.em.encode(query)
            candidates = self.vs.search(q_vec, k=self.retrieve_k)
        else:
            # HybridRetriever takes the raw string
            candidates = self.retriever.search(query, k=self.retrieve_k)
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k=self.rerank_k)
        return candidates

    # ---- Generation -----------------------------------------------------

    def answer(self, query: str, model: Optional[str] = None,
               max_tokens: int = 600) -> Dict[str, Any]:
        chunks = self.retrieve(query)
        context = format_context(chunks)
        prompt = GROUNDED_QA_PROMPT.format(context=context, question=query)
        resp = self.llm.generate(
            prompt=prompt, model=model,
            max_tokens=max_tokens, temperature=0.2,
        )
        return {
            "question": query,
            "answer": resp.get("content", "") if "error" not in resp else f"[ERROR] {resp['error']}",
            "contexts": [c["text"] for c in chunks],
            "sources": [c.get("metadata", {}).get("source") for c in chunks],
            "raw_response": resp,
        }
