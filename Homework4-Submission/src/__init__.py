"""
Week 4: Retrieval-Augmented Generation (RAG) - Shared Modules

This package contains reusable code for all Week 4 notebooks.

Modules with heavy dependencies (sentence-transformers, faiss, chromadb,
langchain, etc.) are imported on-demand in each notebook rather than here.
"""

__version__ = "4.0.0"

# Core modules (lightweight deps only)
from .llm_client import LLMClient
from .cost_tracker import CostTracker
from .utils import (
    estimate_tokens,
    estimate_cost,
    format_response,
    save_task_output,
    append_to_reflection,
)

__all__ = [
    'LLMClient',
    'CostTracker',
    'estimate_tokens',
    'estimate_cost',
    'format_response',
    'save_task_output',
    'append_to_reflection',
]

# Week 4 modules are imported directly in notebooks:
#   from src.document_loader import load_pdf, fetch_arxiv_papers
#   from src.chunking import recursive_chunk, semantic_chunk, contextual_chunk
#   from src.embeddings import EmbeddingModel, compare_models
#   from src.vector_store import FAISSStore, ChromaStore
#   from src.retrieval import HybridRetriever, hyde_retrieve, mmr_search
#   from src.reranker import CrossEncoderReranker, FlashRankReranker
#   from src.rag_evaluation import faithfulness, context_recall, context_precision
