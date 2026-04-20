# Week 4: Retrieval-Augmented Generation (RAG)

**Course:** Machine Learning Engineer in the Generative AI Era  
**Week:** 4 of 10  
**Topics:** Document loading, chunking, embeddings, vector stores, hybrid retrieval, reranking, RAG evaluation  
**Capstone:** Build a RAG-enabled "Resume AI Assistant" (the in-class project)

---

## Overview

This week, you'll graduate from "asking an LLM and hoping" to **grounded** question answering over your own documents. By the end you will have built a complete production-style RAG pipeline that can answer questions about your resume + portfolio (or any corpus you choose) with **citations** and an **automated faithfulness score** — no hallucinations.

You'll touch the entire stack: PDF extraction (PyMuPDF), four chunking strategies including 2024's **contextual retrieval** (Anthropic), three embedding models from the MTEB leaderboard, three vector stores (FAISS, Chroma, Qdrant), advanced retrieval (**hybrid search + RRF**, **HyDE**, **MMR**, multi-query rewriting), cross-encoder + FlashRank reranking, and four **RAGAS-style** evaluation metrics. The final notebook wires everything into a working assistant and (bonus) a FastAPI service.

---

## Learning Objectives

1. Extract clean text from PDFs, web pages, and the arXiv API
2. Compare chunking strategies (recursive, semantic, contextual) and feel why chunk size dominates RAG quality
3. Pick the right embedding model for YOUR corpus by Recall@k — not by the leaderboard
4. Build, persist, and query indexes in FAISS, Chroma, and Qdrant
5. Implement and benchmark **hybrid search**, **HyDE**, **MMR**, and **multi-query** retrieval
6. Add a **reranker** (cross-encoder / FlashRank) — the highest-leverage 100ms in RAG
7. Evaluate end-to-end with reference-free **RAGAS-style** metrics (faithfulness, context precision, answer relevancy)
8. Build a working **Resume RAG Assistant** that refuses out-of-corpus questions

---

## Setup Options

### Path A: Claude API (Cloud) — Recommended

- **Default model:** `claude-sonnet-4-6` (with `claude-haiku-4-5-20251001` for cheap eval calls)
- **Cost:** ~$0.50–$2.00 for the entire week
- **Requires:** `ANTHROPIC_API_KEY` in `.env`

### Path B: Ollama (Local / Free)

- **Default model:** `qwen3.5:27b` (or `llama3.1:8b` if you have less RAM)
- **Cost:** $0
- **Requires:** ~20GB RAM, `ollama pull qwen3.5:27b`

### Path C: Hybrid

- Heavy generation + eval on Claude, light/cheap calls on Ollama
- Best for cost-conscious learning if you have a local GPU

---

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended for `sentence-transformers`)
- ~3 GB free disk (sentence-transformer + cross-encoder weights, FAISS, Chroma)
- No system packages required this week 🎉

---

## Installation

```bash
# 1. Clone / cd into the homework folder
cd Homework4-Submission

# 2. Create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# .venv\Scripts\activate                  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key (Path A)
cp .env.example .env
# edit .env -- paste your ANTHROPIC_API_KEY

# 5. (Path B only) install + start Ollama, then pull a model
#    https://ollama.com/download
ollama serve &
ollama pull qwen3.5:27b

# 6. Verify everything in nb00
jupyter notebook notebooks/00_setup_verification.ipynb
```

---

## Repository Structure

```
Homework4-Submission/
├── README.md                             # This file
├── requirements.txt                      # All Python deps
├── .env.example                          # Copy to .env
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 00_setup_verification.ipynb       # ~5 min  -- env check
│   ├── 01_environment_setup.ipynb        # ~20 min -- pick path A/B/C
│   ├── 02_document_loading.ipynb         # ~30 min -- PDF, arXiv, web
│   ├── 03_chunking_strategies.ipynb      # ~30 min -- recursive, semantic, contextual
│   ├── 04_embeddings_deep_dive.ipynb     # ~30 min -- MiniLM, BGE, OpenAI
│   ├── 05_vector_stores.ipynb            # ~30 min -- FAISS, Chroma, Qdrant
│   ├── 06_retrieval_strategies.ipynb     # ~35 min -- hybrid, HyDE, MMR, multi-query
│   ├── 07_reranking_evaluation.ipynb     # ~35 min -- cross-encoder + RAGAS metrics
│   └── 08_project_integration.ipynb      # ~40 min -- Resume RAG agent + FastAPI
├── src/
│   ├── __init__.py
│   ├── config.py                         # PATH, CLAUDE_MODEL, OLLAMA_MODEL
│   ├── llm_client.py                     # Unified Claude + Ollama client (reused)
│   ├── cost_tracker.py                   # Reused
│   ├── utils.py                          # Reused
│   ├── prompt_templates.py               # CO-STAR templates
│   ├── document_loader.py                # PDF, arXiv, web, directory loaders   [NEW]
│   ├── chunking.py                       # Recursive, semantic, contextual      [NEW]
│   ├── embeddings.py                     # Multi-backend embedding wrapper       [NEW]
│   ├── vector_store.py                   # FAISS / Chroma / Qdrant unified      [NEW]
│   ├── retrieval.py                      # BM25, hybrid+RRF, HyDE, MMR          [NEW]
│   ├── reranker.py                       # CrossEncoder, FlashRank, Cohere      [NEW]
│   ├── rag_evaluation.py                 # Faithfulness, recall, precision      [NEW]
│   └── rag_pipeline.py                   # End-to-end pipeline glue             [NEW]
├── outputs/                              # Auto-created by notebooks
│   ├── homework_reflection.md            # 70% — built incrementally
│   ├── my_project_update.md              # 20% — generated by nb08
│   ├── path_selection.md                 # nb01
│   ├── setup_summary.txt                 # nb01
│   ├── corpus_stats.json                 # nb02
│   ├── embedding_scoreboard.json         # nb04
│   ├── retrieval_scoreboard.json         # nb06
│   ├── eval_result_07.json               # nb07
│   ├── ab_reranker.json                  # nb07
│   ├── demo_qa.json                      # nb08
│   ├── project_index/                    # nb08 -- persisted FAISS index
│   ├── chroma_db/                        # nb05 -- persisted Chroma DB
│   └── main.py                           # nb08 bonus -- FastAPI server
├── test_data/
│   ├── sample_resume.pdf                 # Shipped fixture
│   ├── portfolio_notes.txt               # Shipped fixture
│   ├── arxiv/                            # nb02 downloads
│   └── my_corpus/                        # YOUR docs (curated in nb02)
└── docs/
```

---

## Assignment Structure

| Notebook | Topic | Time | Key Deliverable |
|---|---|---|---|
| 00 | Setup Verification | 5 min | — |
| 01 | Environment Setup | 20 min | `path_selection.md` |
| 02 | Document Loading & Extraction | 30 min | `corpus_stats.json` + curated `my_corpus/` |
| 03 | Chunking Strategies | 30 min | Reflection + chunk-length plots |
| 04 | Embeddings Deep Dive | 30 min | `embedding_scoreboard.json` |
| 05 | Vector Stores | 30 min | Persisted Chroma + FAISS indexes |
| 06 | Retrieval Strategies | 35 min | `retrieval_scoreboard.json` |
| 07 | Reranking & Evaluation | 35 min | `ab_reranker.json` + RAGAS scores |
| 08 | **Project Integration** | 40 min | `my_project_update.md` + `outputs/main.py` |

**Total estimated time:** ~4–5 hours

---

## Deliverables

### Required (graded)

| File | % | Notes |
|---|---|---|
| `outputs/homework_reflection.md` | **70%** | Built incrementally by `append_to_reflection()` in nb01-08. Depth, evidence of experimentation, and clear reasoning. |
| `outputs/my_project_update.md`   | **20%** | Generated by nb08. Architecture, demo Q&A with sources + faithfulness scores, adversarial test, FastAPI write-up, next-step. |
| All 9 notebooks executed         | **10%** | TODOs filled in, no runtime errors. |

### Bonus (extra credit)

- Working **FastAPI** server (`outputs/main.py`) with a `curl` transcript in your reflection (+5%)
- Try a third embedding model not in the default set (Voyage / OpenAI / BGE-M3) and compare (+3%)
- Add a **knowledge-graph** layer (e.g. `networkx` over entity mentions) and show one query where it beats vanilla hybrid (+5%)
- Replace our `evaluate_rag` with the real **RAGAS** library on the same questions and compare scores (+3%)

---

## Cost Estimates

| Path | Model | Approx. cost (full week) |
|---|---|---|
| A | claude-sonnet-4-6 + claude-haiku-4-5 (eval) | $0.50 – $2.00 |
| B | qwen3.5:27b (Ollama, local) | $0.00 |
| C | mostly haiku for eval, sonnet for nb08 | $0.20 – $0.80 |

*Major cost drivers: nb03 contextual chunking (~10 LLM calls), nb06 multi-query rewriting + HyDE (~30 calls), nb07 RAGAS eval (~50 LLM-judge calls), nb08 demo + adversarial (~20 calls). The eval metrics dominate — that's why we default the judge to Haiku.*

---

## Troubleshooting

### `import sentence_transformers` is slow first time
Downloads ~80MB on first call. Cached after that. If behind a firewall, set `HF_HUB_OFFLINE=1` once cached.

### `import faiss` fails on Mac
`pip install faiss-cpu` (the GPU build doesn't ship for macOS — that's fine, CPU is plenty for this homework).

### `chromadb` complains about `sqlite3`
Some Linux distros ship a too-old SQLite. Quick fix: `pip install pysqlite3-binary` then add at the top of any failing notebook:
```python
__import__('pysqlite3'); import sys; sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### `pytubefix` / `pypdf` warnings about deprecation
Safe to ignore — the chunked output is still valid.

### Ollama is slow on Apple Silicon
qwen3.5:27b is ~17GB and won't fit in unified memory on most M1 / 8GB devices. Use `llama3.1:8b` instead, or switch to Path A.

### arXiv `403 Forbidden`
You're rate-limited. Wait 60 seconds, or set `download_pdfs=False` in `fetch_arxiv_papers` and use the metadata only.

### `[Errno 28] No space left on device` from sentence-transformers
Big models (BGE-base) cache to `~/.cache/huggingface`. Either clean it or set `HF_HOME=/some/large/disk`.

---

## Resources

### RAG concepts
- [Anthropic — Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval)
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [LangChain RAG docs](https://python.langchain.com/docs/concepts/rag/)
- [LlamaIndex RAG docs](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

### Embeddings & rerankers
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — current SOTA
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — versatile open-source
- [Cohere Rerank docs](https://docs.cohere.com/docs/rerank-overview)

### Vector stores
- [Chroma docs](https://docs.trychroma.com/)
- [Qdrant quickstart](https://qdrant.tech/documentation/quickstart/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)

### Evaluation
- [RAGAS docs](https://docs.ragas.io/)
- [DeepEval docs](https://docs.confident-ai.com/)
- [TruLens RAG triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)

---

## Timeline (suggested 7-day schedule)

| Day | Notebooks | Focus |
|---|---|---|
| Mon | 00, 01 | Get the env right, pick your path |
| Tue | 02 | Curate your corpus (this matters a lot — pick docs you actually want to query) |
| Wed | 03 | Try at least 2 chunking strategies on your corpus |
| Thu | 04, 05 | Pick your embedding model + vector store |
| Fri | 06 | Retrieval bake-off — find what works on YOUR data |
| Sat | 07 | Add the reranker, run RAGAS metrics |
| Sun | 08 | Build the Resume RAG Agent + (bonus) FastAPI; write `my_project_update.md` |

---

## What's Next

**Week 5: Supervised Fine-Tuning (SFT)** — you'll move from retrieval to *teaching* the model your style with custom Q&A pairs. Many real-world systems combine the two: RAG for knowledge, SFT for tone.

---

## Support

- Discord: `#week-4` channel
- Office hours: see course calendar
- Stuck? Open an issue or message the instructor with: (a) which notebook, (b) the cell, (c) the full traceback
