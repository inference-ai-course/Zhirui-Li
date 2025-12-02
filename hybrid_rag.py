"""
Hybrid Retrieval System - Week 5 Extension
Combines dense vector semantic search (FAISS) with sparse keyword search (BM25)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import ssl
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

import fitz 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import arxiv
from rank_bm25 import BM25Okapi

# FIX SSL CERTIFICATE ISSUES
ssl._create_default_https_context = ssl._create_unverified_context
print("✅ SSL verification configured")
print("✅ Running in CPU-ONLY mode")
print("✅ Hybrid Retrieval: FAISS + BM25\n")


class HybridRAGPipeline:
    """Hybrid RAG pipeline combining semantic (FAISS) and keyword (BM25) search"""
    
    def __init__(self, 
                 papers_dir: str = "papers",
                 processed_dir: str = "processed",
                 db_path: str = "hybrid_index.db",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self.db_path = db_path
        self.papers_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        print(f"Loading embedding model: {model_name} (CPU mode)")
        self.model = SentenceTransformer(model_name, device='cpu')
        print("✅ Model loaded on CPU")
        
        # Storage for chunks and metadata
        self.chunks = []
        self.chunk_metadata = []
        self.paper_metadata = []  # NEW: Store paper-level metadata
        
        # Search indices
        self.faiss_index = None
        self.bm25_index = None
        
        # Initialize SQLite database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with FTS5 for keyword search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table for metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE,
                title TEXT,
                authors TEXT,
                year INTEGER,
                abstract TEXT,
                keywords TEXT,
                arxiv_id TEXT
            )
        """)
        
        # Create chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                content TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        
        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
            USING FTS5(content, content='chunks', content_rowid='chunk_id')
        """)
        
        conn.commit()
        conn.close()
        print("✅ SQLite database initialized")
        
    def download_papers(self, query: str, max_results: int = 50):
        """Download papers from arXiv with metadata extraction"""
        print(f"Searching arXiv for: '{query}'")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        downloaded = 0
        errors = 0
        papers_metadata = []
        
        for paper in client.results(search):
            try:
                paper_id = paper.get_short_id()
                pdf_path = self.papers_dir / f"{paper_id}.pdf"
                
                # Extract metadata
                metadata = {
                    'paper_id': paper_id,
                    'title': paper.title,
                    'authors': ', '.join([a.name for a in paper.authors]),
                    'year': paper.published.year,
                    'abstract': paper.summary,
                    'keywords': ', '.join(paper.categories) if paper.categories else '',
                    'arxiv_id': paper.entry_id
                }
                papers_metadata.append(metadata)
                
                if pdf_path.exists():
                    print(f"  ✓ Already exists: {paper_id}")
                    downloaded += 1
                    continue
                
                paper.download_pdf(filename=str(pdf_path))
                print(f"  ✓ Downloaded: {paper_id} - {paper.title[:50]}...")
                downloaded += 1
                
            except Exception as e:
                errors += 1
                print(f"  ✗ Error: {str(e)[:80]}")
                continue
        
        # Save metadata to database
        self._save_paper_metadata(papers_metadata)
        self.paper_metadata = papers_metadata
                
        print(f"\n✅ Downloaded/found: {downloaded} papers")
        if errors > 0:
            print(f"⚠️  Failed: {errors} papers")
        return downloaded
    
    def _save_paper_metadata(self, papers_metadata: List[Dict]):
        """Save paper metadata to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metadata in papers_metadata:
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (paper_id, title, authors, year, abstract, keywords, arxiv_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['paper_id'],
                metadata['title'],
                metadata['authors'],
                metadata['year'],
                metadata['abstract'],
                metadata['keywords'],
                metadata['arxiv_id']
            ))
        
        conn.commit()
        conn.close()
        print("✅ Paper metadata saved to database")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page in doc:
                page_text = page.get_text()
                pages.append(page_text)
            
            full_text = "\n".join(pages)
            doc.close()
            return full_text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        step = max_tokens - overlap
        
        for i in range(0, len(words), step):
            chunk = words[i:i + max_tokens]
            chunk_text = " ".join(chunk)
            if len(chunk_text.strip()) > 50:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_all_papers(self, max_tokens: int = 512, overlap: int = 50):
        """Process papers and store in SQLite database"""
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found.")
            return
        
        print(f"\nProcessing {len(pdf_files)} papers...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_chunks = []
        all_metadata = []
        
        for idx, pdf_path in enumerate(pdf_files):
            paper_id = pdf_path.stem
            print(f"  [{idx+1}/{len(pdf_files)}] Processing {paper_id}...")
            
            # Get doc_id from database
            cursor.execute("SELECT doc_id FROM documents WHERE paper_id = ?", (paper_id,))
            result = cursor.fetchone()
            if not result:
                print(f"    ⚠️  Paper metadata not found in database")
                continue
            doc_id = result[0]
            
            # Extract text
            text = self.extract_text_from_pdf(str(pdf_path))
            
            if not text:
                print(f"    ⚠️  No text extracted")
                continue
            
            # Chunk text
            chunks = self.chunk_text(text, max_tokens, overlap)
            print(f"    ✓ Created {len(chunks)} chunks")
            
            # Store chunks in database
            for chunk_idx, chunk in enumerate(chunks):
                cursor.execute("""
                    INSERT INTO chunks (doc_id, chunk_index, content)
                    VALUES (?, ?, ?)
                """, (doc_id, chunk_idx, chunk))
                
                all_chunks.append(chunk)
                all_metadata.append({
                    "paper_id": paper_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                })
        
        # Update FTS5 index
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        
        conn.commit()
        conn.close()
        
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        
        print(f"\n✅ Total chunks: {len(self.chunks)}")
        print("✅ Chunks stored in SQLite database")
    
    def build_faiss_index(self):
        """Build FAISS index for semantic search"""
        if not self.chunks:
            print("No chunks to index.")
            return
        
        print(f"\nGenerating embeddings for {len(self.chunks)} chunks (CPU mode)...")
        print("⏳ This will take a few minutes on CPU...")
        
        batch_size = 16
        embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 5 == 0 or i == 0:
                print(f"  Progress: {i + len(batch)}/{len(self.chunks)} chunks")
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        print("Building FAISS index...")
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def build_bm25_index(self):
        """Build BM25 index for keyword search"""
        if not self.chunks:
            print("No chunks to index.")
            return
        
        print("\nBuilding BM25 index for keyword search...")
        
        # Tokenize chunks
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        print(f"✅ BM25 index built with {len(self.chunks)} documents")
    
    def semantic_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built.")
        
        query_embedding = self.model.encode([query])
        distances, indices = self.faiss_index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    "chunk_id": int(idx),
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "score": float(similarity),
                    "distance": float(distances[0][i]),
                    "method": "semantic"
                })
        
        return results
    
    def keyword_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Keyword search using BM25"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "chunk_id": int(idx),
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "score": float(scores[idx]),
                    "method": "keyword"
                })
        
        return results
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (0-1). keyword weight = 1-alpha
        
        Returns:
            Combined and re-ranked results
        """
        # Get results from both methods (retrieve more for better fusion)
        semantic_results = self.semantic_search(query, k=min(50, len(self.chunks)))
        keyword_results = self.keyword_search(query, k=min(50, len(self.chunks)))
        
        # Normalize scores to 0-1 range
        def normalize_scores(results):
            if not results:
                return results
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range == 0:
                return results
            
            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / score_range
            return results
        
        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Combine scores using weighted sum
        combined_scores = {}
        
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_scores[chunk_id] = {
                'semantic_score': result['normalized_score'] * alpha,
                'keyword_score': 0,
                'chunk': result['chunk'],
                'metadata': result['metadata']
            }
        
        for result in keyword_results:
            chunk_id = result['chunk_id']
            keyword_score = result['normalized_score'] * (1 - alpha)
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['keyword_score'] = keyword_score
            else:
                combined_scores[chunk_id] = {
                    'semantic_score': 0,
                    'keyword_score': keyword_score,
                    'chunk': result['chunk'],
                    'metadata': result['metadata']
                }
        
        # Calculate final scores
        final_results = []
        for chunk_id, scores in combined_scores.items():
            final_score = scores['semantic_score'] + scores['keyword_score']
            final_results.append({
                "chunk_id": chunk_id,
                "chunk": scores['chunk'],
                "metadata": scores['metadata'],
                "hybrid_score": final_score,
                "semantic_score": scores['semantic_score'],
                "keyword_score": scores['keyword_score'],
                "method": "hybrid"
            })
        
        # Sort by final score and return top-k
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return final_results[:k]
    
    def evaluate_retrieval(self, test_queries: List[Dict], k: int = 3) -> Dict[str, float]:
        """
        Evaluate retrieval quality using recall@k
        
        Args:
            test_queries: List of dict with 'query' and 'relevant_chunks' (list of chunk IDs)
            k: Number of top results to consider
        
        Returns:
            Dict with recall scores for each method
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Retrieval Performance (Recall@{k})")
        print(f"{'='*60}")
        
        methods = {
            'semantic': self.semantic_search,
            'keyword': self.keyword_search,
            'hybrid': self.hybrid_search
        }
        
        results_summary = {}
        
        for method_name, search_func in methods.items():
            total_recall = 0
            query_results = []
            
            for test in test_queries:
                query = test['query']
                relevant_chunks = set(test['relevant_chunks'])
                
                # Get search results
                search_results = search_func(query, k=k)
                retrieved_chunks = {r['chunk_id'] for r in search_results}
                
                # Calculate recall
                hits = len(retrieved_chunks & relevant_chunks)
                recall = hits / len(relevant_chunks) if relevant_chunks else 0
                
                total_recall += recall
                query_results.append({
                    'query': query,
                    'recall': recall,
                    'hits': hits,
                    'relevant': len(relevant_chunks)
                })
            
            avg_recall = total_recall / len(test_queries) if test_queries else 0
            results_summary[method_name] = {
                'average_recall': avg_recall,
                'queries': query_results
            }
            
            print(f"\n{method_name.upper()} Search:")
            print(f"  Average Recall@{k}: {avg_recall:.2%}")
        
        return results_summary
    
    def save(self):
        """Save processed data and indices"""
        print("\nSaving processed data...")
        
        with open(self.processed_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(self.processed_dir / "chunk_metadata.pkl", "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        
        with open(self.processed_dir / "paper_metadata.pkl", "wb") as f:
            pickle.dump(self.paper_metadata, f)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.processed_dir / "faiss_index.bin"))
        
        if self.bm25_index is not None:
            with open(self.processed_dir / "bm25_index.pkl", "wb") as f:
                pickle.dump(self.bm25_index, f)
        
        print("✅ Data saved!")
    
    def load(self):
        """Load processed data and indices"""
        print("\nLoading processed data...")
        
        chunks_path = self.processed_dir / "chunks.pkl"
        
        if not chunks_path.exists():
            print("No processed data found.")
            return False
        
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(self.processed_dir / "chunk_metadata.pkl", "rb") as f:
            self.chunk_metadata = pickle.load(f)
        
        paper_metadata_path = self.processed_dir / "paper_metadata.pkl"
        if paper_metadata_path.exists():
            with open(paper_metadata_path, "rb") as f:
                self.paper_metadata = pickle.load(f)
        
        faiss_path = self.processed_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        bm25_path = self.processed_dir / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        if self.faiss_index:
            print(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        if self.bm25_index:
            print(f"✅ Loaded BM25 index")
        
        return True


# FastAPI Application
app = FastAPI(title="Hybrid RAG Search API")
pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = HybridRAGPipeline()
    if not pipeline.load():
        print("No processed data found.")


@app.get("/")
async def root():
    return {
        "message": "Hybrid RAG Search API (Week 5)",
        "methods": {
            "semantic": "Dense vector search with FAISS",
            "keyword": "Sparse keyword search with BM25",
            "hybrid": "Combined search (default)"
        },
        "endpoints": {
            "/search/semantic": "Semantic search only",
            "/search/keyword": "Keyword search only",
            "/search/hybrid": "Hybrid search (recommended)",
            "/stats": "Get index statistics"
        }
    }


@app.get("/search/semantic")
async def search_semantic(q: str, k: int = 3):
    """Semantic search using FAISS"""
    if pipeline is None or pipeline.faiss_index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    try:
        results = pipeline.semantic_search(q, k)
        return {
            "query": q,
            "method": "semantic",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/search/keyword")
async def search_keyword(q: str, k: int = 3):
    """Keyword search using BM25"""
    if pipeline is None or pipeline.bm25_index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    try:
        results = pipeline.keyword_search(q, k)
        return {
            "query": q,
            "method": "keyword",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/search/hybrid")
async def search_hybrid(q: str, k: int = 3, alpha: float = 0.7):
    """Hybrid search combining semantic and keyword"""
    if pipeline is None or pipeline.faiss_index is None or pipeline.bm25_index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    try:
        results = pipeline.hybrid_search(q, k, alpha)
        return {
            "query": q,
            "method": "hybrid",
            "alpha": alpha,
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/stats")
async def stats():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "total_chunks": len(pipeline.chunks) if pipeline.chunks else 0,
        "faiss_index_size": pipeline.faiss_index.ntotal if pipeline.faiss_index else 0,
        "bm25_index_exists": pipeline.bm25_index is not None,
        "database_path": str(pipeline.db_path),
        "papers_directory": str(pipeline.papers_dir),
        "processed_directory": str(pipeline.processed_dir)
    }

@app.get("/papers")
async def list_papers():
    """Get list of all papers with metadata"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT paper_id, title, authors, year, abstract, keywords 
        FROM documents
    """)
    
    papers = []
    for row in cursor.fetchall():
        papers.append({
            "paper_id": row[0],
            "title": row[1],
            "authors": row[2],
            "year": row[3],
            "abstract": row[4],
            "keywords": row[5]
        })
    
    conn.close()
    return {"total": len(papers), "papers": papers}


@app.get("/papers/{paper_id}")
async def get_paper_info(paper_id: str):
    """Get detailed info for a specific paper"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    conn = sqlite3.connect(pipeline.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT paper_id, title, authors, year, abstract, keywords, arxiv_id
        FROM documents 
        WHERE paper_id = ?
    """, (paper_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return {
        "paper_id": row[0],
        "title": row[1],
        "authors": row[2],
        "year": row[3],
        "abstract": row[4],
        "keywords": row[5],
        "arxiv_id": row[6]
    }

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Hybrid RAG Pipeline Builder - Week 5")
    print("=" * 60)
    
    pipeline = HybridRAGPipeline()
    
    # Check for existing data
    if pipeline.processed_dir.exists() and list(pipeline.processed_dir.glob("*.pkl")):
        load = input("\nFound existing processed data. Load it? (y/n): ").lower()
        if load == 'y':
            if pipeline.load():
                if pipeline.faiss_index and pipeline.bm25_index:
                    print("\n✅ Pipeline ready with existing indices!")
                    print("Run: uvicorn hybrid_rag:app --reload")
                    sys.exit(0)
                else:
                    print("\nIndices not complete. Building missing indices...")
                    if not pipeline.faiss_index:
                        pipeline.build_faiss_index()
                    if not pipeline.bm25_index:
                        pipeline.build_bm25_index()
                    pipeline.save()
                    print("\n✅ Pipeline complete!")
                    print("Run: uvicorn hybrid_rag:app --reload")
                    sys.exit(0)
    
    # Download papers
    print("\n" + "=" * 60)
    print("STEP 1: Download Papers (with metadata)")
    print("=" * 60)
    
    download = input("Download papers from arXiv? (y/n): ").lower()
    
    if download == 'y':
        query = input("Enter search query (default: 'machine learning'): ").strip()
        if not query:
            query = "machine learning"
        
        num_papers = input("Number of papers (default: 10): ").strip()
        num_papers = int(num_papers) if num_papers else 10
        
        downloaded = pipeline.download_papers(query, num_papers)
        
        if downloaded == 0:
            print("\n❌ No papers downloaded.")
            sys.exit(1)
    
    # Process papers
    print("\n" + "=" * 60)
    print("STEP 2: Process Papers")
    print("=" * 60)
    
    process = input("Process papers? (y/n): ").lower()
    
    if process == 'y':
        pipeline.process_all_papers(max_tokens=512, overlap=50)
    else:
        sys.exit(0)
    
    # Build FAISS index
    print("\n" + "=" * 60)
    print("STEP 3: Build FAISS Index (Semantic Search)")
    print("=" * 60)
    
    build_faiss = input("Build FAISS index? (y/n): ").lower()
    
    if build_faiss == 'y':
        pipeline.build_faiss_index()
    else:
        sys.exit(0)
    
    # Build BM25 index
    print("\n" + "=" * 60)
    print("STEP 4: Build BM25 Index (Keyword Search)")
    print("=" * 60)
    
    build_bm25 = input("Build BM25 index? (y/n): ").lower()
    
    if build_bm25 == 'y':
        pipeline.build_bm25_index()
    
    # Test searches
    print("\n" + "=" * 60)
    print("STEP 5: Test Hybrid Search")
    print("=" * 60)
    
    test = input("Test hybrid search? (y/n): ").lower()
    
    if test == 'y':
        query = input("Enter test query: ").strip()
        if query:
            print(f"\n--- SEMANTIC SEARCH ---")
            sem_results = pipeline.semantic_search(query, k=3)
            for r in sem_results[:3]:
                print(f"Score: {r['score']:.3f} | {r['chunk'][:100]}...")
            
            print(f"\n--- KEYWORD SEARCH (BM25) ---")
            kw_results = pipeline.keyword_search(query, k=3)
            for r in kw_results[:3]:
                print(f"Score: {r['score']:.3f} | {r['chunk'][:100]}...")
            
            print(f"\n--- HYBRID SEARCH (70% semantic + 30% keyword) ---")
            hyb_results = pipeline.hybrid_search(query, k=3, alpha=0.7)
            for r in hyb_results[:3]:
                print(f"Hybrid: {r['hybrid_score']:.3f} (Sem: {r['semantic_score']:.3f}, Kw: {r['keyword_score']:.3f})")
                print(f"  {r['chunk'][:100]}...")
    
    # Save
    print("\n" + "=" * 60)
    print("STEP 6: Save Data")
    print("=" * 60)
    
    pipeline.save()
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Pipeline Complete!")
    print("=" * 60)
    print("\nTo start the FastAPI server:")
    print("  uvicorn hybrid_rag:app --reload")
    print("\nEndpoints:")
    print("  /search/semantic?q=query&k=3")
    print("  /search/keyword?q=query&k=3")
    print("  /search/hybrid?q=query&k=3&alpha=0.7")