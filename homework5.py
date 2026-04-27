import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score
import json
from pathlib import Path
import sys


DOCUMENTS = [
    {
        'doc_id': 1,
        'title': 'Resume',
        'content': 'Python developer with Django experience at DoorDash. Built data pipelines using PostgreSQL. Worked on REST APIs.',
        'keywords': 'Python,Django,PostgreSQL,DoorDash,REST',
        'year': 2024
    },
    {
        'doc_id': 2,
        'title': 'Portfolio',
        'content': 'Cryptocurrency Tracker: Full-stack application with Python backend, PostgreSQL database, and D3.js frontend for data visualization.',
        'keywords': 'Python,PostgreSQL,D3.js,API,Full-stack',
        'year': 2024
    },
    {
        'doc_id': 3,
        'title': 'Skills',
        'content': 'Proficient in Python, JavaScript, Java. Database experience: PostgreSQL, Oracle, MongoDB. Frontend: React, Angular.',
        'keywords': 'Python,JavaScript,Java,PostgreSQL,Oracle,MongoDB,React,Angular',
        'year': 2024
    },
    {
        'doc_id': 4,
        'title': 'Experience',
        'content': 'Worked on REST APIs with Django framework. Handled cloud infrastructure with AWS. Deployed to production using Docker.',
        'keywords': 'Django,REST,AWS,Cloud,Docker,Production',
        'year': 2024
    },
    {
        'doc_id': 5,
        'title': 'Projects',
        'content': 'Led a team to build machine learning pipelines. Used Python and TensorFlow for deep learning models. Data processing with Pandas.',
        'keywords': 'Python,TensorFlow,ML,Pipeline,Pandas,DeepLearning',
        'year': 2024
    },
    {
        'doc_id': 6,
        'title': 'Backend',
        'content': 'Specialized in backend development using Python. Built microservices with FastAPI. Used Redis for caching.',
        'keywords': 'Python,FastAPI,Microservices,Redis,Backend',
        'year': 2024
    },
    {
        'doc_id': 7,
        'title': 'Frontend',
        'content': 'Frontend development with React and TypeScript. Built responsive UI with CSS. Integrated with GraphQL APIs.',
        'keywords': 'React,TypeScript,CSS,GraphQL,Frontend,UI',
        'year': 2024
    },
    {
        'doc_id': 8,
        'title': 'DevOps',
        'content': 'DevOps engineer with Kubernetes and Docker experience. Managed CI/CD pipelines with Jenkins. AWS infrastructure as code.',
        'keywords': 'Kubernetes,Docker,Jenkins,CI/CD,AWS,Infrastructure',
        'year': 2024
    },
]

TEST_QUERIES = [
    {'query': 'What Python projects?', 'relevant': [1, 2, 5, 6]},
    {'query': 'PostgreSQL experience', 'relevant': [1, 2, 3]},
    {'query': 'Django and REST APIs', 'relevant': [1, 4]},
    {'query': 'Frontend with JavaScript', 'relevant': [3, 7]},
    {'query': 'Cloud infrastructure', 'relevant': [4, 8]},
    {'query': 'Machine learning', 'relevant': [5]},
    {'query': 'Database management', 'relevant': [1, 2, 3]},
    {'query': 'Full-stack development', 'relevant': [1, 2, 6, 7]},
    {'query': 'AWS services', 'relevant': [4, 8]},
    {'query': 'Team leadership', 'relevant': [5, 8]},
]


class HybridIndexBuilder:
    def __init__(self, db_path="hybrid_index.db", embedding_model="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.conn = sqlite3.connect(db_path)
        self.faiss_index = None
        self.doc_id_to_text = {}
        self.doc_ids = None
        
    def create_schema(self):
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                keywords TEXT,
                year INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
                content,
                content='documents',
                content_rowid='doc_id'
            )
        """)
        
        self.conn.commit()
        print("✓ SQLite schema created")
    
    def insert_documents(self, documents):
        cursor = self.conn.cursor()
        
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM doc_chunks")
        
        texts = []
        doc_ids = []
        
        for doc in documents:
            cursor.execute("""
                INSERT INTO documents 
                (doc_id, title, content, keywords, year)
                VALUES (?, ?, ?, ?, ?)
            """, (doc['doc_id'], doc['title'], doc['content'], 
                  doc['keywords'], doc['year']))
            
            cursor.execute("""
                INSERT INTO doc_chunks(rowid, content)
                VALUES (?, ?)
            """, (doc['doc_id'], doc['content']))
            
            texts.append(doc['content'])
            doc_ids.append(doc['doc_id'])
            self.doc_id_to_text[doc['doc_id']] = doc['content']
        
        self.conn.commit()
        
        try:
            import faiss
            embeddings = self.embedding_model.encode(texts)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            self.faiss_index.add(embeddings)
            self.doc_ids = np.array(doc_ids)
            
            print(f"✓ Inserted {len(documents)} documents")
            print(f"✓ FAISS index contains {self.faiss_index.ntotal} vectors")
        except ImportError:
            print("⚠ FAISS not installed, using BM25 only")
            self.faiss_index = None


class HybridSearcher:
    def __init__(self, builder):
        self.builder = builder
        self.conn = builder.conn
        self.embedding_model = builder.embedding_model
        
        texts = [doc['content'] for doc in self._get_all_documents()]
        self.bm25 = BM25Okapi([text.split() for text in texts])
        self.texts = texts
    
    def _get_all_documents(self):
        cursor = self.conn.cursor()
        docs = cursor.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [
            {
                'doc_id': d[0],
                'title': d[1],
                'content': d[2],
                'keywords': d[3],
                'year': d[4]
            }
            for d in docs
        ]
    
    def vector_search(self, query, k=10):
        if self.builder.faiss_index is None:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.builder.faiss_index.search(query_embedding, k)
        
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):
            similarity = 1 / (1 + distance)
            results.append({
                'doc_id': int(self.builder.doc_ids[idx]),
                'score': float(similarity),
                'rank': rank,
                'method': 'vector'
            })
        
        return results
    
    def keyword_search(self, query, k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), 
                            key=lambda i: scores[i], 
                            reverse=True)[:k]
        
        results = []
        docs = self._get_all_documents()
        
        for rank, idx in enumerate(top_indices, 1):
            if idx < len(docs):
                score = scores[idx]
                max_score = max(scores)
                normalized_score = score / max_score if max_score > 0 else 0
                
                results.append({
                    'doc_id': docs[idx]['doc_id'],
                    'score': float(normalized_score),
                    'rank': rank,
                    'method': 'keyword'
                })
        
        return results


class ScoreFusion:
    @staticmethod
    def reciprocal_rank_fusion(vec_results, kw_results, k=3, k_param=60):
        combined = {}
        
        for result in vec_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_score = 1 / (k_param + rank)
            combined[doc_id] = combined.get(doc_id, 0) + rrf_score
        
        for result in kw_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_score = 1 / (k_param + rank)
            combined[doc_id] = combined.get(doc_id, 0) + rrf_score
        
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [
            {'doc_id': doc_id, 'score': score, 'rank': rank, 'method': 'hybrid_rrf'}
            for rank, (doc_id, score) in enumerate(ranked, 1)
        ]
    
    @staticmethod
    def weighted_sum(vec_results, kw_results, k=3, alpha=0.6):
        vec_dict = {r['doc_id']: r['score'] for r in vec_results}
        kw_dict = {r['doc_id']: r['score'] for r in kw_results}
        
        combined = {}
        all_docs = set(vec_dict.keys()) | set(kw_dict.keys())
        
        for doc_id in all_docs:
            v_score = vec_dict.get(doc_id, 0)
            k_score = kw_dict.get(doc_id, 0)
            combined[doc_id] = alpha * v_score + (1 - alpha) * k_score
        
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [
            {'doc_id': doc_id, 'score': score, 'rank': rank, 'method': f'hybrid_weighted(α={alpha})'}
            for rank, (doc_id, score) in enumerate(ranked, 1)
        ]


class HybridRetrieval:
    def __init__(self, builder, searcher):
        self.builder = builder
        self.searcher = searcher
        self.fusion = ScoreFusion()
    
    def search(self, query, k=3, method='rrf'):
        vec_results = self.searcher.vector_search(query, k=10)
        kw_results = self.searcher.keyword_search(query, k=10)
        
        if method == 'rrf':
            final_results = self.fusion.reciprocal_rank_fusion(vec_results, kw_results, k=k)
        else:
            final_results = self.fusion.weighted_sum(vec_results, kw_results, k=k, alpha=0.6)
        
        for result in final_results:
            doc = self.builder.conn.execute(
                "SELECT title, content FROM documents WHERE doc_id = ?",
                (result['doc_id'],)
            ).fetchone()
            if doc:
                result['title'] = doc[0]
                result['content'] = doc[1][:100] + "..."
        
        return final_results


class Evaluator:
    @staticmethod
    def recall_at_k(retrieved, relevant, k=3):
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0
        
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    @staticmethod
    def mrr(retrieved, relevant):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1 / rank
        return 0
    
    @staticmethod
    def ndcg_at_k(retrieved, relevant, k=3):
        relevances = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
        ideal_relevances = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
        
        if sum(relevances) == 0:
            return 0
        
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0


def evaluate_all_methods(test_queries, builder, searcher, hybrid_retrieval):
    evaluator = Evaluator()
    results = {
        'vector_only': {'recall@3': [], 'mrr': [], 'ndcg@3': []},
        'keyword_only': {'recall@3': [], 'mrr': [], 'ndcg@3': []},
        'hybrid_rrf': {'recall@3': [], 'mrr': [], 'ndcg@3': []},
        'hybrid_weighted': {'recall@3': [], 'mrr': [], 'ndcg@3': []},
    }
    
    for query_data in test_queries:
        query = query_data['query']
        relevant = query_data['relevant']
        
        vec_results = searcher.vector_search(query, k=10)
        vec_docs = [r['doc_id'] for r in vec_results]
        results['vector_only']['recall@3'].append(evaluator.recall_at_k(vec_docs, relevant, k=3))
        results['vector_only']['mrr'].append(evaluator.mrr(vec_docs, relevant))
        results['vector_only']['ndcg@3'].append(evaluator.ndcg_at_k(vec_docs, relevant, k=3))
        
        kw_results = searcher.keyword_search(query, k=10)
        kw_docs = [r['doc_id'] for r in kw_results]
        results['keyword_only']['recall@3'].append(evaluator.recall_at_k(kw_docs, relevant, k=3))
        results['keyword_only']['mrr'].append(evaluator.mrr(kw_docs, relevant))
        results['keyword_only']['ndcg@3'].append(evaluator.ndcg_at_k(kw_docs, relevant, k=3))
        
        hybrid_rrf = hybrid_retrieval.search(query, k=10, method='rrf')
        hybrid_rrf_docs = [r['doc_id'] for r in hybrid_rrf]
        results['hybrid_rrf']['recall@3'].append(evaluator.recall_at_k(hybrid_rrf_docs, relevant, k=3))
        results['hybrid_rrf']['mrr'].append(evaluator.mrr(hybrid_rrf_docs, relevant))
        results['hybrid_rrf']['ndcg@3'].append(evaluator.ndcg_at_k(hybrid_rrf_docs, relevant, k=3))
        
        hybrid_weighted = hybrid_retrieval.search(query, k=10, method='weighted')
        hybrid_weighted_docs = [r['doc_id'] for r in hybrid_weighted]
        results['hybrid_weighted']['recall@3'].append(evaluator.recall_at_k(hybrid_weighted_docs, relevant, k=3))
        results['hybrid_weighted']['mrr'].append(evaluator.mrr(hybrid_weighted_docs, relevant))
        results['hybrid_weighted']['ndcg@3'].append(evaluator.ndcg_at_k(hybrid_weighted_docs, relevant, k=3))
    
    return results


def main():
    print("=" * 70)
    print("Week 5: Hybrid Retrieval System")
    print("=" * 70)
    
    print("\n[1/4] Building index...")
    builder = HybridIndexBuilder()
    builder.create_schema()
    builder.insert_documents(DOCUMENTS)
    
    print("\n[2/4] Initializing searcher...")
    searcher = HybridSearcher(builder)
    hybrid_retrieval = HybridRetrieval(builder, searcher)
    
    print("\n[3/4] Evaluating methods...")
    results = evaluate_all_methods(TEST_QUERIES, builder, searcher, hybrid_retrieval)
    
    print("\n[4/4] Results:")
    print("=" * 70)
    
    for method, metrics in results.items():
        print(f"\n{method.upper()}")
        print(f"  Recall@3:  {np.mean(metrics['recall@3']):.2%}")
        print(f"  MRR:       {np.mean(metrics['mrr']):.2%}")
        print(f"  NDCG@3:    {np.mean(metrics['ndcg@3']):.2%}")
    
    print("\n" + "=" * 70)
    print("Example Search Results")
    print("=" * 70)
    
    demo_query = "What Python projects?"
    print(f"\nQuery: {demo_query}")
    sys.stdout.flush()
    
    hybrid_results = hybrid_retrieval.search(demo_query, k=3, method='rrf')
    
    if hybrid_results:
        for result in hybrid_results:
            print(f"\n#{result['rank']} (doc_id={result['doc_id']}, score={result['score']:.3f})")
            print(f"  Title: {result['title']}")
            print(f"  Content: {result['content']}")
        sys.stdout.flush()
    
    print("\n" + "=" * 70)
    output_data = {
        'metrics': {k: {mk: float(np.mean(v[mk])) for mk in v} for k, v in results.items()},
        'test_queries': len(TEST_QUERIES),
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("✓ Results saved to evaluation_results.json")
    sys.stdout.flush()


if __name__ == "__main__":
    main()