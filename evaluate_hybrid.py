"""
Evaluation Script for Hybrid Retrieval System
Tests and compares semantic, keyword, and hybrid search methods
"""

from hybrid_rag import HybridRAGPipeline
import json

# Initialize pipeline
print("Loading hybrid RAG pipeline...")
pipeline = HybridRAGPipeline()

if not pipeline.load():
    print("ERROR: No processed data found. Run hybrid_rag.py first!")
    exit(1)

# Define test queries with known relevant chunks
# NOTE: You'll need to manually identify relevant chunks for your dataset
test_queries = [
    {
        "query": "machine learning algorithms",
        "relevant_chunks": [16, 14, 13]  # Example: manually identified relevant chunk IDs
    },
    {
        "query": "neural network training",
        "relevant_chunks": [16, 17, 2]
    },
    {
        "query": "supervised learning methods",
        "relevant_chunks": [2, 13, 16]
    },
    {
        "query": "deep learning optimization",
        "relevant_chunks": [16, 17, 4]
    },
    {
        "query": "feature extraction techniques",
        "relevant_chunks": [5, 3, 7]
    },
    {
        "query": "backpropagation algorithm",
        "relevant_chunks": [17, 6, 16]
    },
    {
        "query": "convolutional neural networks",
        "relevant_chunks": [16, 17, 4]
    },
    {
        "query": "gradient descent",
        "relevant_chunks": [2, 17, 3]
    },
    {
        "query": "model evaluation metrics",
        "relevant_chunks": [10, 9, 11]
    },
    {
        "query": "LLM",
        "relevant_chunks": [18, 4, 12]
    }
]

print(f"\n{'='*80}")
print("HYBRID RETRIEVAL EVALUATION")
print(f"{'='*80}")
print(f"\nTest Set: {len(test_queries)} queries")
print(f"Evaluation Metric: Recall@3 (proportion of relevant docs in top-3)\n")

# Evaluate all methods
results = pipeline.evaluate_retrieval(test_queries, k=3)

# Print detailed results
print(f"\n{'='*80}")
print("DETAILED RESULTS BY QUERY")
print(f"{'='*80}")

for method_name, data in results.items():
    print(f"\n{method_name.upper()} Search Results:")
    print("-" * 80)
    for query_result in data['queries']:
        print(f"Query: '{query_result['query']}'")
        print(f"  Recall@3: {query_result['recall']:.2%} ({query_result['hits']}/{query_result['relevant']} relevant found)")

# Compare methods
print(f"\n{'='*80}")
print("OVERALL COMPARISON")
print(f"{'='*80}\n")

comparison = []
for method_name, data in results.items():
    comparison.append({
        'method': method_name,
        'avg_recall': data['average_recall']
    })

comparison.sort(key=lambda x: x['avg_recall'], reverse=True)

print(f"{'Method':<15} {'Recall@3':<15} {'Improvement':<15}")
print("-" * 45)

baseline_recall = comparison[-1]['avg_recall']  # Lowest performing method as baseline

for item in comparison:
    method = item['method']
    recall = item['avg_recall']
    improvement = ((recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
    
    print(f"{method.capitalize():<15} {recall:<15.2%} {improvement:>13.1f}%")

# Save results to JSON
output_file = "evaluation_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Detailed results saved to: {output_file}")

# Example queries demonstration
print(f"\n{'='*80}")
print("EXAMPLE SEARCH COMPARISON")
print(f"{'='*80}")

example_query = "neural network training"
print(f"\nQuery: '{example_query}'")
print("-" * 80)

print("\n1. SEMANTIC SEARCH (FAISS):")
sem_results = pipeline.semantic_search(example_query, k=3)
for i, r in enumerate(sem_results, 1):
    print(f"  Rank {i} (Score: {r['score']:.3f}): {r['chunk'][:80]}...")

print("\n2. KEYWORD SEARCH (BM25):")
kw_results = pipeline.keyword_search(example_query, k=3)
for i, r in enumerate(kw_results, 1):
    print(f"  Rank {i} (Score: {r['score']:.3f}): {r['chunk'][:80]}...")

print("\n3. HYBRID SEARCH (70% semantic + 30% keyword):")
hyb_results = pipeline.hybrid_search(example_query, k=3, alpha=0.7)
for i, r in enumerate(hyb_results, 1):
    print(f"  Rank {i} (Hybrid: {r['hybrid_score']:.3f}, Sem: {r['semantic_score']:.3f}, Kw: {r['keyword_score']:.3f})")
    print(f"    {r['chunk'][:80]}...")

print(f"\n{'='*80}")
print("ANALYSIS NOTES:")
print(f"{'='*80}")
print("""
Key Observations:
1. Semantic search excels at finding conceptually similar content
2. Keyword search is better for exact term matching
3. Hybrid search combines strengths of both approaches

Recommendations:
- Use alpha=0.7 (70% semantic, 30% keyword) as a good default
- Adjust alpha based on query type:
  * High alpha (0.8-0.9) for conceptual queries
  * Low alpha (0.3-0.5) for exact terminology queries
- Hybrid search generally provides best overall performance
""")

print(f"\n{'='*80}")
print("✅ Evaluation Complete!")
print(f"{'='*80}")
