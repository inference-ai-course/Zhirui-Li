"""
Simple example of using the RAG pipeline
This script demonstrates basic usage without the full pipeline
"""

from rag_pipeline import RAGPipeline

def main():
    print("=" * 60)
    print("RAG Pipeline - Simple Example")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = RAGPipeline()
    
    # Try to load existing data
    print("\n2. Attempting to load existing data...")
    if pipeline.load():
        print("✅ Successfully loaded existing data!")
    else:
        print("❌ No existing data found.")
        print("\nTo build the index, run:")
        print("  python rag_pipeline.py")
        print("\nOr manually:")
        print("  pipeline.download_papers('machine learning', max_results=10)")
        print("  pipeline.process_all_papers()")
        print("  pipeline.build_index()")
        print("  pipeline.save()")
        return
    
    # Test search
    print("\n3. Testing search functionality...")
    
    test_queries = [
        "machine learning algorithms",
        "neural network architecture",
        "deep learning optimization"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        try:
            results = pipeline.search(query, k=2)
            
            for result in results:
                print(f"\n📄 Rank {result['rank']} (Distance: {result['distance']:.4f})")
                print(f"   Paper ID: {result['metadata']['paper_id']}")
                print(f"   Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}")
                print(f"   Preview: {result['chunk'][:150]}...")
                
        except Exception as e:
            print(f"Error searching: {e}")
    
    # Show statistics
    print(f"\n{'='*60}")
    print("Index Statistics")
    print('='*60)
    print(f"Total chunks: {len(pipeline.chunks)}")
    print(f"Index size: {pipeline.index.ntotal if pipeline.index else 0}")
    
    print("\n✅ Example complete!")
    print("\nTo start the API server:")
    print("  uvicorn rag_pipeline:app --reload")


if __name__ == "__main__":
    main()