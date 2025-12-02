"""
Diagnostic Evaluation Script - Tests if JSON creation works
"""

from hybrid_rag import HybridRAGPipeline
import json
import os

print("="*60)
print("DIAGNOSTIC EVALUATION TEST")
print("="*60)

# Initialize pipeline
print("\n1. Loading pipeline...")
pipeline = HybridRAGPipeline()

if not pipeline.load():
    print("ERROR: No processed data found. Run hybrid_rag.py first!")
    exit(1)

print(f"✅ Loaded {len(pipeline.chunks)} chunks")

# Simple test queries
test_queries = [
    {
        "query": "machine learning",
        "relevant_chunks": [0, 1, 2]  # Simple test with first 3 chunks
    },
    {
        "query": "neural network",
        "relevant_chunks": [3, 4, 5]
    },
    {
        "query": "deep learning",
        "relevant_chunks": [6, 7, 8]
    }
]

print(f"\n2. Running evaluation with {len(test_queries)} test queries...")

# Evaluate
try:
    results = pipeline.evaluate_retrieval(test_queries, k=3)
    print("✅ Evaluation completed successfully")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    exit(1)

# Check results
print(f"\n3. Results summary:")
for method, data in results.items():
    print(f"   {method}: {data['average_recall']:.2%}")

# Save to JSON
output_file = "evaluation_results.json"
print(f"\n4. Attempting to save to: {output_file}")
print(f"   Current directory: {os.getcwd()}")

try:
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ File written successfully!")
except Exception as e:
    print(f"❌ Failed to write file: {e}")
    exit(1)

# Verify file exists
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"✅ File exists! Size: {file_size} bytes")
    print(f"   Full path: {os.path.abspath(output_file)}")
    
    # Show first few lines
    print(f"\n5. File contents preview:")
    with open(output_file, 'r') as f:
        lines = f.readlines()[:10]
        for line in lines:
            print(f"   {line.rstrip()}")
else:
    print(f"❌ File does not exist after writing!")

print("\n" + "="*60)
print("✅ DIAGNOSTIC TEST COMPLETE")
print("="*60)
