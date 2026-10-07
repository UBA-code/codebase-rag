import time
from pathlib import Path
from src.indexer.crawler import crawl_and_chunk_repo
from src.indexer.bm25 import BM25Index

REPO_PATH = "data/raw/vllm-0.10.1"
INDEX_PATH = "data/index/bm25.pkl"

print(f"--- Starting Crawl & Chunk on {REPO_PATH} ---")

t0 = time.perf_counter()
chunks = crawl_and_chunk_repo(REPO_PATH, max_chunk_size=2000)
t_crawl = time.perf_counter() - t0
print(f"Discovered {len(chunks)} chunks in {t_crawl:.2f}s")

print(f"\n--- Building BM25 Inverted Index ---")
t0 = time.perf_counter()
index = BM25Index().fit(chunks)
t_index = time.perf_counter() - t0
print(f"Built index (Vocabulary: {len(index.idf)} terms, avgdl: {index.avgdl:.1f}) in {t_index:.2f}s")

print(f"\n--- Persisting to {INDEX_PATH} ---")
index.save(INDEX_PATH)
size_mb = Path(INDEX_PATH).stat().st_size / (1024 * 1024)
print(f"Saved index file ({size_mb:.2f} MB)")

# Test a realistic vLLM query
print("\n--- Testing Search ---")
results = index.search("LLMEngine decode step", top_k=3)
for i, (score, doc) in enumerate(results, start=1):
    print(f"\n[{i}] Score: {score:.4f} | {doc['file']} ({doc['first_character_index']}:{doc['last_character_index']})")
    first_line = doc['text'].strip().splitlines()[0]
    print(f"    Preview: {first_line[:80]}")
