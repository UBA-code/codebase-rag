from pathlib import Path
import fire

from src.indexer.crawler import crawl_and_chunk_repo
from src.indexer.bm25 import BM25Index
from src.retrieval.retriever import Retriever
from src.generation.engine import GeneratorEngine


class RAGCLI:
    """CLI interface for the local codebase RAG engine."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        index_path: str = "data/index/bm25.pkl",
        max_chunk_size: int = 2000,
    ) -> None:
        """Crawl a repo, chunk Python files, fit BM25 index, and save to disk."""
        print(f"--- 1. Crawling repository: {repo_path} ---")
        chunks = crawl_and_chunk_repo(repo_path, max_chunk_size=max_chunk_size)
        print(f"Found {len(chunks)} code chunks.")

        print("--- 2. Fitting BM25 Inverted Index ---")
        index = BM25Index().fit(chunks)
        print(f"Index built: {len(index.idf)} terms, avgdl={index.avgdl:.1f}")

        print(f"--- 3. Persisting index to: {index_path} ---")
        index.save(index_path)
        size_mb = Path(index_path).stat().st_size / (1024 * 1024)
        print(f"Saved successfully ({size_mb:.2f} MB).")

    def search(
        self,
        query: str,
        index_path: str = "data/index/bm25.pkl",
        top_k: int = 5,
    ) -> None:
        """Run a raw BM25 search to inspect retrieved chunks."""
        retriever = Retriever.from_saved_index(index_path)
        results = retriever.search_raw(query, top_k=top_k)

        print(f"\n--- Top {top_k} Results for: '{query}' ---")
        for i, (score, doc) in enumerate(results, start=1):
            print(f"\n[{i}] Score: {score:.4f} | {doc['file']} ({doc['first_character_index']}:{doc['last_character_index']})")
            preview = doc["text"].strip().splitlines()[0][:80]
            print(f"    Preview: {preview}")

    def query(
        self,
        question: str,
        index_path: str = "data/index/bm25.pkl",
        max_tokens: int = 1500,
        model_id: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Retrieve context, run LLM generation, and verify citations."""
        print("1. Loading index...")
        retriever = Retriever.from_saved_index(index_path)

        print("2. Retrieving context under token budget...")
        ctx = retriever.retrieve_context(question, max_tokens=max_tokens)
        print(f"Selected {len(ctx.chunks_used)} chunks ({ctx.total_characters} chars).")

        print(f"3. Initializing generator ({model_id})...")
        generator = GeneratorEngine(model_id=model_id)

        print("4. Generating grounded response...\n")
        result = generator.generate(question=question, assembled_context=ctx)

        print("=" * 20 + " ANSWER " + "=" * 20)
        print(result.answer)
        print("=" * 48)

        print(f"\nTokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
        print(f"Citations Verified: {result.valid_citations_count}/{result.total_citations_count}")

        if result.citations:
            print("\nCitation Details:")
            for c in result.citations:
                status = "VALID" if c.is_valid else "INVALID (Hallucinated)"
                print(f"  [{status}] {c.file} ({c.start}:{c.end})")


def main() -> None:
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()
