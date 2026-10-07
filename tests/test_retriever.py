from pathlib import Path
from src.indexer.bm25 import BM25Index
from src.retrieval.retriever import Retriever


def test_retriever_pipeline(tmp_path: Path):
  chunks = [
      {
          "file": "vllm/core.py",
          "first_character_index": 0,
          "last_character_index": 30,
          "text": "class Scheduler: def step(): pass",
      },
      {
          "file": "vllm/utils.py",
          "first_character_index": 0,
          "last_character_index": 20,
          "text": "def helper(): pass",
      },
  ]

  # 1. Save an index
  index = BM25Index().fit(chunks)
  index_path = tmp_path / "index.pkl"
  index.save(index_path)

  # 2. Load through Retriever facade
  retriever = Retriever.from_saved_index(index_path)

  # 3. Test context retrieval
  ctx = retriever.retrieve_context("Scheduler step", max_tokens=200)

  assert len(ctx.chunks_used) == 1
  assert "vllm/core.py" in ctx.context_text
  assert "Scheduler" in ctx.context_text
