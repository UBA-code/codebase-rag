from pathlib import Path
from src.indexer.bm25 import BM25Index


def test_bm25_ranking():
  chunks = [
      {
          "file": "vllm/engine/llm_engine.py",
          "text": "class LLMEngine: pass",
          "first_character_index": 0,
          "last_character_index": 21,
      },
      {
          "file": "vllm/config.py",
          "text": "class ModelConfig: max_model_len = 4096",
          "first_character_index": 0,
          "last_character_index": 39,
      },
  ]

  index = BM25Index().fit(chunks)
  results = index.search("LLMEngine", top_k=1)

  assert len(results) == 1
  score, doc = results[0]
  assert score > 0
  assert doc["file"] == "vllm/engine/llm_engine.py"


def test_bm25_save_and_load(tmp_path: Path):
  chunks = [{
      "file": "foo.py",
      "text": "def compute(): return 42",
      "first_character_index": 0,
      "last_character_index": 24,
  }]

  index = BM25Index().fit(chunks)
  index_file = tmp_path / "index.pkl"
  index.save(index_file)

  assert index_file.exists()

  loaded_index = BM25Index.load(index_file)
  results = loaded_index.search("compute", top_k=1)

  assert len(results) == 1
  assert results[0][1]["file"] == "foo.py"


def test_bm25_empty():
  index = BM25Index().fit([])
  assert index.search("anything") == []
