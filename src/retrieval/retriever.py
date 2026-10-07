from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.indexer.bm25 import BM25Index
from src.retrieval.context import AssembledContext, assemble_context


class Retriever:

  def __init__(self, index: BM25Index):
    self.index = index

  @classmethod
  def from_saved_index(cls, index_path: str | Path) -> "Retriever":
    """Loads a persisted index file and initializes the Retriever."""
    index = BM25Index.load(index_path)
    return cls(index=index)

  def search_raw(
      self, query: str, top_k: int = 10
  ) -> List[Tuple[float, Dict[str, Any]]]:
    """Returns raw scored chunks from the BM25 index."""
    return self.index.search(query=query, top_k=top_k)

  def retrieve_context(
      self,
      query: str,
      top_k: int = 10,
      max_tokens: int = 1500,
      min_score: float = 0.1,
  ) -> AssembledContext:
    """End-to-end retrieval: searches the index, filters, and formats

    the context block to fit within the given token budget.
    """
    raw_results = self.search_raw(query=query, top_k=top_k)
    return assemble_context(
        ranked_results=raw_results,
        max_tokens=max_tokens,
        min_score=min_score,
    )
