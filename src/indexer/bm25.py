from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

from src.indexer.tokenizer import tokenize


class BM25Index:

  def __init__(self, k1: float = 1.5, b: float = 0.75):
    self.k1 = k1
    self.b = b

    self.docs: List[Dict[str, Any]] = []
    self.doc_lengths: List[int] = []
    self.avgdl: float = 0.0
    self.doc_count: int = 0

    self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    self.idf: Dict[str, float] = {}

  def fit(self, chunks: List[Dict[str, Any]]) -> "BM25Index":
    """Builds the inverted index, calculates doc lengths, and precomputes IDF."""
    self.docs = chunks
    self.doc_count = len(chunks)

    if self.doc_count == 0:
      return self

    total_tokens = 0
    doc_freq: Dict[str, int] = defaultdict(int)

    for doc_id, chunk in enumerate(chunks):
      tokens = tokenize(chunk["text"])
      length = len(tokens)
      self.doc_lengths.append(length)
      total_tokens += length

      term_counts = Counter(tokens)
      for term, count in term_counts.items():
        self.inverted_index[term].append((doc_id, count))
        doc_freq[term] += 1

    self.avgdl = total_tokens / self.doc_count

    for term, freq in doc_freq.items():
      numerator = self.doc_count - freq + 0.5
      denominator = freq + 0.5
      self.idf[term] = math.log((numerator / denominator) + 1.0)

    return self

  def score_document(
      self, query_terms: List[str], doc_id: int, term_frequencies: Dict[str, int]
  ) -> float:
    """Computes BM25 score for a single document against query terms."""
    doc_len = self.doc_lengths[doc_id]
    len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)

    score = 0.0
    for term in query_terms:
      tf = term_frequencies.get(term, 0)
      if tf == 0:
        continue

      idf = self.idf.get(term, 0.0)
      tf_factor = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
      score += idf * tf_factor

    return score

  def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    """Retrieves top-k most relevant chunks using the inverted index."""
    query_terms = tokenize(query)
    if not query_terms or self.doc_count == 0:
      return []

    candidate_tfs: Dict[int, Dict[str, int]] = defaultdict(dict)
    for term in query_terms:
      if term in self.inverted_index:
        for doc_id, tf in self.inverted_index[term]:
          candidate_tfs[doc_id][term] = tf

    if not candidate_tfs:
      return []

    scores: List[Tuple[float, Dict[str, Any]]] = []
    for doc_id, term_tfs in candidate_tfs.items():
      score = self.score_document(query_terms, doc_id, term_tfs)
      scores.append((score, self.docs[doc_id]))

    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[:top_k]

  def save(self, file_path: str | Path) -> None:
    """Serializes the index to disk using pickle."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
      pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

  @classmethod
  def load(cls, file_path: str | Path) -> "BM25Index":
    """Loads a serialized index from disk."""
    with open(file_path, "rb") as f:
      return pickle.load(f)
