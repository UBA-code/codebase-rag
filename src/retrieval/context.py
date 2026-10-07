from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class AssembledContext:
  context_text: str
  chunks_used: List[Dict[str, Any]]
  total_characters: int


def estimate_tokens(text: str) -> int:
  """Rough heuristic: ~4 characters per token for English and code."""
  return len(text) // 4


def format_chunk_block(doc: Dict[str, Any], index: int) -> str:
  """Formats a single chunk with explicit file and coordinate metadata."""
  file_path = doc.get("file", "unknown")
  start = doc.get("first_character_index", 0)
  end = doc.get("last_character_index", 0)
  content = doc.get("text", "").strip()

  return (
      f"[START CONTEXT CHUNK {index}]\n"
      f"File: {file_path}\n"
      f"Offsets: {start}:{end}\n"
      f"---\n"
      f"{content}\n"
      f"[END CONTEXT CHUNK {index}]"
  )


def assemble_context(
    ranked_results: List[Tuple[float, Dict[str, Any]]],
    max_tokens: int = 1500,
    min_score: float = 0.1,
) -> AssembledContext:
  """Greedily selects top-ranked chunks within a token budget and formats them.

  Args:
      ranked_results: List of (score, doc_dict) tuples sorted descending.
      max_tokens: Hard ceiling for total tokens in the assembled context.
      min_score: Minimum BM25 score required to be considered.

  Returns:
      AssembledContext containing the combined text and chosen chunk metadata.
  """
  max_characters = max_tokens * 4
  formatted_blocks: List[str] = []
  used_chunks: List[Dict[str, Any]] = []
  current_char_count = 0

  chunk_index = 1
  for score, doc in ranked_results:
    if score < min_score:
      continue

    block_str = format_chunk_block(doc, chunk_index)
    block_len = len(block_str)

    # Check if adding this block exceeds the budget
    separator_len = 2 if formatted_blocks else 0
    if current_char_count + block_len + separator_len > max_characters:
      # If this is the very first chunk and it's oversized, include it truncated
      if not formatted_blocks:
        formatted_blocks.append(block_str[:max_characters])
        used_chunks.append(doc)
        current_char_count = max_characters
      break

    formatted_blocks.append(block_str)
    used_chunks.append(doc)
    current_char_count += block_len + separator_len
    chunk_index += 1

  return AssembledContext(
      context_text="\n\n".join(formatted_blocks),
      chunks_used=used_chunks,
      total_characters=current_char_count,
  )
