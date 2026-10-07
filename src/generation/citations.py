from dataclasses import dataclass
import re
from typing import Any, Dict, List

# Matches: [file/path.py (start:end)]
CITATION_REGEX = re.compile(r"\[([^\]]+?)\s*\((\d+):(\d+)\)\]")


@dataclass
class Citation:
  raw: str
  file: str
  start: int
  end: int
  is_valid: bool  # True if it matches one of the chunks given in the context


def extract_citations(
    text: str, context_chunks: List[Dict[str, Any]]
) -> List[Citation]:
  """Extracts citations from text and validates whether they match retrieved chunks."""
  citations: List[Citation] = []
  matches = CITATION_REGEX.findall(text)

  for raw_match in CITATION_REGEX.finditer(text):
    file_path = raw_match.group(1).strip()
    start = int(raw_match.group(2))
    end = int(raw_match.group(3))

    # Verify if this matches any chunk provided to the model
    valid = any(
        c.get("file") == file_path
        and c.get("first_character_index") == start
        and c.get("last_character_index") == end
        for c in context_chunks
    )

    citations.append(
        Citation(
            raw=raw_match.group(0),
            file=file_path,
            start=start,
            end=end,
            is_valid=valid,
        )
    )

  return citations
