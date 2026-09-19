import re
from typing import List, Tuple, Dict, Any


def find_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all non-empty paragraphs in text and return their:
    (start_index, end_index, text)
    """
    results: List[Tuple[int, int, str]] = []

    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        results.append((match.start(), match.end(), match.group()))
    return results


def group_paragraphs_into_chunks(
    full_text: str, paragraphs: List[Tuple[int, int, str]], max_chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    """
    Groups adjacent paragraphs into chunks without exceeding max_chunk_size.
    Returns a list of dicts with:
      'first_character_index', 'last_character_index', and 'text'
    """
    if not paragraphs:
        return []

    chunks: List[Dict[str, Any]] = []

    # Track the current chunk's boundaries
    current_start = paragraphs[0][0]
    current_end = paragraphs[0][1]

    for start, end, text in paragraphs[1:]:
        if end - current_start <= max_chunk_size:
            current_end = end
        else:
            chunks.append(
                {
                    "first_character_index": current_start,
                    "last_character_index": current_end,
                    "text": full_text[current_start:current_end]
                }
            )
            current_start = start
            current_end = end

    chunks.append(
        {
            "first_character_index": current_start,
            "last_character_index": current_end,
            "text": full_text[current_start:current_end]
        }
    )

    return chunks


def split_oversized_paragraph(
    para: Tuple[int, int, str],
    max_chunk_size: int = 2000
) -> List[Tuple[int, int, str]]:
    """
    If end - start <= max_chunk_size, returns [para].
    Otherwise, slices the paragraph into pieces of at most max_chunk_size.
    """
    start, end, text = para
    if end - start <= max_chunk_size:
        return [para]
        
    chunks: List[Tuple[int, int, str]] = []

    for sub_start in range(start, end, max_chunk_size):
      sub_end = min(sub_start + max_chunk_size, end)
      chunks.append((sub_start, sub_end, text[sub_start - start:sub_end - start]))

    return chunks


def chunk_markdown(
    text: str,
    max_chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    # Step 1: Find all base paragraphs
    raw_paragraphs = find_paragraphs(text)
    
    # Step 2: Normalize so no single paragraph is oversized
    normalized_paragraphs: List[Tuple[int, int, str]] = []
    for para in raw_paragraphs:
      normalized_paragraphs.extend(split_oversized_paragraph(para, max_chunk_size))
    
    # Step 3: Group the normalized paragraphs into final chunks
    return group_paragraphs_into_chunks(text, normalized_paragraphs, max_chunk_size)
