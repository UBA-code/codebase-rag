from indexer.md_chunker import split_oversized_paragraph
from indexer.md_chunker import group_paragraphs_into_chunks
from indexer.md_chunker import chunk_markdown
from typing import Any, Dict, List, Tuple
import ast

def compute_line_starts(source_code: str) -> List[int]:
    """
    Returns a list where line_starts[lineno] is the character index
    where that line starts in source_code.
    1-indexed: line_starts[1] is the start of line 1.
    """

    line_starts = [0, 0]
    
    current_char_count = 0

    for line in source_code.splitlines(keepends=True):
      current_char_count += len(line)
      line_starts.append(current_char_count)

    return line_starts

def extract_ast_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parses code with AST and returns a list of (start, end, text)
    tuples for every top-level node in tree.body.
    """
    line_starts = compute_line_starts(code)
    tree = ast.parse(code)
    blocks: List[Tuple[int, int, str]] = []

    for node in tree.body:
        start_char = line_starts[node.lineno] + node.col_offset
        end_char = line_starts[node.end_lineno] + node.end_col_offset
        blocks.append((start_char, end_char, code[start_char:end_char]))

    return blocks


def chunk_python(
    code: str,
    max_chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    """
    Chunks Python code by AST nodes, splitting oversized definitions
    and grouping small ones up to max_chunk_size.
    Falls back to chunk_markdown on SyntaxError.
    """
    if not code.strip():
        return []

    try:
        raw_blocks = extract_ast_blocks(code)
        chunks: List[Dict[str, Any]] = []

        for block in raw_blocks:
          chunks.extend(split_oversized_paragraph(block, max_chunk_size))

        return group_paragraphs_into_chunks(code, chunks, max_chunk_size)
    except SyntaxError:
        return chunk_markdown(code, max_chunk_size=max_chunk_size)
