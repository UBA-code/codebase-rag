import os
from pathlib import Path
from typing import Any, Dict, List

from src.indexer.md_chunker import chunk_markdown
from src.indexer.py_chunker import chunk_python

# Directories to ignore completely during traversal
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".egg-info",
    ".pytest_cache",
    ".mypy_cache",
}


def crawl_and_chunk_repo(
    repo_root: str | Path,
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    root = Path(repo_root).resolve()
    all_chunks: List[Dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune ignored directories in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for fname in filenames:
            file_path = Path(dirpath) / fname
            suffix = file_path.suffix.lower()

            if suffix not in (".py", ".md"):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Forward-slash relative path
            rel_path = file_path.relative_to(root).as_posix()

            if suffix == ".py":
                chunks = chunk_python(content, max_chunk_size=max_chunk_size)
            else:
                chunks = chunk_markdown(content, max_chunk_size=max_chunk_size)

            for c in chunks:
                c["file"] = rel_path
                all_chunks.append(c)

    return all_chunks
