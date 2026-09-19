from pathlib import Path
from indexer.crawler import crawl_and_chunk_repo


def test_crawl_and_chunk_repo_basic(tmp_path: Path):
    # 1. Setup a mini fake repo structure inside tmp_path
    sub_dir = tmp_path / "pkg"
    sub_dir.mkdir()

    # Create a Python file
    py_file = sub_dir / "app.py"
    py_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    # Create a Markdown file
    md_file = tmp_path / "README.md"
    md_file.write_text("# Title\n\nIntro paragraph.\n", encoding="utf-8")

    # Create a file that should be IGNORED
    ignored_dir = tmp_path / ".git"
    ignored_dir.mkdir()
    (ignored_dir / "config.py").write_text("secret = True\n", encoding="utf-8")

    # Create an unsupported extension that should be IGNORED
    (tmp_path / "notes.txt").write_text("just text", encoding="utf-8")

    # 2. Run crawler
    chunks = crawl_and_chunk_repo(tmp_path)

    # 3. Assertions
    files = {c["file"] for c in chunks}
    assert "pkg/app.py" in files
    assert "README.md" in files
    assert not any(".git" in f for f in files)
    assert not any("notes.txt" in f for f in files)

    # Verify schema of emitted chunks
    for c in chunks:
        assert "file" in c
        assert "first_character_index" in c
        assert "last_character_index" in c
        assert "text" in c
