from src.indexer.md_chunker import chunk_markdown


def test_chunk_markdown_basic():
    doc = "# Heading\n\nFirst para.\n\nSecond para."
    chunks = chunk_markdown(doc, max_chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0]["text"] == doc
    assert chunks[0]["first_character_index"] == 0
    assert chunks[0]["last_character_index"] == len(doc)


def test_chunk_markdown_oversized_split():
    doc = "A" * 250
    chunks = chunk_markdown(doc, max_chunk_size=100)
    assert len(chunks) == 3
    assert chunks[0]["last_character_index"] - chunks[0]["first_character_index"] == 100
    assert chunks[1]["last_character_index"] - chunks[1]["first_character_index"] == 100
    assert chunks[2]["last_character_index"] - chunks[2]["first_character_index"] == 50
    # Confirm offsets map directly to the text
    for c in chunks:
        assert doc[c["first_character_index"]:c["last_character_index"]] == c["text"]


def test_chunk_markdown_empty():
    assert chunk_markdown("") == []
