from indexer.py_chunker import chunk_python


def test_chunk_python_functions_and_classes():
    code = (
        '"""Module."""\n\n'
        "def foo():\n"
        "    return 42\n\n"
        "class Bar:\n"
        "    pass\n"
    )
    chunks = chunk_python(code, max_chunk_size=2000)
    assert len(chunks) == 1
    assert chunks[0]["text"] == code.strip()
    assert (
        code[chunks[0]["first_character_index"] : chunks[0]["last_character_index"]]
        == chunks[0]["text"]
    )


def test_chunk_python_oversized():
    code = "def big():\n    x = '" + ("A" * 300) + "'\n    return x\n"
    chunks = chunk_python(code, max_chunk_size=100)
    assert len(chunks) > 1
    for c in chunks:
        length = c["last_character_index"] - c["first_character_index"]
        assert length <= 100
        assert code[c["first_character_index"] : c["last_character_index"]] == c["text"]


def test_chunk_python_syntax_error_fallback():
    broken_code = "def broken(:\n    print('bad syntax')\n\nvalid = 10\n"
    chunks = chunk_python(broken_code, max_chunk_size=100)
    assert len(chunks) >= 1
    for c in chunks:
        assert (
            broken_code[c["first_character_index"] : c["last_character_index"]]
            == c["text"]
        )


def test_chunk_python_empty():
    assert chunk_python("") == []
