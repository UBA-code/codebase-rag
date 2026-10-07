from src.generation.citations import extract_citations


def test_extract_valid_citation():
  context_chunks = [{
      "file": "vllm/engine.py",
      "first_character_index": 100,
      "last_character_index": 300,
  }]

  text = "The engine starts here [vllm/engine.py (100:300)]."
  citations = extract_citations(text, context_chunks)

  assert len(citations) == 1
  assert citations[0].file == "vllm/engine.py"
  assert citations[0].start == 100
  assert citations[0].end == 300
  assert citations[0].is_valid is True


def test_extract_invalid_citation():
  context_chunks = [{
      "file": "vllm/engine.py",
      "first_character_index": 100,
      "last_character_index": 300,
  }]

  # Model hallucinated offsets 500:600
  text = "The engine starts here [vllm/engine.py (500:600)]."
  citations = extract_citations(text, context_chunks)

  assert len(citations) == 1
  assert citations[0].is_valid is False


def test_extract_multiple_citations():
  context_chunks = [
      {
          "file": "vllm/a.py",
          "first_character_index": 0,
          "last_character_index": 50,
      },
      {
          "file": "vllm/b.py",
          "first_character_index": 10,
          "last_character_index": 80,
      },
  ]

  text = "Class A [vllm/a.py (0:50)] calls function B [vllm/b.py (10:80)]."
  citations = extract_citations(text, context_chunks)

  assert len(citations) == 2
  assert all(c.is_valid for c in citations)
