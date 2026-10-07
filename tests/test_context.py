from src.retrieval.context import assemble_context


def test_assemble_context_basic():
  chunks = [
      (
          10.5,
          {
              "file": "foo.py",
              "first_character_index": 0,
              "last_character_index": 20,
              "text": "def foo(): return 1",
          },
      ),
      (
          8.2,
          {
              "file": "bar.py",
              "first_character_index": 10,
              "last_character_index": 35,
              "text": "def bar(): return 2",
          },
      ),
  ]

  res = assemble_context(chunks, max_tokens=1000)

  assert len(res.chunks_used) == 2
  assert "[START CONTEXT CHUNK 1]" in res.context_text
  assert "File: foo.py" in res.context_text
  assert "Offsets: 0:20" in res.context_text
  assert "[START CONTEXT CHUNK 2]" in res.context_text
  assert "File: bar.py" in res.context_text


def test_assemble_context_budget_enforcement():
  # Create a large chunk that takes ~100 tokens
  big_text = "x" * 400
  chunks = [
      (
          10.0,
          {
              "file": "one.py",
              "first_character_index": 0,
              "last_character_index": 400,
              "text": big_text,
          },
      ),
      (
          9.0,
          {
              "file": "two.py",
              "first_character_index": 0,
              "last_character_index": 400,
              "text": big_text,
          },
      ),
  ]

  # Budget allows roughly ~125 tokens (500 chars) -> should only fit 1 block
  res = assemble_context(chunks, max_tokens=125)

  assert len(res.chunks_used) == 1
  assert "File: one.py" in res.context_text
  assert "File: two.py" not in res.context_text


def test_assemble_context_skips_low_scores():
  chunks = [(
      0.05,
      {
          "file": "low.py",
          "first_character_index": 0,
          "last_character_index": 10,
          "text": "pass",
      },
  )]
  res = assemble_context(chunks, min_score=0.1)
  assert len(res.chunks_used) == 0
  assert res.context_text == ""
