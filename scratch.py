
from indexer.py_chunker import chunk_python
code_sample = '''"""Docstring."""

def small_func_1():
    return 1

def small_func_2():
    return 2

def oversized_func():
    # Let's create a long block
    msg = "''' + ("X" * 120) + '''"
    return msg
'''

# Use max_chunk_size = 80 so:
# - Docstring and small_func_1 can group together
# - oversized_func is forced to split across boundaries
chunks = chunk_python(code_sample, max_chunk_size=80)

for idx, c in enumerate(chunks):
  s = c["first_character_index"]
  e = c["last_character_index"]
  print(f"Chunk {idx} [{s}:{e}] (len {e - s}):\n{repr(c['text'])}\n")
