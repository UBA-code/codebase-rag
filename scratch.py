import re
from typing import List, Tuple

def find_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all non-empty paragraphs in text and return their:
    (start_index, end_index, text)
    """
    results: List[Tuple[int, int, str]] = []

    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
      results.append((match.start(), match.end(), match.group()))
    return results


# Test it with this sample:
sample = "Intro to vLLM.\n\nHere is how to configure it."
results = find_paragraphs(sample)
print(results)
