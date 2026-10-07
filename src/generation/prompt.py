SYSTEM_PROMPT = """You are a precise codebase question-answering assistant.
Answer the user's question using ONLY the provided code snippets in the CONTEXT section.

Rules:
1. Ground every claim directly in the provided context. If the answer cannot be determined from the snippets, respond: "I cannot find sufficient evidence in the indexed codebase to answer this question."
2. Never invent functions, files, or parameters not present in the snippets.
3. When referencing code, quote the relevant identifier and append a citation in this exact format:
  
   Example: The engine initializes workers in.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_generation_prompt(question: str, context_text: str) -> str:
  """Builds a complete, formatted prompt for the generator model."""
  if not context_text.strip():
    context_text = "No relevant context found."

  user_content = USER_PROMPT_TEMPLATE.format(
      context=context_text, question=question
  )

  return (
      f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
      f"<|im_start|>user\n{user_content}<|im_end|>\n"
      f"<|im_start|>assistant\n"
  )
