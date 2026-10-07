import re

# Regex to split camelCase/PascalCase boundaries (e.g., LLMEngine -> LLM Engine, getBatchSize -> get Batch Size)
CAMEL_SPLIT_REGEX = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
# Matches sequences of alphanumeric characters
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenizes code and natural language into normalized lowercase terms.

    Splits snake_case, camelCase, PascalCase, dots, and symbols.
    Preserves original compound tokens alongside split sub-tokens.
    """
    if not text:
        return []

    tokens: list[str] = []

    # 1. Match alphanumeric clusters
    matches = TOKEN_PATTERN.findall(text)

    for word in matches:
        lower_word = word.lower()
        tokens.append(lower_word)

        # 2. If it has camelCase or PascalCase, split and index the sub-parts
        parts = CAMEL_SPLIT_REGEX.split(word)
        if len(parts) > 1:
            for p in parts:
                p_lower = p.lower()
                if p_lower and p_lower != lower_word:
                    tokens.append(p_lower)

    return tokens
