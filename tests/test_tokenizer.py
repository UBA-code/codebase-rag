from src.indexer.tokenizer import tokenize


def test_tokenize_plain_text():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_snake_case():
    assert tokenize("max_model_len = 4096") == ["max", "model", "len", "4096"]


def test_tokenize_camel_and_pascal_case():
    tokens = tokenize("class LLMEngine:")
    # Contains the full term and the sub-components
    assert "llmengine" in tokens
    assert "llm" in tokens
    assert "engine" in tokens


def test_tokenize_empty():
    assert tokenize("") == []
