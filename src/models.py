"""Data models for RAG pipeline evaluation and exchange."""

import uuid
from typing import List, Union
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a specific character span in a file."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """An unanswered question query."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Ground truth question with references and target answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Wrapper for input question sets."""
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Ranked retrieved sources for a single question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results augmented with generated answer text."""
    answer: str


class StudentSearchResults(BaseModel):
    """Output contract for search_dataset command."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output contract for answer_dataset command."""
    search_results: List[MinimalAnswer]
    k: int
