# tests/test_retriever.py
"""
Тесты для ретривера с mock-хранилищем.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from rag.retriever import Retriever, RetrievedChunk


def make_mock_store(results):
    """Создаёт mock VectorStore с заданными результатами."""
    store = MagicMock()
    store.is_empty = False
    store.similarity_search_with_score.return_value = results
    return store


def test_retriever_returns_chunks():
    doc = Document(
        page_content="Трансформеры — архитектура нейросетей",
        metadata={"source": "ml_notes.md"}
    )
    store = make_mock_store([(doc, 0.2)])
    retriever = Retriever(store=store, top_k=3)
    chunks = retriever.retrieve("что такое трансформеры")
    assert len(chunks) == 1
    assert chunks[0].text == "Трансформеры — архитектура нейросетей"
    assert chunks[0].source == "ml_notes.md"
    assert 0 < chunks[0].score <= 1.0


def test_retriever_empty_store():
    store = MagicMock()
    store.similarity_search_with_score.return_value = []
    retriever = Retriever(store=store, top_k=3)
    chunks = retriever.retrieve("любой запрос")
    assert chunks == []


def test_retriever_sorted_by_score():
    docs_scores = [
        (Document(page_content="B", metadata={"source": "b.md"}), 0.5),
        (Document(page_content="A", metadata={"source": "a.md"}), 0.1),
        (Document(page_content="C", metadata={"source": "c.md"}), 0.9),
    ]
    store = make_mock_store(docs_scores)
    retriever = Retriever(store=store, top_k=3)
    chunks = retriever.retrieve("запрос")
    # Должны быть отсортированы по убыванию сходства
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_format_context():
    chunks = [
        RetrievedChunk(text="Текст фрагмента", source="notes/my_file.md", page=None, score=0.87),
    ]
    retriever = Retriever(store=MagicMock(), top_k=3)
    context = retriever.format_context(chunks)
    assert "my_file.md" in context
    assert "87%" in context
    assert "Текст фрагмента" in context