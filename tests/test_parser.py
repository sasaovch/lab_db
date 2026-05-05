# tests/test_parser.py
"""
Тесты для модуля парсинга заметок.
"""
import pytest
from pathlib import Path
from ingestion.parser import _strip_markdown, chunk_documents
from langchain_core.documents import Document


def test_strip_markdown_headings():
    md = "# Заголовок 1\n## Заголовок 2\nТекст параграфа"
    result = _strip_markdown(md)
    assert "Заголовок 1" in result
    assert "#" not in result


def test_strip_markdown_bold():
    md = "Это **жирный** текст и __тоже жирный__"
    result = _strip_markdown(md)
    assert "жирный" in result
    assert "**" not in result


def test_strip_markdown_links():
    md = "Посетите [наш сайт](https://example.com)"
    result = _strip_markdown(md)
    assert "наш сайт" in result
    assert "https://example.com" not in result


def test_chunk_documents_basic():
    doc = Document(page_content="Слово " * 200, metadata={"source": "test.md"})
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # небольшой допуск


def test_chunk_documents_preserves_metadata():
    doc = Document(
        page_content="Тестовый документ для проверки метаданных. " * 50,
        metadata={"source": "my_notes.md", "type": "markdown"},
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "my_notes.md"
        assert chunk.metadata["type"] == "markdown"


def test_chunk_empty_document():
    doc = Document(page_content="", metadata={})
    chunks = chunk_documents([doc])
    assert chunks == []