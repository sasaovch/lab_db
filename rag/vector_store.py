"""
rag/vector_store.py — Управление FAISS-индексом с Sentence Transformers эмбеддингами.
Префиксы E5 ("query: " / "passage: ") добавляются вручную перед передачей в модель.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class E5Embeddings(HuggingFaceEmbeddings):
    """
    Обёртка над HuggingFaceEmbeddings с поддержкой E5-префиксов.

    Модели семейства E5 (multilingual-e5-base, e5-large и др.) требуют:
      - "query: ..."   при поиске  (embed_query)
      - "passage: ..." при индексировании (embed_documents)

    Без префиксов качество поиска значительно снижается.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Добавляет префикс 'passage: ' перед индексированием."""
        prefixed = [
            t if t.startswith("passage: ") else f"passage: {t}"
            for t in texts
        ]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Добавляет префикс 'query: ' перед поиском."""
        if not text.startswith("query: "):
            text = f"query: {text}"
        return super().embed_query(text)


class VectorStore:
    """
    Обёртка над FAISS + E5Embeddings.
    Поддерживает добавление документов, поиск, сохранение/загрузку индекса.
    """

    def __init__(self, model_name: str, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._faiss_file = self.index_path / "index.faiss"
        self._store: Optional[FAISS] = None

        logger.info(f"[VectorStore] Загружаю модель: {model_name}")
        self.embeddings = E5Embeddings(
            model=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="./models/multilingual-e5-base",        # ← папка кэша для HuggingFace
        )
        logger.info("[VectorStore] Модель загружена.")
        self._load_if_exists()

    # ── Свойства ──────────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return self._store is None or self._store.index.ntotal == 0

    # ── Запись ────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> int:
        """Добавляет документы в индекс и сохраняет на диск."""
        if not docs:
            logger.warning("[VectorStore] add_documents: передан пустой список.")
            return 0

        logger.info(f"[VectorStore] Индексирую {len(docs)} чанков...")

        if self._store is None:
            self._store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._store.add_documents(docs)

        self._save()
        logger.info(
            f"[VectorStore] Добавлено {len(docs)} чанков. "
            f"Итого: {self.get_count()} векторов."
        )
        return len(docs)

    # ── Поиск ─────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """Возвращает top-k похожих документов."""
        if self.is_empty:
            return []
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 3
    ) -> list[tuple[Document, float]]:
        """
        Возвращает top-k документов с L2-расстоянием.
        Меньше расстояние = выше сходство.
        Конвертация: similarity = 1 / (1 + distance)
        """
        if self.is_empty:
            return []
        return self._store.similarity_search_with_score(query, k=k)

    # ── Статистика ────────────────────────────────────────────────────────────

    def get_count(self) -> int:
        """Количество векторов в индексе."""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def get_document_count(self) -> int:
        """Алиас для get_count() — обратная совместимость с handlers.py."""
        return self.get_count()

    def get_sources(self) -> list[str]:
        """Список уникальных файлов-источников в индексе."""
        if self._store is None:
            return []
        sources = set()
        for doc in self._store.docstore._dict.values():
            src = doc.metadata.get("source", "unknown")
            sources.add(Path(src).name)
        return sorted(sources)

    def get_stats(self) -> dict:
        """Статистика индекса."""
        return {
            "total_vectors": self.get_count(),
            "sources":       self.get_sources(),
            "index_path":    str(self.index_path),
            "is_empty":      self.is_empty,
        }

    # ── Управление ────────────────────────────────────────────────────────────

    def clear(self):
        """Удаляет все векторы и файлы индекса."""
        self._store = None
        for f in self.index_path.glob("*"):
            try:
                f.unlink()
            except Exception as e:
                logger.error(f"[VectorStore] Не удалось удалить {f}: {e}")
        logger.info("[VectorStore] Индекс очищен.")

    # ── Приватные методы ──────────────────────────────────────────────────────

    def _save(self):
        if self._store:
            self._store.save_local(str(self.index_path))
            logger.debug(f"[VectorStore] Сохранён: {self.index_path}")

    def _load_if_exists(self):
        if not self._faiss_file.exists():
            logger.info("[VectorStore] Индекс не найден, начинаем с чистого листа.")
            return
        try:
            self._store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                f"[VectorStore] Загружен индекс: {self.get_count()} векторов, "
                f"источники: {self.get_sources()}"
            )
        except Exception as e:
            logger.error(
                f"[VectorStore] Ошибка загрузки индекса: {e}. "
                f"Начинаем с чистого листа."
            )
            self._store = None