"""
rag/retriever.py — Семантический ретривер поверх VectorStore.

Возможности:
  - Поиск top-K релевантных чанков по запросу
  - Конвертация L2-дистанции FAISS → оценка сходства (0..1)
  - Фильтрация по минимальному порогу сходства
  - Форматирование контекста для промпта LLM
  - Дедупликация одинаковых фрагментов
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ── Датакласс результата ──────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Один найденный фрагмент с метаданными."""
    text:       str
    source:     str             # полный путь или имя файла
    page:       int | None      # номер страницы (только для PDF)
    score:      float           # сходство 0..1 (1 = идеальное совпадение)
    chunk_idx:  int = 0         # порядковый номер чанка в результатах

    @property
    def source_name(self) -> str:
        """Только имя файла без пути."""
        return Path(self.source).name

    @property
    def score_pct(self) -> str:
        """Сходство в процентах, например '87%'."""
        return f"{self.score:.0%}"

    def short_preview(self, max_chars: int = 200) -> str:
        """Короткий превью текста для Telegram-сообщений."""
        preview = self.text.replace("\n", " ").strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "…"
        return preview

    def __repr__(self) -> str:
        return (
            f"<RetrievedChunk source={self.source_name!r} "
            f"score={self.score:.3f} len={len(self.text)}>"
        )


# ── Ретривер ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Семантический ретривер поверх VectorStore.

    Args:
        store:          Инициализированный VectorStore с FAISS-индексом
        top_k:          Максимальное кол-во возвращаемых чанков
        min_score:      Минимальный порог сходства (0..1). Чанки ниже порога отбрасываются.
        deduplicate:    Удалять ли одинаковые фрагменты (по точному совпадению текста)
    """

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 3,
        min_score: float = 0.0,
        deduplicate: bool = True,
    ):
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.deduplicate = deduplicate

    # ── Основной метод ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Выполняет семантический поиск и возвращает top_k чанков.

        Pipeline:
          1. similarity_search_with_score → список (Document, L2_distance)
          2. L2-дистанция → сходство через 1 / (1 + dist)
          3. Фильтрация по min_score
          4. Опциональная дедупликация
          5. Сортировка по убыванию сходства

        Args:
            query: Текст запроса пользователя

        Returns:
            Список RetrievedChunk, отсортированных по релевантности (лучшие первые)
        """
        if self.store.is_empty:
            logger.warning("[Retriever] Индекс пуст — поиск невозможен.")
            return []

        if not query.strip():
            logger.warning("[Retriever] Пустой запрос.")
            return []

        logger.debug(f"[Retriever] Запрос: {query!r}, top_k={self.top_k}")

        # Шаг 1: поиск в FAISS (возвращает L2-расстояния)
        raw_results = self.store.similarity_search_with_score(query, k=self.top_k)

        if not raw_results:
            logger.info("[Retriever] Ничего не найдено.")
            return []

        # Шаг 2–3: конвертация дистанции → сходство + фильтрация
        chunks: list[RetrievedChunk] = []
        for idx, (doc, distance) in enumerate(raw_results):
            similarity = self._distance_to_similarity(distance)

            if similarity < self.min_score:
                logger.debug(
                    f"[Retriever] Чанк отброшен: score={similarity:.3f} < min_score={self.min_score}"
                )
                continue

            chunk = RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                score=round(similarity, 4),
                chunk_idx=idx,
            )
            chunks.append(chunk)

        # Шаг 4: дедупликация
        if self.deduplicate:
            chunks = self._deduplicate(chunks)

        # Шаг 5: сортировка по убыванию сходства
        chunks.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            f"[Retriever] Найдено {len(chunks)} чанков для запроса: {query[:60]!r}"
        )
        return chunks

    # ── Форматирование контекста ──────────────────────────────────────────────

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Форматирует список чанков в строку контекста для системного промпта LLM.

        Формат каждого блока:
            [Фрагмент N | filename.md | стр. X | сходство: 87%]
            <текст фрагмента>

        Args:
            chunks: Список RetrievedChunk (обычно от retrieve())

        Returns:
            Многострочная строка с разделителями
        """
        if not chunks:
            return "Контекст не найден."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            header_parts = [f"Фрагмент {i}", chunk.source_name]
            if chunk.page:
                header_parts.append(f"стр. {chunk.page}")
            header_parts.append(f"сходство: {chunk.score_pct}")

            header = "[" + " | ".join(header_parts) + "]"
            parts.append(f"{header}\n{chunk.text.strip()}")

        return "\n\n---\n\n".join(parts)

    def format_search_result(self, chunks: list[RetrievedChunk]) -> str:
        """
        Форматирует результаты для отображения в Telegram (/search команда).
        Использует короткий превью текста.

        Args:
            chunks: Список RetrievedChunk

        Returns:
            Строка для Telegram-сообщения (Markdown)
        """
        if not chunks:
            return "😔 Ничего не найдено по вашему запросу."

        lines = [f"🔍 *Найдено фрагментов: {len(chunks)}*\n"]
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.source_name
            if chunk.page:
                source_info += f" (стр. {chunk.page})"

            lines.append(
                f"*{i}. {source_info}* | {chunk.score_pct}\n"
                f"_{chunk.short_preview(200)}_"
            )

        return "\n\n".join(lines)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        Конвертирует L2-расстояние FAISS в оценку сходства [0..1].

        Формула: similarity = 1 / (1 + distance)
        - distance = 0   → similarity = 1.0  (идеальное совпадение)
        - distance = 1   → similarity = 0.5
        - distance → ∞   → similarity → 0.0

        Работает корректно потому что FAISS с normalize_embeddings=True
        возвращает расстояния в диапазоне [0..2].
        """
        return round(1.0 / (1.0 + max(distance, 0.0)), 4)

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Удаляет дублирующиеся фрагменты по точному совпадению текста.
        При дубликатах оставляет чанк с более высоким score.

        Args:
            chunks: Список чанков (могут быть дубликаты)

        Returns:
            Список без дубликатов
        """
        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = chunk.text.strip()
            if key not in seen or chunk.score > seen[key].score:
                seen[key] = chunk
        return list(seen.values())