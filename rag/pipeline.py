# rag/pipeline.py
"""
RAG-пайплайн: связывает ретривер и генератор.
"""
from __future__ import annotations
from rag.retriever import Retriever
from rag.generator import Generator


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    async def query(self, user_question: str) -> tuple[str, list]:
        """
        Основной метод: вопрос → ответ.
        Возвращает (ответ, список найденных чанков).
        """
        chunks = self.retriever.retrieve(user_question)
        answer = await self.generator.generate(user_question, chunks)
        return answer, chunks