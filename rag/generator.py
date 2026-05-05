from __future__ import annotations
from typing import List
import httpx

from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """Ты — интеллектуальный ассистент по личным заметкам пользователя.
Отвечай только на основе переданных фрагментов заметок.
Если информации недостаточно, честно скажи об этом.
Отвечай на языке вопроса.
"""

class Generator:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "В ваших заметках не найдено релевантной информации по этому вопросу."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_name = chunk.source.split("/")[-1]
            context_parts.append(
                f"Фрагмент {i} (файл: {source_name}, сходство: {chunk.score:.0%}):\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Контекст из заметок:\n{context}\n\n"
            f"Вопрос: {query}\n\n"
            f"Ответ:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024
            }
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
        except Exception as e:
            return f"Ошибка генерации через Ollama: {e}"