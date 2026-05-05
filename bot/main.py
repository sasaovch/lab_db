# bot/main.py
"""
Точка входа. Запуск бота в режиме polling или webhook.
"""
from __future__ import annotations

import logging
import os
import sys

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator
from rag.pipeline import RAGPipeline
from bot.handlers import (
    cmd_start, cmd_help, cmd_stats, cmd_clear, cmd_search,
    cmd_ask, handle_message, handle_document, callback_clear,
)
from bot.middleware import register_middleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Application:
    config.validate()

    # Инициализируем компоненты RAG
    vector_store = VectorStore(
        model_name=config.embedding.model_name,
        index_path=config.storage.faiss_index_path,
    )
    retriever = Retriever(store=vector_store, top_k=config.rag.top_k)
    generator = Generator(model=config.ollama.model,base_url=config.ollama.base_url)
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    # Создаём Application
    app = (
        Application.builder()
        .token(config.telegram.token)
        .build()
    )

    # Подключаем middleware (логирование всех обновлений)
    register_middleware(app, config.telegram.allowed_users)

    # Передаём зависимости через bot_data
    app.bot_data["vector_store"] = vector_store
    app.bot_data["retriever"] = retriever
    app.bot_data["pipeline"] = pipeline

    # Регистрируем хэндлеры
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CallbackQueryHandler(callback_clear, pattern="^clear_"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def main():
    app = create_app()

    # # После создания app:
    # for mw in build_middleware_stack(
    #     allowed_user_ids=config.telegram.allowed_users,
    #     rate_limit_requests=10,
    #     rate_limit_window=60.0,
    #     enable_db_sync=config.database.enabled,
    # ):
    #     app.add_middleware(mw)

    if config.telegram.webhook_url:
        # ── Webhook режим (продакшн) ──────────────────────────────────────────
        logger.info(f"Запуск в режиме webhook: {config.telegram.webhook_url}")
        app.run_webhook(
            listen="0.0.0.0",
            port=config.telegram.webhook_port,
            url_path=config.telegram.token,
            webhook_url=f"{config.telegram.webhook_url}/{config.telegram.token}",
            drop_pending_updates=True,
        )
    else:
        # ── Polling режим (разработка) ────────────────────────────────────────
        logger.info("Запуск в режиме polling (разработка)")
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )


if __name__ == "__main__":
    main()