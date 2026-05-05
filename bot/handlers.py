# bot/handlers.py
"""
Обработчики команд и сообщений Telegram-бота.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from config import config
from ingestion.parser import parse_file, chunk_documents


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 *Привет! Я бот для семантического поиска по вашим заметкам.*\n\n"
        "📌 *Доступные команды:*\n"
        "• /upload — загрузить заметки (MD, PDF, TXT)\n"
        "• /search `<запрос>` — найти релевантные фрагменты\n"
        "• /ask `<вопрос>` — задать вопрос и получить ответ\n"
        "• /stats — статистика индекса\n"
        "• /clear — очистить индекс\n"
        "• /help — справка\n\n"
        "💡 Или просто напишите вопрос — я отвечу на основе ваших заметок!"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── /help ─────────────────────────────────────────────────────────────────────

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *Справка по использованию бота*\n\n"
        "*Загрузка заметок:*\n"
        "Отправьте файл (.md, .pdf, .txt) боту напрямую.\n"
        "Или используйте /upload и прикрепите файл.\n\n"
        "*Семантический поиск:*\n"
        "`/search машинное обучение` — найдёт 3 самых релевантных фрагмента без генерации.\n\n"
        "*Вопрос-ответ (RAG):*\n"
        "`/ask Что такое трансформеры?` — найдёт контекст и сгенерирует ответ через GPT.\n\n"
        "Или просто напишите вопрос в чат — бот автоматически запустит RAG-пайплайн.\n\n"
        "*Форматы файлов:*\n"
        "• `.md` / `.markdown` — заметки Obsidian, Notion, Typora\n"
        "• `.pdf` — статьи, книги\n"
        "• `.txt` — обычный текст\n"
        "📌 *Доступные команды:*\n"
        "• /upload — загрузить заметки (MD, PDF, TXT)\n"
        "• /search `<запрос>` — найти релевантные фрагменты\n"
        "• /ask `<вопрос>` — задать вопрос и получить ответ\n"
        "• /stats — статистика индекса\n"
        "• /clear — очистить индекс\n"
        "• /help — справка\n\n"
        "💡 Или просто напишите вопрос — я отвечу на основе ваших заметок!"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── /stats ────────────────────────────────────────────────────────────────────

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    store = ctx.bot_data["vector_store"]
    count = store.get_document_count()
    if count == 0:
        text = "📊 Индекс пуст. Загрузите заметки командой /upload или отправив файл."
    else:
        text = (
            f"📊 *Статистика индекса*\n\n"
            f"• Векторов в индексе: *{count}*\n"
            f"• Модель эмбеддингов: `{config.ollama.model}`\n"
            f"• Топ-K при поиске: *{config.rag.top_k}*\n"
            f"• Размер чанка: *{config.rag.chunk_size}* символов\n"
            f"• Перекрытие чанков: *{config.rag.chunk_overlap}* символов\n"
        )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── /clear ────────────────────────────────────────────────────────────────────

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, очистить", callback_data="clear_confirm"),
            InlineKeyboardButton("❌ Отмена", callback_data="clear_cancel"),
        ]
    ])
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите *очистить весь индекс*? "
        "Это удалит все загруженные заметки.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
    )


async def callback_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "clear_confirm":
        ctx.bot_data["vector_store"].clear()
        await query.edit_message_text("✅ Индекс очищен. Загрузите новые заметки.")
    else:
        await query.edit_message_text("❌ Очистка отменена.")


# ── /search ───────────────────────────────────────────────────────────────────

async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "❗ Укажите запрос: `/search ваш запрос`", parse_mode=ParseMode.MARKDOWN
        )
        return

    query_text = " ".join(args)
    store = ctx.bot_data["vector_store"]

    if store.is_empty:
        await update.message.reply_text(
            "📭 Индекс пуст. Загрузите заметки перед поиском."
        )
        return

    msg = await update.message.reply_text("🔍 Ищу релевантные фрагменты...")
    retriever = ctx.bot_data["retriever"]
    chunks = retriever.retrieve(query_text)

    if not chunks:
        await msg.edit_text("😔 Ничего не найдено по вашему запросу.")
        return

    lines = [f"🔍 *Результаты поиска:* `{query_text}`\n"]
    for i, chunk in enumerate(chunks, 1):
        source = chunk.source.split("/")[-1]
        preview = chunk.text[:200].replace("\n", " ") + ("…" if len(chunk.text) > 200 else "")
        lines.append(
            f"*{i}. {source}* | Сходство: {chunk.score:.0%}\n"
            f"_{preview}_"
        )
    await msg.edit_text("\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ── /ask ──────────────────────────────────────────────────────────────────────

async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "❗ Укажите вопрос: `/ask ваш вопрос`", parse_mode=ParseMode.MARKDOWN
        )
        return
    question = " ".join(args)
    await _run_rag(update, ctx, question)


# ── Обработчик текстовых сообщений (авто-RAG) ─────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Любое текстовое сообщение → RAG-запрос."""
    await _run_rag(update, ctx, update.message.text)


async def _run_rag(update: Update, ctx: ContextTypes.DEFAULT_TYPE, question: str):
    store = ctx.bot_data["vector_store"]
    if store.is_empty:
        await update.message.reply_text(
            "📭 Индекс пуст. Загрузите заметки командой /upload или отправив файл."
        )
        return

    msg = await update.message.reply_text("⏳ Ищу в заметках и генерирую ответ…")
    pipeline = ctx.bot_data["pipeline"]

    try:
        answer, chunks = await pipeline.query(question)
        # Формируем финальный ответ
        sources = list({c.source.split("/")[-1] for c in chunks})
        footer = f"\n\n📎 *Источники:* {', '.join(f'`{s}`' for s in sources)}"
        full_answer = answer + footer

        # Telegram ограничивает сообщения 4096 символами
        if len(full_answer) > 4096:
            full_answer = full_answer[:4090] + "…"

        await msg.edit_text(full_answer, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await msg.edit_text(f"⚠️ Ошибка: {str(e)}")


# ── Обработчик загрузки файлов ────────────────────────────────────────────────

async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Пользователь отправил файл → парсим и индексируем."""
    document = update.message.document
    if not document:
        return

    file_name = document.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()
    allowed = {".md", ".markdown", ".pdf", ".txt"}

    if suffix not in allowed:
        await update.message.reply_text(
            f"❌ Формат `{suffix}` не поддерживается.\n"
            f"Поддерживаемые форматы: MD, PDF, TXT",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    msg = await update.message.reply_text(f"⬇️ Загружаю `{file_name}`…", parse_mode=ParseMode.MARKDOWN)

    try:
        # Скачиваем файл во временную директорию
        tg_file = await ctx.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await tg_file.download_to_drive(str(tmp_path))

        await msg.edit_text(f"🔄 Парсю и индексирую `{file_name}`…", parse_mode=ParseMode.MARKDOWN)

        # Парсим и чанкуем
        docs = parse_file(tmp_path)
        chunks = chunk_documents(docs, config.rag.chunk_size, config.rag.chunk_overlap)

        # Переименовываем источник в оригинальное имя файла
        for chunk in chunks:
            chunk.metadata["source"] = file_name

        # Добавляем в индекс
        store = ctx.bot_data["vector_store"]
        added = store.add_documents(chunks)

        tmp_path.unlink(missing_ok=True)

        await msg.edit_text(
            f"✅ *`{file_name}`* успешно проиндексирован!\n\n"
            f"• Чанков добавлено: *{added}*\n"
            f"• Всего в индексе: *{store.get_document_count()}* векторов\n\n"
            f"Теперь задавайте вопросы! 💬",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        print(e)
        await msg.edit_text(f"⚠️ Ошибка при обработке файла: {str(e)}")