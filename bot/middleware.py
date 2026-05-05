"""
bot/middleware.py — Middleware для python-telegram-bot v20+.

В PTB v20 нет BaseMiddleware. Вместо него используются:
  1. TypeHandler(Update, callback) — перехватывает все обновления
  2. Декоратор @check_access — проверка доступа для конкретных хэндлеров
  3. RateLimiter — класс с токен-бакетом, используется внутри хэндлеров
  4. UserSyncer — синхронизация с БД, вызывается вручную
  5. TimingContext — замер времени через context manager
"""
from __future__ import annotations

import logging
import time
import functools
from collections import defaultdict
from typing import Callable, Awaitable, Any

from telegram import Update
from telegram.ext import ContextTypes, Application, TypeHandler

logger = logging.getLogger(__name__)


# ── 1. Логирование всех обновлений ───────────────────────────────────────────

async def log_all_updates(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Логирует каждое входящее обновление.
    Подключается через app.add_handler(TypeHandler(Update, log_all_updates), group=-1)
    group=-1 гарантирует выполнение ДО всех остальных хэндлеров.
    """
    user = update.effective_user
    chat = update.effective_chat

    if update.message:
        msg = update.message
        if msg.text:
            action = f"text={msg.text[:60]!r}"
        elif msg.document:
            action = f"document={msg.document.file_name!r}"
        else:
            action = f"content_type={msg.content_type}"
    elif update.callback_query:
        action = f"callback={update.callback_query.data!r}"
    else:
        action = f"update_id={update.update_id}"

    user_str = f"@{user.username}" if (user and user.username) else f"id={user.id if user else '?'}"
    chat_str  = f"chat={chat.id}"  if chat  else ""

    logger.info(f"[IN] {user_str} | {chat_str} | {action}")


# ── 2. Проверка доступа (декоратор) ──────────────────────────────────────────

def restricted(allowed_user_ids: list[int]):
    """
    Декоратор: разрешает вызов хэндлера только пользователям из списка.
    Если список пуст — пропускает всех.

    Использование:
        from bot.middleware import restricted
        from config import config

        @restricted(config.telegram.allowed_users)
        async def cmd_start(update, context):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            if not allowed_user_ids:
                return await func(update, context, *args, **kwargs)

            user = update.effective_user
            if user is None or user.id not in set(allowed_user_ids):
                uid = user.id if user else "unknown"
                logger.warning(f"[Auth] Доступ запрещён: user_id={uid}")
                if update.message:
                    await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
                elif update.callback_query:
                    await update.callback_query.answer("⛔ Нет доступа.", show_alert=True)
                return

            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator


# ── 3. Ограничение частоты запросов ──────────────────────────────────────────

class RateLimiter:
    """
    Token Bucket алгоритм ограничения частоты запросов на пользователя.

    Использование в хэндлере:
        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

        async def handle_message(update, context):
            if not rate_limiter.is_allowed(update.effective_user.id):
                await update.message.reply_text("⏳ Слишком много запросов.")
                return
            # ... основная логика
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[int, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        """Проверяет, не превышен ли лимит для пользователя. Обновляет счётчик."""
        now = time.monotonic()

        # Удаляем устаревшие метки времени
        self._buckets[user_id] = [
            t for t in self._buckets[user_id]
            if now - t < self.window
        ]

        if len(self._buckets[user_id]) >= self.max_requests:
            logger.warning(
                f"[RateLimit] Превышен лимит: user_id={user_id} "
                f"({len(self._buckets[user_id])}/{self.max_requests} за {self.window}с)"
            )
            return False

        self._buckets[user_id].append(now)
        return True

    def remaining(self, user_id: int) -> int:
        """Возвращает оставшееся количество запросов в текущем окне."""
        now = time.monotonic()
        active = [t for t in self._buckets.get(user_id, []) if now - t < self.window]
        return max(0, self.max_requests - len(active))

    def reset(self, user_id: int):
        """Сбрасывает счётчик для пользователя (например, после бана)."""
        self._buckets.pop(user_id, None)


# Глобальный экземпляр — используется в handlers.py
rate_limiter = RateLimiter(max_requests=10, window_seconds=60.0)


# ── 4. Декоратор rate limit ───────────────────────────────────────────────────

def with_rate_limit(limiter: RateLimiter | None = None):
    """
    Декоратор: применяет rate limiting к хэндлеру.

    Использование:
        @with_rate_limit()
        async def handle_message(update, context):
            ...

        # или с кастомным лимитером:
        my_limiter = RateLimiter(max_requests=5, window_seconds=30)
        @with_rate_limit(my_limiter)
        async def cmd_ask(update, context):
            ...
    """
    _limiter = limiter or rate_limiter

    def decorator(func: Callable[..., Awaitable[Any]]):
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user = update.effective_user
            if user and not _limiter.is_allowed(user.id):
                remaining_time = int(_limiter.window)
                if update.message:
                    await update.message.reply_text(
                        f"⏳ Слишком много запросов. "
                        f"Подождите {remaining_time} секунд."
                    )
                elif update.callback_query:
                    await update.callback_query.answer(
                        "⏳ Слишком много запросов.", show_alert=True
                    )
                return
            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator


# ── 5. Замер времени (декоратор) ──────────────────────────────────────────────

def with_timing(slow_threshold_ms: int = 5000):
    """
    Декоратор: замеряет время выполнения хэндлера и логирует медленные запросы.

    Использование:
        @with_timing(slow_threshold_ms=3000)
        async def handle_message(update, context):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            start = time.monotonic()
            try:
                return await func(update, context, *args, **kwargs)
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                user = update.effective_user
                uid = user.id if user else "?"
                fname = func.__name__

                if elapsed_ms > slow_threshold_ms:
                    logger.warning(
                        f"[Timing] МЕДЛЕННЫЙ хэндлер '{fname}': "
                        f"user={uid}, {elapsed_ms}ms (порог {slow_threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"[Timing] '{fname}': user={uid}, {elapsed_ms}ms")
        return wrapper
    return decorator


# ── 6. Синхронизация пользователя с БД ───────────────────────────────────────

async def sync_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Синхронизирует пользователя Telegram с таблицей users в PostgreSQL.
    Вызывать вручную в начале хэндлеров или через TypeHandler.

    Пример вызова в хэндлере:
        await sync_user_to_db(update, context)
    """
    from db.connection import is_db_enabled, get_session, UserRepository

    user = update.effective_user
    if user is None or not is_db_enabled():
        return

    try:
        async with get_session() as session:
            db_user, created = await UserRepository.get_or_create(
                session,
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )
            if created:
                logger.info(
                    f"[UserSync] Новый пользователь: id={user.id} @{user.username}"
                )
            # Сохраняем db_user для использования в хэндлерах
            context.user_data["db_user"] = db_user
    except Exception as exc:
        logger.error(f"[UserSync] Ошибка синхронизации: {exc}")


# ── 7. Составной декоратор ────────────────────────────────────────────────────

def with_middleware(
    allowed_user_ids: list[int] | None = None,
    limiter: RateLimiter | None = None,
    slow_threshold_ms: int = 5000,
):
    """
    Удобный составной декоратор: timing + rate_limit + auth за один вызов.

    Использование:
        from bot.middleware import with_middleware
        from config import config

        @with_middleware(allowed_user_ids=config.telegram.allowed_users)
        async def cmd_ask(update, context):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        # Применяем декораторы в обратном порядке (снаружи внутрь):
        # with_timing → with_rate_limit → restricted → func
        wrapped = func
        if allowed_user_ids is not None:
            wrapped = restricted(allowed_user_ids)(wrapped)
        wrapped = with_rate_limit(limiter)(wrapped)
        wrapped = with_timing(slow_threshold_ms)(wrapped)

        @functools.wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            return await wrapped(update, context, *args, **kwargs)
        return wrapper
    return decorator


# ── 8. Регистрация в Application ──────────────────────────────────────────────

def register_middleware(app: Application, allowed_user_ids: list[int] | None = None):
    """
    Регистрирует TypeHandler для логирования всех обновлений.
    Вызывать в bot/main.py после создания Application.

    Args:
        app:              Экземпляр Application
        allowed_user_ids: Список разрешённых user_id (не используется здесь,
                          передаётся напрямую в декораторы хэндлеров)

    Использование в main.py:
        from bot.middleware import register_middleware
        register_middleware(app, config.telegram.allowed_users)
    """
    # group=-1 — выполняется раньше всех остальных хэндлеров
    app.add_handler(TypeHandler(Update, log_all_updates), group=-1)
    logger.info("[Middleware] Логирование всех обновлений подключено (group=-1).")