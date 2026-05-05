"""
ingestion/parser.py — Парсинг заметок из Markdown, PDF и TXT файлов.

Возможности:
  - Парсинг .md / .markdown с очисткой разметки
  - Парсинг .pdf постранично через pypdf
  - Парсинг .txt с определением кодировки
  - Разбивка на перекрывающиеся чанки (RecursiveCharacterTextSplitter)
  - Сохранение метаданных источника в каждом чанке
  - Удаление пустых и слишком коротких чанков
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Минимальная длина чанка — короче отбрасываем как мусор
MIN_CHUNK_LENGTH = 30


# ── Markdown ──────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """
    Удаляет Markdown-разметку и возвращает чистый текст.

    Обрабатывает:
      - Заголовки (# ## ###)
      - Жирный / курсив (** __ * _)
      - Инлайн-код (`code`)
      - Блоки кода (``` ```)
      - Ссылки [текст](url) → текст
      - Изображения ![alt](url) → удаляются
      - Горизонтальные линии (--- ***)
      - Таблицы Markdown → текст строк
      - HTML-теги
      - Лишние пустые строки
    """
    # Блоки кода (``` ... ```) — удаляем целиком
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Инлайн-код
    text = re.sub(r"`[^`]+`", "", text)

    # Изображения — перед ссылками (порядок важен)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Ссылки [текст](url) → текст
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Ссылки-сноски [текст][ref]
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)

    # Заголовки → текст
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Жирный + курсив (*** или ___)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text, flags=re.DOTALL)
    # Жирный (** или __)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text, flags=re.DOTALL)
    # Курсив (* или _)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # Зачёркнутый текст (~~text~~)
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # Горизонтальные линии
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Markdown-таблицы — убираем разделители |---|---|
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    # Строки таблицы | col1 | col2 | → col1  col2
    text = re.sub(r"^\|(.+)\|$", lambda m: m.group(1).replace("|", "  "), text, flags=re.MULTILINE)

    # Маркеры списков (- * +) в начале строки
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Нумерованные списки (1. 2.)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Цитаты Markdown (> ...)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # HTML-теги
    text = re.sub(r"<[^>]+>", "", text)

    # Множественные пустые строки → одна
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_markdown(file_path: Path) -> list[Document]:
    """
    Читает .md / .markdown файл и возвращает список Document.

    Стратегия: читает весь файл, очищает от разметки,
    затем разбивает по заголовкам второго уровня (##) на секции —
    каждая секция становится отдельным Document.

    Args:
        file_path: Путь к .md файлу

    Returns:
        Список Document (одна запись если нет заголовков)
    """
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="cp1251", errors="replace")

    if not raw.strip():
        logger.warning(f"[Parser] Файл пустой: {file_path.name}")
        return []

    # Разбиваем по заголовкам H1/H2 для лучшей гранулярности
    sections = re.split(r"(?=^#{1,2} )", raw, flags=re.MULTILINE)
    docs = []
    for section in sections:
        if not section.strip():
            continue
        # Извлекаем заголовок секции для метаданных
        title_match = re.match(r"^#{1,2}\s+(.+)", section)
        section_title = title_match.group(1).strip() if title_match else file_path.stem

        clean_text = _strip_markdown(section)
        if len(clean_text) < MIN_CHUNK_LENGTH:
            continue

        docs.append(Document(
            page_content=clean_text,
            metadata={
                "source":    str(file_path),
                "type":      "markdown",
                "section":   section_title,
                "file_name": file_path.name,
            },
        ))

    # Если заголовков нет — весь файл как один Document
    if not docs:
        clean_text = _strip_markdown(raw)
        if len(clean_text) >= MIN_CHUNK_LENGTH:
            docs = [Document(
                page_content=clean_text,
                metadata={
                    "source":    str(file_path),
                    "type":      "markdown",
                    "section":   file_path.stem,
                    "file_name": file_path.name,
                },
            )]

    logger.info(f"[Parser] Markdown: {file_path.name} → {len(docs)} секций")
    return docs


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: Path) -> list[Document]:
    """
    Читает .pdf файл и возвращает список Document (по страницам).

    Args:
        file_path: Путь к .pdf файлу

    Returns:
        Список Document — по одному на непустую страницу
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "Установите pypdf для работы с PDF: pip install pypdf"
        )

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        logger.error(f"[Parser] Не удалось открыть PDF {file_path.name}: {e}")
        return []

    docs = []
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"[Parser] Ошибка извлечения текста страницы {page_num}: {e}")
            continue

        # Нормализуем пробелы и переносы строк
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # убираем переносы слов

        if len(text) < MIN_CHUNK_LENGTH:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source":      str(file_path),
                "type":        "pdf",
                "page":        page_num,
                "total_pages": total_pages,
                "file_name":   file_path.name,
            },
        ))

    logger.info(
        f"[Parser] PDF: {file_path.name} → {len(docs)} страниц из {total_pages}"
    )
    return docs


# ── TXT ───────────────────────────────────────────────────────────────────────

def parse_text(file_path: Path) -> list[Document]:
    """
    Читает .txt файл с автоопределением кодировки.
    Пробует UTF-8, затем CP1251 (Windows), затем latin-1.

    Args:
        file_path: Путь к .txt файлу

    Returns:
        Список из одного Document (весь файл)
    """
    text = None
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            text = file_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        logger.error(f"[Parser] Не удалось прочитать файл: {file_path.name}")
        return []

    text = text.strip()
    if len(text) < MIN_CHUNK_LENGTH:
        logger.warning(f"[Parser] Файл слишком короткий: {file_path.name}")
        return []

    logger.info(f"[Parser] TXT: {file_path.name} → {len(text)} символов")
    return [Document(
        page_content=text,
        metadata={
            "source":    str(file_path),
            "type":      "txt",
            "file_name": file_path.name,
        },
    )]


# ── Главный диспетчер ─────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {".md", ".markdown", ".pdf", ".txt"}


def parse_file(file_path: Path) -> list[Document]:
    """
    Автоматически определяет формат файла и вызывает нужный парсер.

    Args:
        file_path: Путь к файлу (.md, .markdown, .pdf, .txt)

    Returns:
        Список Document с текстом и метаданными

    Raises:
        ValueError: если формат файла не поддерживается
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Неподдерживаемый формат: '{suffix}'. "
            f"Поддерживаются: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if file_path.stat().st_size == 0:
        logger.warning(f"[Parser] Файл пустой: {file_path.name}")
        return []

    if suffix in (".md", ".markdown"):
        return parse_markdown(file_path)
    elif suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".txt":
        return parse_text(file_path)

    return []  # недостижимо, но для mypy


# ── Разбивка на чанки ─────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Разбивает список Document на перекрывающиеся чанки.

    Стратегия разделения (от грубого к мелкому):
      1. Двойной перенос строки (конец параграфа)
      2. Одинарный перенос строки
      3. Конец предложения (. ! ?)
      4. Пробел
      5. Пустая строка (крайний случай)

    Сохраняет метаданные исходного документа в каждом чанке.

    Args:
        docs:          Список Document от parse_file()
        chunk_size:    Максимальный размер чанка в символах
        chunk_overlap: Перекрытие между соседними чанками в символах

    Returns:
        Список чанков-Document (короткие < MIN_CHUNK_LENGTH отфильтрованы)
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    # Фильтруем слишком короткие чанки
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    logger.info(
        f"[Parser] Чанкинг: {len(docs)} документов → {len(chunks)} чанков "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ── Утилиты ───────────────────────────────────────────────────────────────────

def parse_directory(directory: Path, recursive: bool = False) -> list[Document]:
    """
    Парсит все поддерживаемые файлы в директории.

    Args:
        directory: Путь к папке с заметками
        recursive: Если True — рекурсивно обходит подпапки

    Returns:
        Объединённый список Document из всех файлов
    """
    if not directory.exists():
        logger.warning(f"[Parser] Директория не найдена: {directory}")
        return []

    pattern = "**/*" if recursive else "*"
    all_docs: list[Document] = []

    for file_path in sorted(directory.glob(pattern)):
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        try:
            docs = parse_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"[Parser] Ошибка парсинга {file_path.name}: {e}")

    logger.info(
        f"[Parser] Директория {directory}: найдено {len(all_docs)} документов "
        f"из {len(list(directory.glob(pattern)))} файлов"
    )
    return all_docs


def get_file_info(file_path: Path) -> dict:
    """
    Возвращает информацию о файле перед парсингом.

    Args:
        file_path: Путь к файлу

    Returns:
        Словарь с именем, форматом, размером
    """
    suffix = file_path.suffix.lower()
    size_kb = file_path.stat().st_size / 1024 if file_path.exists() else 0
    return {
        "name":      file_path.name,
        "format":    suffix.lstrip(".").upper(),
        "size_kb":   round(size_kb, 1),
        "supported": suffix in SUPPORTED_FORMATS,
    }