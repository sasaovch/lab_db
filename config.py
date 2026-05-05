import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TelegramConfig:
    token: str        = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    webhook_url: str  = field(default_factory=lambda: os.getenv("WEBHOOK_URL", ""))
    webhook_port: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_PORT", "8443")))
    allowed_users: list[int] = field(default_factory=lambda: [
        int(uid.strip()) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",")
        if uid.strip().isdigit()
    ])

@dataclass
class OllamaConfig:
    base_url:    str   = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model:       str   = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))
    temperature: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.3")))
    num_predict: int   = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_PREDICT", "1024")))
    timeout:     float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "120.0")))

@dataclass
class EmbeddingConfig:
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    device:     str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))

@dataclass
class RAGConfig:
    top_k:         int = field(default_factory=lambda: int(os.getenv("TOP_K", "3")))
    chunk_size:    int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))

@dataclass
class StorageConfig:
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
    notes_dir:        str = field(default_factory=lambda: os.getenv("NOTES_DIR", "./data/notes"))

@dataclass
class DatabaseConfig:
    host:     str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port:     int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name:     str = field(default_factory=lambda: os.getenv("DB_NAME", "rag_bot"))
    user:     str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_min: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MIN", "2")))
    pool_max: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX", "10")))

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def enabled(self) -> bool:
        return bool(self.password and self.name)

@dataclass
class Config:
    telegram:  TelegramConfig  = field(default_factory=TelegramConfig)
    ollama:    OllamaConfig    = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag:       RAGConfig       = field(default_factory=RAGConfig)
    storage:   StorageConfig   = field(default_factory=StorageConfig)
    database:  DatabaseConfig  = field(default_factory=DatabaseConfig)

    def validate(self):
        errors = []
        if not self.telegram.token:
            errors.append("TELEGRAM_BOT_TOKEN не задан")
        if errors:
            raise ValueError("Ошибки конфигурации:\n" + "\n".join(f"  • {e}" for e in errors))

    def summary(self) -> str:
        return (
            f"Бот: {'webhook' if self.telegram.webhook_url else 'polling'} | "
            f"LLM: Ollama/{self.ollama.model} | "
            f"БД: {'вкл' if self.database.enabled else 'откл'}"
        )

config = Config()