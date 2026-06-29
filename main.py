# Network Influence API — Entry Point

## Estructura del Entry Point

```
app/
├── main.py                 ← Entry point principal
├── core/
│   └── config.py          ← Settings
├── api/
│   └── v1/
│       └── router.py      ← Router agregador
└── lifespan.py            ← Startup/shutdown
```

---

## `app/core/config.py`

```python
"""
Configuración centralizada via Pydantic Settings.
Todas las variables se leen desde environment / .env
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Aplicación ──────────────────────────────────────────────────────────
    APP_NAME: str = "Network Influence API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = Field(default=False)

    # ── Servidor ─────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    ROOT_PATH: str = ""

    # ── Seguridad ────────────────────────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="HMAC secret para firmar tokens internos",
    )
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS origins permitidos (frontend Next.js)",
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_DSN: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/network_influence"
    )
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 20

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_DSN: RedisDsn = Field(default="redis://localhost:6379/0")
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j"
    NEO4J_DATABASE: str = "neo4j"

    # ── Pipeline causal ───────────────────────────────────────────────────────
    BOOTSTRAP_ITERATIONS: int = Field(
        default=1000,
        description="Iteraciones bootstrap para CIs por arista",
    )
    PAGERANK_ALPHA: float = Field(
        default=0.85,
        description="Damping factor del CausalPageRank",
    )
    PAGERANK_EPSILON: float = Field(
        default=1e-6,
        description="Umbral de convergencia del solver",
    )
    GRANGER_MAX_LAG: int = Field(
        default=5,
        description="Máximo lag temporal para Granger causality",
    )
    PC_ALPHA: float = Field(
        default=0.05,
        description="Nivel de significancia para PC Algorithm",
    )

    # ── Jobs ──────────────────────────────────────────────────────────────────
    JOB_TTL_SECONDS: int = Field(
        default=86_400,
        description="TTL de resultados en cache (24h)",
    )
    JOB_MAX_NODES: int = Field(
        default=10_000,
        description="Límite de nodos aceptados por job",
    )
    JOB_MAX_EDGES: int = Field(
        default=100_000,
        description="Límite de aristas aceptadas por job",
    )

    # ── Observabilidad ────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    SENTRY_DSN: str | None = None
    OTLP_ENDPOINT: str | None = None  # OpenTelemetry collector

    @field_validator("PAGERANK_ALPHA")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("PAGERANK_ALPHA debe estar en (0, 1)")
        return v

    @field_validator("PC_ALPHA")
    @classmethod
    def validate_pc_alpha(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("PC_ALPHA debe estar en (0, 1)")
        return v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def postgres_dsn_sync(self) -> str:
        """DSN síncrono para Alembic / scripts de migración."""
        return str(self.POSTGRES_DSN).replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton cacheado: la instancia se crea una sola vez
    y se reutiliza durante todo el ciclo de vida del proceso.
    """
    return Settings()
```

---

## `app/lifespan.py`

```python
"""
Lifespan context manager de FastAPI.
Centraliza startup y shutdown de recursos compartidos:
  - Pool de conexiones PostgreSQL (SQLAlchemy async)
  - Driver Neo4j (async)
  - Conexión Redis
  - (Opcional) inicialización de Sentry / OTLP
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Estado global de la aplicación ───────────────────────────────────────────
# Se almacena en app.state para que las dependencias de FastAPI
# puedan accederlo sin importaciones circulares.

class AppState:
    """Contenedor tipado del estado compartido entre requests."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    neo4j_driver: AsyncGraphDatabase
    redis: Redis


_state = AppState()


async def _init_postgres() -> None:
    """Crea el engine async y verifica conectividad."""
    logger.info("Iniciando pool PostgreSQL…")
    _state.engine = create_async_engine(
        str(settings.POSTGRES_DSN),
        pool_size=settings.POSTGRES_POOL_SIZE,
        max_overflow=settings.POSTGRES_MAX_OVERFLOW,
        pool_pre_ping=True,          # detecta conexiones muertas
        echo=settings.DEBUG,
    )
    _state.session_factory = async_sessionmaker(
        bind=_state.engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    # Verificar que la base de datos responde
    async with _state.engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    logger.info("PostgreSQL conectado ✓")


async def _shutdown_postgres() -> None:
    logger.info("Cerrando pool PostgreSQL…")
    await _state.engine.dispose()
    logger.info("PostgreSQL desconectado ✓")


async def _init_neo4j() -> None:
    logger.info("Iniciando driver Neo4j…")
    _state.neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    # Verificar conectividad
    await _state.neo4j_driver.verify_connectivity()
    logger.info("Neo4j conectado ✓")


async def _shutdown_neo4j() -> None:
    logger.info("Cerrando driver Neo4j…")
    await _state.neo4j_driver.close()
    logger.info("Neo4j desconectado ✓")


async def _init_redis() -> None:
    logger.info("Iniciando conexión Redis…")
    _state.redis = Redis.from_url(
        str(settings.REDIS_DSN),
        encoding="utf-8",
        decode_responses=True,
    )
    await _state.redis.ping()
    logger.info("Redis conectado ✓")


async def _shutdown_redis() -> None:
    logger.info("Cerrando conexión Redis…")
    await _state.redis.aclose()
    logger.info("Redis desconectado ✓")


def _init_sentry() -> None:
    """Configura Sentry solo en staging/production si el DSN está presente."""
    if settings.SENTRY_DSN and settings.is_production:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.APP_ENV,
                release=settings.APP_VERSION,
                integrations=[FastApiIntegration(), SqlalchemyIntegration()],
                traces_sample_rate=0.1,
            )
            logger.info("Sentry inicializado ✓")
        except ImportError:
            logger.warning("sentry-sdk no instalado, omitiendo Sentry")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Ciclo de vida completo de la aplicación.

    FastAPI llama a este context manager en startup (antes de aceptar
    requests) y en shutdown (después de procesar los últimos requests).
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info(
        "Iniciando %s v%s [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )

    _init_sentry()
    await _init_postgres()
    await _init_neo4j()
    await _init_redis()

    # Exponer estado en app.state para las dependencias
    app.state.db_engine = _state.engine
    app.state.db_session_factory = _state.session_factory
    app.state.neo4j_driver = _state.neo4j_driver
    app.state.redis = _state.redis

    logger.info("Todos los recursos inicializados. Aceptando requests ✓")

    yield  # ← FastAPI sirve requests aquí

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Iniciando shutdown…")
    await _shutdown_redis()
    await _shutdown_neo4j()
    await _shutdown_postgres()
    logger.info("Shutdown completo ✓")


def get_app_state() -> AppState:
    """Acceso programático al estado (útil en tests y scripts)."""
    return _state
```

---

## `app/api/v1/router.py`

```python
"""
Router raíz de la API v1.
Agrega todos los sub-routers de los módulos core.
Cada sub-router se importa con importación lazy-friendly
para facilitar el testing unitario de módulos aislados.
"""

from __future__ import annotations

from fastapi import APIRouter

# Sub-routers de cada módulo (se crean en sus respectivos archivos)
from app.api.v1.endpoints.jobs import router as jobs_router
from app.api.v1.endpoints.graphs import router as graphs_router
from app.api.v1.endpoints.analysis import router as analysis_router
from app.api.v1.endpoints.export import router as export_router
from app.api.v1.endpoints.health import router as health_router

v1_router = APIRouter(prefix="/v1")

# ── Registro de sub-routers ───────────────────────────────────────────────────
v1_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"],
)
v1_router.include_router(
    graphs_router,
    prefix="/graphs",
    tags=["Graphs"],
    # Requiere API Key en todos los endpoints de este sub-router
)
v1_router.include_router(
    jobs_router,
    prefix="/jobs",
    tags=["Jobs"],
)
v1_router.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Analysis"],
)
v1_router.include_router(
    export_router,
    prefix="/export",
    tags=["Export & Audit"],
)
```

---

## `app/main.py`

```python
"""
Entry point principal de Network Influence API.

Responsabilidades de este módulo (y SOLO estas):
  1. Instanciar la aplicación FastAPI con lifespan y metadata OpenAPI.
  2. Registrar middleware (CORS, logging, autenticación de API Key).
  3. Montar el router v1.
  4. Exponer el handler de excepciones globales.
  5. Exponer `app` para Uvicorn / Gunicorn.

Regla de oro: main.py no contiene lógica de negocio.
Cada responsabilidad está delegada a su módulo correspondiente.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.core.config import get_settings
from app.lifespan import lifespan

# ── Configuración de logging estructurado ────────────────────────────────────
settings = get_settings()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if not settings.is_production
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.LOG_LEVEL)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


# ── Factory de la aplicación ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    """