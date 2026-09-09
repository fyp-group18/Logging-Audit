# Extracted from the companion project for evaluation reproducibility.
# Contains only the database connection layer required by the
# audit/ metric computation modules.
#
# Requires psycopg 3 (psycopg[binary]) — the connect_args use
# libpq TCP keepalive parameters specific to this driver.

import logging
import os
import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger("Database")

_engine = None
_SessionLocal = None
_init_lock = threading.Lock()


def _normalize_url(raw: str) -> str:
    if raw.startswith("postgresql+psycopg2://"):
        raw = "postgresql://" + raw[len("postgresql+psycopg2://"):]
    elif raw.startswith("postgresql+psycopg://"):
        raw = "postgresql://" + raw[len("postgresql+psycopg://"):]
    return raw.replace("postgresql://", "postgresql+psycopg://", 1)


def _init_engine():
    global _engine, _SessionLocal
    if _engine is not None:
        return
    with _init_lock:
        if _engine is not None:
            return
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable is required")
        _engine = create_engine(
            _normalize_url(url),
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            connect_args={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def is_connection_error(e):
    err_str = str(e).lower()
    if "ssl" in err_str or "closed unexpectedly" in err_str or "operationalerror" in err_str:
        logger.warning("DB connection dropped (Neon cold start). Retrying...")
        return True
    return False


with_db_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=2.0),
    retry=retry_if_exception(is_connection_error),
    reraise=True,
)


DATABASE_URL_RAW = os.getenv("DATABASE_URL", "")
DATABASE_URL = _normalize_url(DATABASE_URL_RAW) if DATABASE_URL_RAW else ""

Base = declarative_base()


def SessionLocal():
    """Return a new SQLAlchemy session, initializing the engine on first call."""
    _init_engine()
    return _SessionLocal()


def get_db():
    """FastAPI dependency that provides a database session per request."""
    with SessionLocal() as db:
        yield db


@with_db_retry
def init_db():
    from sqlalchemy import text
    from core.models import Base  # noqa: F811

    _init_engine()
    with _engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=_engine)
