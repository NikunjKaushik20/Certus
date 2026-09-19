"""Engine, session and declarative base. SQLite by default; any SQLAlchemy URL works."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

_sqlite = settings.db_url.startswith("sqlite")
engine = create_engine(settings.db_url, future=True, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if _sqlite else {})

if _sqlite:
    @event.listens_for(engine, "connect")
    def _pragmas(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")        # readers never block the ingest worker
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def ensure_columns(table: str, columns: dict[str, str]) -> list[str]:
    """Add missing columns to an existing table. SQLite only, additive only.

    create_all() creates tables that do not exist and silently leaves existing ones alone, so a
    database written before a new column was declared keeps working right up until the first query
    that mentions it. A full migration tool is the wrong weight for one demo database; refusing to
    start and telling the user to delete their visits is worse. ADD COLUMN on SQLite is O(1) and
    cannot lose data, so the narrow fix is the honest one. Returns the columns it added.
    """
    if not _sqlite:
        return []
    added = []
    with engine.begin() as conn:
        have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
        if not have:
            return []                                  # table does not exist yet; create_all owns it
        for name, decl in columns.items():
            if name not in have:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                added.append(name)
    return added


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
