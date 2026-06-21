"""
SQLAlchemy database setup.

Replaces the old "current_params.json on disk, single session" model with
a real SQLite database. This gives us:
  - persistence across server restarts
  - multiple uploads / conversations addressed by id, instead of one
    global file that gets overwritten by the next upload
  - a clean path to swap SQLite -> Postgres later (just change DATABASE_URL,
    nothing else in the app needs to know)
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./risk_intelligence.db")

# check_same_thread=False is needed because FastAPI can hand requests to
# different threads, and SQLite by default only allows the thread that
# created a connection to use it.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency - yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables if they don't exist yet. Call once at app startup."""
    from models import db_models  # noqa: F401  (registers models on Base)
    Base.metadata.create_all(bind=engine)