"""SQLite register + relation graph. Mirrors the register/graph groups of
the Postgres schema in the DGX plan, so migrating later is a schema superset,
not a redesign.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS circular (
  circular_id   TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  native_id     TEXT NOT NULL,
  official_ref  TEXT,
  issue_date    TEXT NOT NULL,
  title         TEXT,
  department    TEXT,
  category      TEXT,
  segment       TEXT,
  doc_type      TEXT DEFAULT 'circular',
  status        TEXT DEFAULT 'unknown',
  status_reason TEXT,
  raw_path      TEXT,
  parsed_path   TEXT,
  UNIQUE(source, native_id, issue_date)
);

CREATE TABLE IF NOT EXISTS chunk_meta (
  chunk_id      TEXT PRIMARY KEY,
  circular_id   TEXT NOT NULL REFERENCES circular(circular_id),
  ordinal       INTEGER NOT NULL,
  section_path  TEXT,
  text_preview  TEXT
);

CREATE TABLE IF NOT EXISTS circular_relation (
  relation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  from_circular  TEXT NOT NULL REFERENCES circular(circular_id),
  to_circular    TEXT REFERENCES circular(circular_id),
  raw_reference  TEXT NOT NULL,
  relation       TEXT,
  confidence     REAL,
  method         TEXT
);

CREATE TABLE IF NOT EXISTS query_log (
  query_id     TEXT PRIMARY KEY,
  asked_at     TEXT,
  question     TEXT,
  retrieved_chunk_ids TEXT,
  answer       TEXT,
  latency_ms   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_circular_native ON circular(source, native_id);
CREATE INDEX IF NOT EXISTS idx_relation_to ON circular_relation(to_circular);
CREATE INDEX IF NOT EXISTS idx_chunk_circular ON chunk_meta(circular_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_circular(conn, **fields) -> str:
    existing = conn.execute(
        "SELECT circular_id FROM circular WHERE source=? AND native_id=? AND issue_date=?",
        (fields["source"], fields["native_id"], fields["issue_date"]),
    ).fetchone()
    if existing:
        circular_id = existing["circular_id"]
        set_clause = ", ".join(f"{k}=?" for k in fields if k != "circular_id")
        conn.execute(
            f"UPDATE circular SET {set_clause} WHERE circular_id=?",
            (*[v for k, v in fields.items() if k != "circular_id"], circular_id),
        )
        return circular_id

    circular_id = new_id()
    cols = ["circular_id", *fields.keys()]
    placeholders = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO circular ({', '.join(cols)}) VALUES ({placeholders})",
        (circular_id, *fields.values()),
    )
    return circular_id


def find_circular_by_native_id(conn, source: str, native_id: str):
    return conn.execute(
        "SELECT * FROM circular WHERE source=? AND native_id=? ORDER BY issue_date DESC",
        (source, native_id),
    ).fetchall()


def insert_chunk_meta(conn, chunk_id, circular_id, ordinal, section_path, text_preview):
    conn.execute(
        "INSERT OR REPLACE INTO chunk_meta (chunk_id, circular_id, ordinal, section_path, text_preview) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id, circular_id, ordinal, section_path, text_preview),
    )


def insert_relation(conn, from_circular, to_circular, raw_reference, relation, confidence, method):
    conn.execute(
        "INSERT INTO circular_relation (from_circular, to_circular, raw_reference, relation, confidence, method) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (from_circular, to_circular, raw_reference, relation, confidence, method),
    )


def relations_into(conn, circular_id):
    """Edges where another circular points at this one (used to compute status)."""
    return conn.execute(
        "SELECT * FROM circular_relation WHERE to_circular=?", (circular_id,)
    ).fetchall()


def log_query(conn, question, retrieved_chunk_ids, answer, latency_ms):
    import json
    from datetime import datetime, timezone

    conn.execute(
        "INSERT INTO query_log (query_id, asked_at, question, retrieved_chunk_ids, answer, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            new_id(),
            datetime.now(timezone.utc).isoformat(),
            question,
            json.dumps(retrieved_chunk_ids),
            answer,
            latency_ms,
        ),
    )
