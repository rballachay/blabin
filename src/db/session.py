from __future__ import annotations

import asyncio
import csv
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, db_path: str = 'data/sessions.db') -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL;')
        self._conn.execute('PRAGMA synchronous=NORMAL;')
        self._conn.execute('PRAGMA foreign_keys=ON;')
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,             -- ISO8601 UTC
                ended_at   TEXT,                      -- ISO8601 UTC
                duration_sec REAL,                   -- total wall time
                input_mode TEXT,                     -- audio|text
                turns_total INTEGER,
                turns_user INTEGER,
                turns_assistant INTEGER,
                user_chars INTEGER,
                assistant_chars INTEGER,
                user_words INTEGER,
                assistant_words INTEGER,
                user_tokens_approx INTEGER,
                assistant_tokens_approx INTEGER,
                resp_latency_avg_ms REAL,
                resp_latency_p95_ms REAL,
                errors INTEGER DEFAULT 0,
                notes TEXT
            );
            """
        )
        self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    # High-level API
    async def upsert_session(self, session_row: dict[str, Any]) -> None:
        await asyncio.to_thread(self._upsert_session_sync, session_row)

    async def insert_turns(self, session_id: int, turns: Iterable[dict[str, Any]]) -> None:
        await asyncio.to_thread(self._insert_turns_sync, session_id, list(turns))

    # CSV export
    async def export_sessions_csv(self, csv_path: str) -> Path:
        return await asyncio.to_thread(self._export_sessions_csv_sync, csv_path)

    async def export_turns_csv(self, csv_path: str, session_id: int | None = None) -> Path:
        return await asyncio.to_thread(self._export_turns_csv_sync, csv_path, session_id)

    # ---------- sync impl ----------

    def _upsert_session_sync(self, r: dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions(
                session_id, created_at, ended_at, duration_sec, input_mode,
                turns_total, turns_user, turns_assistant,
                user_chars, assistant_chars, user_words, assistant_words,
                user_tokens_approx, assistant_tokens_approx,
                resp_latency_avg_ms, resp_latency_p95_ms, errors, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                created_at=excluded.created_at,
                ended_at=excluded.ended_at,
                duration_sec=excluded.duration_sec,
                input_mode=excluded.input_mode,
                turns_total=excluded.turns_total,
                turns_user=excluded.turns_user,
                turns_assistant=excluded.turns_assistant,
                user_chars=excluded.user_chars,
                assistant_chars=excluded.assistant_chars,
                user_words=excluded.user_words,
                assistant_words=excluded.assistant_words,
                user_tokens_approx=excluded.user_tokens_approx,
                assistant_tokens_approx=excluded.assistant_tokens_approx,
                resp_latency_avg_ms=excluded.resp_latency_avg_ms,
                resp_latency_p95_ms=excluded.resp_latency_p95_ms,
                errors=excluded.errors,
                notes=excluded.notes
            """,
            (
                r.get('session_id'),
                r.get('created_at'),
                r.get('ended_at'),
                r.get('duration_sec'),
                r.get('input_mode'),
                r.get('turns_total'),
                r.get('turns_user'),
                r.get('turns_assistant'),
                r.get('user_chars'),
                r.get('assistant_chars'),
                r.get('user_words'),
                r.get('assistant_words'),
                r.get('user_tokens_approx'),
                r.get('assistant_tokens_approx'),
                r.get('resp_latency_avg_ms'),
                r.get('resp_latency_p95_ms'),
                r.get('errors', 0),
                r.get('notes'),
            ),
        )
        self._conn.commit()

    def _insert_turns_sync(self, session_id: int, turns: list[dict[str, Any]]) -> None:
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO session_turns(
                session_id, turn_index, role, text_chars, text_words, tokens_approx,
                model, started_at, ended_at, latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    session_id,
                    t.get('turn_index'),
                    t.get('role'),
                    t.get('text_chars'),
                    t.get('text_words'),
                    t.get('tokens_approx'),
                    t.get('model'),
                    t.get('started_at'),
                    t.get('ended_at'),
                    t.get('latency_ms'),
                )
                for t in turns
            ],
        )
        self._conn.commit()

    def _export_sessions_csv_sync(self, csv_path: str) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT session_id, created_at, ended_at, duration_sec, input_mode,
                   turns_total, turns_user, turns_assistant,
                   user_chars, assistant_chars, user_words, assistant_words,
                   user_tokens_approx, assistant_tokens_approx,
                   resp_latency_avg_ms, resp_latency_p95_ms, errors, notes
            FROM sessions
            ORDER BY created_at DESC, session_id DESC
            """
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow(r)
        return out

    def _export_turns_csv_sync(self, csv_path: str, session_id: int | None) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cur = self._conn.cursor()
        if session_id is None:
            cur.execute(
                """
                SELECT session_id, turn_index, role, text_chars, text_words, tokens_approx,
                       model, started_at, ended_at, latency_ms
                FROM session_turns
                ORDER BY session_id DESC, turn_index ASC, role ASC
                """
            )
        else:
            cur.execute(
                """
                SELECT session_id, turn_index, role, text_chars, text_words, tokens_approx,
                       model, started_at, ended_at, latency_ms
                FROM session_turns
                WHERE session_id=?
                ORDER BY turn_index ASC, role ASC
                """,
                (session_id,),
            )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow(r)
        return out
