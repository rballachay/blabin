import asyncio
import csv
import hashlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    session_id: int
    turn_index: int
    user_text: str
    assistant_text: str
    timestamp: str  # ISO UTC


class MistakeStore:
    def __init__(self, db_path: str = 'data/mistakes.db') -> None:
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
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              name TEXT
            );
            CREATE TABLE IF NOT EXISTS mistakes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              turn_index INTEGER NOT NULL,
              hash TEXT NOT NULL, -- dedupe key within a session+turn
              mistake_type TEXT NOT NULL,
              error TEXT NOT NULL,
              correction TEXT NOT NULL,
              explanation TEXT NOT NULL,
              context TEXT NOT NULL,        -- user's utterance
              assistant_text TEXT NOT NULL, -- assistant reply for that turn
              difficulty TEXT NOT NULL,
              timestamp TEXT NOT NULL       -- turn timestamp
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mistake_unique
              ON mistakes(session_id, turn_index, hash);
            """
        )
        self._conn.commit()

    async def start_session(self, name: str | None = None) -> int:
        return await asyncio.to_thread(self._start_session_sync, name)

    def _start_session_sync(self, name: str | None) -> int:
        cur = self._conn.cursor()
        cur.execute('INSERT INTO sessions(name) VALUES (?) RETURNING id', (name,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError('No id returned from sessions insert')
        self._conn.commit()
        return int(row[0])

    async def record_turn(
        self,
        turn: TurnRecord,
        mistakes: Iterable[dict[str, Any]],
    ) -> None:
        # Store only mistakes; no per-turn row is created.
        await asyncio.to_thread(self._record_turn_sync, turn, list(mistakes))

    def _record_turn_sync(self, turn: TurnRecord, mistakes: list[dict[str, Any]]) -> None:
        if mistakes:
            cur = self._conn.cursor()
            for m in mistakes:
                h = hashlib.sha1(
                    f'{m.get("context", "")}|{m.get("error", "")}|{m.get("correction", "")}'.encode()
                ).hexdigest()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO mistakes(
                      session_id, turn_index, hash, mistake_type, error, correction,
                      explanation, context, assistant_text, difficulty, timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.session_id,
                        turn.turn_index,
                        h,
                        m.get('mistake_type', ''),
                        m.get('error', ''),
                        m.get('correction', ''),
                        m.get('explanation', ''),
                        m.get('context', turn.user_text),
                        turn.assistant_text,
                        m.get('difficulty', 'A2'),
                        m.get('timestamp', turn.timestamp),
                    ),
                )
            self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    async def export_csv(self, csv_path: str, session_id: int | None = None) -> Path:
        return await asyncio.to_thread(self._export_csv_sync, csv_path, session_id)

    def _export_csv_sync(self, csv_path: str, session_id: int | None) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cols = [
            'mistake_id',
            'session_id',
            'session_name',
            'session_created_at',
            'turn_index',
            'turn_timestamp',
            'user_text',  # from context
            'assistant_text',
            'mistake_type',
            'error',
            'correction',
            'explanation',
            'context',
            'difficulty',
            'mistake_timestamp',
            'hash',
        ]
        where = 'WHERE m.session_id = ?' if session_id is not None else ''
        args: tuple[Any, ...] = (session_id,) if session_id is not None else ()

        sql = f"""
        SELECT
          m.id         AS mistake_id,
          m.session_id AS session_id,
          s.name       AS session_name,
          s.created_at AS session_created_at,
          m.turn_index AS turn_index,
          m.timestamp  AS turn_timestamp,
          m.context    AS user_text,
          m.assistant_text AS assistant_text,
          m.mistake_type AS mistake_type,
          m.error      AS error,
          m.correction AS correction,
          m.explanation AS explanation,
          m.context    AS context,
          m.difficulty AS difficulty,
          m.timestamp  AS mistake_timestamp,
          m.hash       AS hash
        FROM mistakes m
        LEFT JOIN sessions s ON s.id = m.session_id
        {where}
        ORDER BY m.session_id, m.turn_index, m.id
        """
        cur = self._conn.cursor()
        cur.execute(sql, args)

        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader()
            for row in cur.fetchall():
                record = {k: row[i] for i, k in enumerate(cols)}
                writer.writerow(record)

        return out
