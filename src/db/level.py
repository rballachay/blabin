from __future__ import annotations

import asyncio
import csv
import sqlite3
from pathlib import Path
from typing import Any


class LevelStore:
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
            -- Ensure sessions table exists (shared with mistakes)
            CREATE TABLE IF NOT EXISTS sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              name TEXT
            );

            -- Level estimates are stored per session and turn_index (no turns table)
            CREATE TABLE IF NOT EXISTS level_estimates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
              turn_index INTEGER NOT NULL,
              cefr TEXT NOT NULL,
              confidence REAL NOT NULL,
              method TEXT NOT NULL,          -- 'llm'|'heuristic'|'fallback'|'smoothed'
              window_size INTEGER NOT NULL,
              explanation TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );

            CREATE INDEX IF NOT EXISTS idx_levels_session_turn ON level_estimates(session_id, turn_index);
            """
        )
        self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    async def record_level(
        self,
        session_id: int,
        turn_index: int,
        cefr: str,
        confidence: float,
        method: str,
        window_size: int,
        explanation: str = '',
    ) -> None:
        await asyncio.to_thread(
            self._record_level_sync,
            session_id,
            turn_index,
            cefr,
            confidence,
            method,
            window_size,
            explanation,
        )

    def _record_level_sync(
        self,
        session_id: int,
        turn_index: int,
        cefr: str,
        confidence: float,
        method: str,
        window_size: int,
        explanation: str = '',
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO level_estimates(session_id, turn_index, cefr, confidence, method, window_size, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                cefr,
                float(confidence),
                method,
                int(window_size),
                explanation or '',
            ),
        )
        self._conn.commit()

    async def export_levels_csv(self, csv_path: str, session_id: int | None = None) -> Path:
        return await asyncio.to_thread(self._export_levels_csv_sync, csv_path, session_id)

    def _export_levels_csv_sync(self, csv_path: str, session_id: int | None) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cols = [
            'level_id',
            'session_id',
            'session_name',
            'session_created_at',
            'turn_index',
            'cefr',
            'confidence',
            'method',
            'window_size',
            'explanation',
            'created_at',
        ]

        where = 'WHERE l.session_id = ?' if session_id is not None else ''
        args: tuple[Any, ...] = (session_id,) if session_id is not None else ()

        sql = f"""
        SELECT
          l.id         AS level_id,
          l.session_id AS session_id,
          s.name       AS session_name,
          s.created_at AS session_created_at,
          l.turn_index AS turn_index,
          l.cefr       AS cefr,
          l.confidence AS confidence,
          l.method     AS method,
          l.window_size AS window_size,
          l.explanation AS explanation,
          l.created_at AS created_at
        FROM level_estimates l
        LEFT JOIN sessions s ON s.id = l.session_id
        {where}
        ORDER BY l.session_id, l.turn_index, l.id
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
