import asyncio
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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
            -- Session-level summaries (preferred storage for end-of-session aggregation)
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                records_json TEXT NOT NULL,        -- JSON array of mistake records
                counts_json  TEXT NOT NULL,        -- JSON array of [mistake_type, count]
                total_mistakes INTEGER NOT NULL,   -- convenience total
                level_cefr TEXT,                   -- optional session-level CEFR
                level_confidence REAL,
                level_method TEXT,
                level_window INTEGER,
                level_explanation TEXT
            );
            """
        )
        self._conn.commit()

    async def record_session_summary(
        self,
        session_id: int,
        *,
        records: Iterable[dict[str, Any]],
        counts: Iterable[tuple[str, int]] | Iterable[list[Any]],
        level: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._record_session_summary_sync,
            session_id,
            list(records),
            list(counts),
            level or {},
            timestamp,
        )

    def _record_session_summary_sync(
        self,
        session_id: int,
        records: list[dict[str, Any]],
        counts: list[Any],
        level: dict[str, Any],
        timestamp: str | None,
    ) -> None:
        ts = (
            timestamp
            or sqlite3.connect(':memory:')
            .execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')")
            .fetchone()[0]
        )
        total = int(len(records))
        rec_json = json.dumps(records, ensure_ascii=False)
        cnt_json = json.dumps(counts, ensure_ascii=False)
        cefr = level.get('cefr')
        conf = level.get('confidence')
        method = level.get('method')
        window = level.get('window_size') or level.get('window')
        expl = level.get('explanation')
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO session_summaries(
              session_id, created_at, records_json, counts_json, total_mistakes,
              level_cefr, level_confidence, level_method, level_window, level_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              created_at=excluded.created_at,
              records_json=excluded.records_json,
              counts_json=excluded.counts_json,
              total_mistakes=excluded.total_mistakes,
              level_cefr=excluded.level_cefr,
              level_confidence=excluded.level_confidence,
              level_method=excluded.level_method,
              level_window=excluded.level_window,
              level_explanation=excluded.level_explanation
            """,
            (session_id, ts, rec_json, cnt_json, total, cefr, conf, method, window, expl),
        )
        self._conn.commit()

    async def get_session_summary(self, session_id: int) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_session_summary_sync, session_id)

    def _get_session_summary_sync(self, session_id: int) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT session_id, created_at, records_json, counts_json, total_mistakes,
                   level_cefr, level_confidence, level_method, level_window, level_explanation
            FROM session_summaries WHERE session_id=?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            'session_id': row[0],
            'created_at': row[1],
            'records': json.loads(row[2] or '[]'),
            'counts': json.loads(row[3] or '[]'),
            'total_mistakes': row[4],
            'level': {
                'cefr': row[5],
                'confidence': row[6],
                'method': row[7],
                'window_size': row[8],
                'explanation': row[9],
            },
        }

    async def export_summary_json(self, json_path: str, session_id: int) -> Path:
        return await asyncio.to_thread(self._export_summary_json_sync, json_path, session_id)

    def _export_summary_json_sync(self, json_path: str, session_id: int) -> Path:
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self._get_session_summary_sync(session_id) or {}
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return out

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)
