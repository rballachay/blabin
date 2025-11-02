from __future__ import annotations

import asyncio
import csv
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArticleRow:
    id: int
    source: str
    title: str
    link: str
    published: str | None
    text: str
    fetched_at: str


class NewsStore:
    def __init__(self, db_path: str = 'data/news.db') -> None:
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
            CREATE TABLE IF NOT EXISTS articles (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source TEXT NOT NULL,
              title TEXT NOT NULL,
              link TEXT NOT NULL UNIQUE,
              published TEXT,
              text TEXT NOT NULL,
              fetched_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
            CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
            CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
            """
        )
        self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    async def upsert_articles(self, source: str, rows: Iterable[dict[str, Any]]) -> None:
        await asyncio.to_thread(self._upsert_articles_sync, source, list(rows))

    def _upsert_articles_sync(self, source: str, rows: list[dict[str, Any]]) -> None:
        cur = self._conn.cursor()
        for r in rows:
            cur.execute(
                """
                INSERT INTO articles(source, title, link, published, text, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(link) DO UPDATE SET
                  source=excluded.source,
                  title=excluded.title,
                  published=COALESCE(excluded.published, articles.published),
                  text=excluded.text,
                  fetched_at=excluded.fetched_at
                """,
                (
                    source,
                    r.get('title', ''),
                    r.get('link', ''),
                    r.get('published'),
                    r.get('text', ''),
                    r.get('fetched_at', ''),
                ),
            )
        self._conn.commit()

    async def recent_titles(
        self, limit: int = 5, source: str | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._recent_titles_sync, limit, source)

    def _recent_titles_sync(self, limit: int, source: str | None) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        if source:
            cur.execute(
                """
                SELECT id, title, link, published FROM articles
                WHERE source = ?
                ORDER BY COALESCE(published, fetched_at) DESC, id DESC
                LIMIT ?
                """,
                (source, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, title, link, published FROM articles
                ORDER BY COALESCE(published, fetched_at) DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        out: list[dict[str, Any]] = []
        for row in cur.fetchall():
            out.append(
                {'id': int(row[0]), 'title': str(row[1]), 'link': str(row[2]), 'published': row[3]}
            )
        return out

    async def get_article(self, article_id: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_article_sync, article_id)

    def _get_article_sync(self, article_id: int) -> dict[str, Any]:
        cur = self._conn.cursor()
        row = cur.execute(
            'SELECT id, source, title, link, published, text, fetched_at FROM articles WHERE id = ?',
            (article_id,),
        ).fetchone()
        return {
            'id': int(row[0]),
            'source': str(row[1]),
            'title': str(row[2]),
            'link': str(row[3]),
            'published': row[4],
            'text': str(row[5]),
            'fetched_at': str(row[6]),
        }

    async def last_fetch(self, source: str | None = None) -> datetime | None:
        return await asyncio.to_thread(self._last_fetch_sync, source)

    def _last_fetch_sync(self, source: str | None) -> datetime | None:
        cur = self._conn.cursor()
        if source:
            row = cur.execute(
                'SELECT MAX(fetched_at) FROM articles WHERE source = ?', (source,)
            ).fetchone()
        else:
            row = cur.execute('SELECT MAX(fetched_at) FROM articles').fetchone()
        s = row[0] if row else None
        if not s:
            return None
        # fetched_at is stored as UTC like 2025-11-02T13:45:00Z
        try:
            return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        except Exception:
            try:
                # fallback for ISO without Z
                return datetime.fromisoformat(s)
            except Exception:
                return None

    async def export_news_csv(
        self,
        csv_path: str,
        source: str | None = None,
        limit: int | None = None,
    ) -> Path:
        """
        Export articles to CSV. Optionally filter by source and/or limit row count.
        Columns: id, source, title, link, published, fetched_at, text
        """
        return await asyncio.to_thread(self._export_news_csv_sync, csv_path, source, limit)

    def _export_news_csv_sync(
        self,
        csv_path: str,
        source: str | None,
        limit: int | None,
    ) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cols = ['id', 'source', 'title', 'link', 'published', 'fetched_at', 'text']

        where = ''
        args: list[Any] = []
        if source:
            where = 'WHERE source = ?'
            args.append(source)

        order = 'ORDER BY COALESCE(published, fetched_at) DESC, id DESC'
        lim = ''
        if isinstance(limit, int) and limit > 0:
            lim = 'LIMIT ?'
            args.append(limit)

        sql = f"""
        SELECT id, source, title, link, published, fetched_at, text
        FROM articles
        {where}
        {order}
        {lim}
        """
        cur = self._conn.cursor()
        cur.execute(sql, tuple(args))

        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader()
            for row in cur.fetchall():
                record = {k: row[i] for i, k in enumerate(cols)}
                writer.writerow(record)

        return out
