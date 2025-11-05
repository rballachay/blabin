from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from google.cloud import bigquery


class NewsStore:
    """BigQuery-backed news article store."""

    def __init__(
        self,
        project: str,
        dataset: str,
        table: str = 'articles',
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.table = table

        self.client = bigquery.Client()
        self.dataset_fq = f'{self.project}.{self.dataset}'
        self.table_fq = f'{self.dataset_fq}.{self.table}'

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)

    async def upsert_articles(self, source: str, rows: list[dict[str, Any]]) -> None:
        """
        Insert articles using streaming inserts (append-only).
        For true deduplication, use MERGE in a batch job (not implemented here).
        """
        if not rows:
            return
        await asyncio.to_thread(self._insert_articles_sync, source, rows)

    def _insert_articles_sync(self, source: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        # Use parameterized queries to avoid escaping issues
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            # Build INSERT with parameterized VALUES
            params = []
            value_placeholders = []

            for idx, r in enumerate(batch):
                article_id = abs(hash(r.get('link', ''))) % (10**15)

                # Convert timestamp formats to ISO 8601
                published = self._parse_timestamp(r.get('published'))
                fetched_at = self._parse_timestamp(r.get('fetched_at'))

                text = r.get('text', '')
                if isinstance(text, str):
                    text = text.strip()

                title = r.get('title', '')
                if isinstance(title, str):
                    title = title.strip()

                params.extend(
                    [
                        bigquery.ScalarQueryParameter(f'id_{idx}', 'INT64', article_id),
                        bigquery.ScalarQueryParameter(f'source_{idx}', 'STRING', source),
                        bigquery.ScalarQueryParameter(f'title_{idx}', 'STRING', title),
                        bigquery.ScalarQueryParameter(
                            f'link_{idx}', 'STRING', r.get('link', '') or ''
                        ),
                        bigquery.ScalarQueryParameter(f'published_{idx}', 'TIMESTAMP', published),
                        bigquery.ScalarQueryParameter(f'text_{idx}', 'STRING', text),
                        bigquery.ScalarQueryParameter(f'fetched_{idx}', 'TIMESTAMP', fetched_at),
                    ]
                )

                value_placeholders.append(
                    f'(@id_{idx}, @source_{idx}, @title_{idx}, @link_{idx}, @published_{idx}, @text_{idx}, @fetched_{idx})'
                )

            values_clause = ',\n  '.join(value_placeholders)

            query = f"""
            INSERT INTO `{self.table_fq}`
              (id, source, title, link, published, text, fetched_at)
            VALUES
              {values_clause}
            """

            job_config = bigquery.QueryJobConfig(query_parameters=params)
            self.client.query(query, job_config=job_config).result()

    @staticmethod
    def _parse_timestamp(ts: str | None) -> str | None:
        """
        Parse various timestamp formats and convert to ISO 8601.
        Handles: RFC 2822, ISO 8601, etc.
        """
        if not ts:
            return None

        try:
            # Try parsing as RFC 2822 (RSS feed format)
            dt = parsedate_to_datetime(ts)
            return dt.isoformat()
        except (TypeError, ValueError):
            pass

        try:
            # Try parsing as ISO 8601
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.isoformat()
        except (TypeError, ValueError):
            pass
        # Return None if unparseable
        return None

    async def recent_titles(
        self, limit: int = 5, source: str | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._recent_titles_sync, limit, source)

    def _recent_titles_sync(self, limit: int, source: str | None) -> list[dict[str, Any]]:
        where = ''
        params = []
        if source:
            where = 'WHERE source = @source'
            params.append(bigquery.ScalarQueryParameter('source', 'STRING', source))

        query = f"""
        SELECT id, title, link, published
        FROM `{self.table_fq}`
        {where}
        ORDER BY COALESCE(published, fetched_at) DESC, id DESC
        LIMIT @limit
        """
        params.append(bigquery.ScalarQueryParameter('limit', 'INT64', limit))

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        result = self.client.query(query, job_config=job_config).result()

        out = []
        for row in result:
            out.append(
                {
                    'id': row.id,
                    'title': row.title,
                    'link': row.link,
                    'published': row.published.isoformat() if row.published else None,
                }
            )
        return out

    async def get_article(self, article_id: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_article_sync, article_id)

    def _get_article_sync(self, article_id: int) -> dict[str, Any]:
        query = f"""
        SELECT id, source, title, link, published, text, fetched_at
        FROM `{self.table_fq}`
        WHERE id = @article_id
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('article_id', 'INT64', article_id)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            return {}
        row = rows[0]
        return {
            'id': row.id,
            'source': row.source,
            'title': row.title,
            'link': row.link,
            'published': row.published.isoformat() if row.published else None,
            'text': row.text,
            'fetched_at': row.fetched_at.isoformat() if row.fetched_at else None,
        }

    async def last_fetch(self, source: str | None = None) -> datetime | None:
        return await asyncio.to_thread(self._last_fetch_sync, source)

    def _last_fetch_sync(self, source: str | None) -> datetime | None:
        where = ''
        params = []
        if source:
            where = 'WHERE source = @source'
            params.append(bigquery.ScalarQueryParameter('source', 'STRING', source))

        query = f"""
        SELECT MAX(fetched_at) as max_fetched
        FROM `{self.table_fq}`
        {where}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows or not rows[0].max_fetched:
            return None
        return rows[0].max_fetched

    async def export_news_csv(
        self, csv_path: str, source: str | None = None, limit: int | None = None
    ) -> Path:
        return await asyncio.to_thread(self._export_news_csv_sync, csv_path, source, limit)

    def _export_news_csv_sync(self, csv_path: str, source: str | None, limit: int | None) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        where = ''
        params = []
        if source:
            where = 'WHERE source = @source'
            params.append(bigquery.ScalarQueryParameter('source', 'STRING', source))

        lim = ''
        if limit:
            lim = 'LIMIT @limit'
            params.append(bigquery.ScalarQueryParameter('limit', 'INT64', limit))

        query = f"""
        SELECT id, source, title, link, published, fetched_at, text
        FROM `{self.table_fq}`
        {where}
        ORDER BY COALESCE(published, fetched_at) DESC, id DESC
        {lim}
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        result = self.client.query(query, job_config=job_config).result()

        cols = ['id', 'source', 'title', 'link', 'published', 'fetched_at', 'text']
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in result:
                writer.writerow(
                    {
                        'id': row.id,
                        'source': row.source,
                        'title': row.title,
                        'link': row.link,
                        'published': row.published.isoformat() if row.published else None,
                        'fetched_at': row.fetched_at.isoformat() if row.fetched_at else None,
                        'text': row.text,
                    }
                )
        return out
