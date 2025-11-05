from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from google.cloud.bigquery import DatasetReference


class MistakeStore:
    """BigQuery-backed session mistake summaries store."""

    TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS `{table_fq}` (
      session_id INT64,
      created_at TIMESTAMP NOT NULL,
      records_json JSON,
      counts_json JSON,
      total_mistakes INT64 NOT NULL,
      level_cefr STRING,
      level_confidence FLOAT64,
      level_method STRING,
      level_window INT64,
      level_explanation STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY session_id
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        table: str = 'session_summaries',
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.table = table

        self.client = bigquery.Client()
        self.dataset_fq = f'{self.project}.{self.dataset}'
        self.table_fq = f'{self.dataset_fq}.{self.table}'

        self._ensure_dataset()
        self._ensure_table()

    def _ensure_dataset(self) -> None:
        dataset_ref = DatasetReference(self.project, self.dataset)
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(self.dataset_fq)
            dataset.location = os.getenv('BIGQUERY_LOCATION', 'US')
            self.client.create_dataset(dataset, exists_ok=True)

    def _ensure_table(self) -> None:
        ddl = self.TABLE_DDL.format(table_fq=self.table_fq)
        self.client.query(ddl).result()

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)

    async def record_session_summary(
        self,
        session_id: int,
        *,
        records: list[dict[str, Any]],
        counts: list[Any],
        level: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._record_session_summary_sync,
            session_id,
            records,
            counts,
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
        from datetime import datetime, timezone

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        lvl = level or {}

        # Use parameterized query
        params = [
            bigquery.ScalarQueryParameter('session_id', 'INT64', session_id),
            bigquery.ScalarQueryParameter('created_at', 'TIMESTAMP', ts),
            bigquery.ScalarQueryParameter('records_json', 'JSON', json.dumps(records)),
            bigquery.ScalarQueryParameter('counts_json', 'JSON', json.dumps(counts)),
            bigquery.ScalarQueryParameter('total_mistakes', 'INT64', len(records)),
            bigquery.ScalarQueryParameter('level_cefr', 'STRING', lvl.get('cefr')),
            bigquery.ScalarQueryParameter('level_confidence', 'FLOAT64', lvl.get('confidence')),
            bigquery.ScalarQueryParameter('level_method', 'STRING', lvl.get('method')),
            bigquery.ScalarQueryParameter(
                'level_window', 'INT64', lvl.get('window_size') or lvl.get('window')
            ),
            bigquery.ScalarQueryParameter('level_explanation', 'STRING', lvl.get('explanation')),
        ]

        query = f"""
        INSERT INTO `{self.table_fq}`
          (session_id, created_at, records_json, counts_json, total_mistakes,
           level_cefr, level_confidence, level_method, level_window, level_explanation)
        VALUES (
          @session_id,
          @created_at,
          @records_json,
          @counts_json,
          @total_mistakes,
          @level_cefr,
          @level_confidence,
          @level_method,
          @level_window,
          @level_explanation
        )
        """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    async def get_session_summary(self, session_id: int) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_session_summary_sync, session_id)

    def _get_session_summary_sync(self, session_id: int) -> dict[str, Any] | None:
        query = f"""
        SELECT session_id, created_at, records_json, counts_json, total_mistakes,
               level_cefr, level_confidence, level_method, level_window, level_explanation
        FROM `{self.table_fq}`
        WHERE session_id = @session_id
        ORDER BY created_at DESC
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('session_id', 'INT64', session_id)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            return None

        row = rows[0]
        return {
            'session_id': row.session_id,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'records': row.records_json,
            'counts': row.counts_json,
            'total_mistakes': row.total_mistakes,
            'level': {
                'cefr': row.level_cefr,
                'confidence': row.level_confidence,
                'method': row.level_method,
                'window_size': row.level_window,
                'explanation': row.level_explanation,
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

    async def export_csv(self, csv_path: str, session_id: int | None = None) -> Path:
        return await asyncio.to_thread(self._export_csv_sync, csv_path, session_id)

    def _export_csv_sync(self, csv_path: str, session_id: int | None) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        where = ''
        params = []
        if session_id is not None:
            where = 'WHERE session_id = @session_id'
            params.append(bigquery.ScalarQueryParameter('session_id', 'INT64', session_id))

        query = f"""
        SELECT session_id, created_at, total_mistakes,
               level_cefr, level_confidence, level_method, level_window, level_explanation,
               TO_JSON_STRING(records_json) as records_json,
               TO_JSON_STRING(counts_json) as counts_json
        FROM `{self.table_fq}`
        {where}
        ORDER BY created_at DESC, session_id DESC
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        result = self.client.query(query, job_config=job_config).result()

        cols = [
            'session_id',
            'created_at',
            'total_mistakes',
            'level_cefr',
            'level_confidence',
            'level_method',
            'level_window',
            'level_explanation',
            'records_json',
            'counts_json',
        ]

        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for row in result:
                writer.writerow(
                    [
                        row.session_id,
                        row.created_at.isoformat() if row.created_at else None,
                        row.total_mistakes,
                        row.level_cefr,
                        row.level_confidence,
                        row.level_method,
                        row.level_window,
                        row.level_explanation,
                        row.records_json,
                        row.counts_json,
                    ]
                )
        return out
