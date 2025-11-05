from __future__ import annotations

import asyncio
import csv
import os
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from google.cloud.bigquery import DatasetReference


class SessionStore:
    """BigQuery-backed session statistics store."""

    SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS `{table_fq}` (
      session_id INT64,
      created_at TIMESTAMP NOT NULL,
      ended_at TIMESTAMP,
      duration_sec FLOAT64,
      input_mode STRING,
      turns_total INT64,
      turns_user INT64,
      turns_assistant INT64,
      user_chars INT64,
      assistant_chars INT64,
      user_words INT64,
      assistant_words INT64,
      user_tokens_approx INT64,
      assistant_tokens_approx INT64,
      resp_latency_avg_ms FLOAT64,
      resp_latency_p95_ms FLOAT64,
      errors INT64,
      notes STRING
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY session_id
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        sessions_table: str = 'sessions',
        turns_table: str = 'session_turns',
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.sessions_table = sessions_table
        self.turns_table = turns_table

        self.client = bigquery.Client()
        self.dataset_fq = f'{self.project}.{self.dataset}'
        self.sessions_fq = f'{self.dataset_fq}.{self.sessions_table}'

        self._ensure_dataset()
        self._ensure_tables()

    def _ensure_dataset(self) -> None:
        dataset_ref = DatasetReference(self.project, self.dataset)
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(self.dataset_fq)
            dataset.location = os.getenv('BIGQUERY_LOCATION', 'US')
            self.client.create_dataset(dataset, exists_ok=True)

    def _ensure_tables(self) -> None:
        # Create sessions table
        ddl = self.SESSIONS_DDL.format(table_fq=self.sessions_fq)
        self.client.query(ddl).result()

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)

    async def upsert_session(self, session_row: dict[str, Any]) -> None:
        await asyncio.to_thread(self._upsert_session_sync, session_row)

    def _upsert_session_sync(self, r: dict[str, Any]) -> None:
        params = [
            bigquery.ScalarQueryParameter('session_id', 'INT64', r.get('session_id')),
            bigquery.ScalarQueryParameter('created_at', 'TIMESTAMP', r.get('created_at')),
            bigquery.ScalarQueryParameter('ended_at', 'TIMESTAMP', r.get('ended_at')),
            bigquery.ScalarQueryParameter('duration_sec', 'FLOAT64', r.get('duration_sec')),
            bigquery.ScalarQueryParameter('input_mode', 'STRING', r.get('input_mode')),
            bigquery.ScalarQueryParameter('turns_total', 'INT64', r.get('turns_total', 0)),
            bigquery.ScalarQueryParameter('turns_user', 'INT64', r.get('turns_user', 0)),
            bigquery.ScalarQueryParameter('turns_assistant', 'INT64', r.get('turns_assistant', 0)),
            bigquery.ScalarQueryParameter('user_chars', 'INT64', r.get('user_chars', 0)),
            bigquery.ScalarQueryParameter('assistant_chars', 'INT64', r.get('assistant_chars', 0)),
            bigquery.ScalarQueryParameter('user_words', 'INT64', r.get('user_words', 0)),
            bigquery.ScalarQueryParameter('assistant_words', 'INT64', r.get('assistant_words', 0)),
            bigquery.ScalarQueryParameter(
                'user_tokens_approx', 'INT64', r.get('user_tokens_approx', 0)
            ),
            bigquery.ScalarQueryParameter(
                'assistant_tokens_approx', 'INT64', r.get('assistant_tokens_approx', 0)
            ),
            bigquery.ScalarQueryParameter(
                'resp_latency_avg_ms', 'FLOAT64', r.get('resp_latency_avg_ms')
            ),
            bigquery.ScalarQueryParameter(
                'resp_latency_p95_ms', 'FLOAT64', r.get('resp_latency_p95_ms')
            ),
            bigquery.ScalarQueryParameter('errors', 'INT64', r.get('errors', 0)),
            bigquery.ScalarQueryParameter('notes', 'STRING', r.get('notes')),
        ]

        query = f"""
        INSERT INTO `{self.sessions_fq}`
          (session_id, created_at, ended_at, duration_sec, input_mode,
           turns_total, turns_user, turns_assistant,
           user_chars, assistant_chars, user_words, assistant_words,
           user_tokens_approx, assistant_tokens_approx,
           resp_latency_avg_ms, resp_latency_p95_ms, errors, notes)
        VALUES (
          @session_id, @created_at, @ended_at, @duration_sec, @input_mode,
          @turns_total, @turns_user, @turns_assistant,
          @user_chars, @assistant_chars, @user_words, @assistant_words,
          @user_tokens_approx, @assistant_tokens_approx,
          @resp_latency_avg_ms, @resp_latency_p95_ms, @errors, @notes
        )
        """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    async def export_sessions_csv(self, csv_path: str) -> Path:
        return await asyncio.to_thread(self._export_sessions_csv_sync, csv_path)

    def _export_sessions_csv_sync(self, csv_path: str) -> Path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        query = f"""
        SELECT session_id, created_at, ended_at, duration_sec, input_mode,
               turns_total, turns_user, turns_assistant,
               user_chars, assistant_chars, user_words, assistant_words,
               user_tokens_approx, assistant_tokens_approx,
               resp_latency_avg_ms, resp_latency_p95_ms, errors, notes
        FROM `{self.sessions_fq}`
        ORDER BY created_at DESC, session_id DESC
        """
        result = self.client.query(query).result()

        cols = [
            'session_id',
            'created_at',
            'ended_at',
            'duration_sec',
            'input_mode',
            'turns_total',
            'turns_user',
            'turns_assistant',
            'user_chars',
            'assistant_chars',
            'user_words',
            'assistant_words',
            'user_tokens_approx',
            'assistant_tokens_approx',
            'resp_latency_avg_ms',
            'resp_latency_p95_ms',
            'errors',
            'notes',
        ]

        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in result:
                writer.writerow(
                    {
                        'session_id': row.session_id,
                        'created_at': row.created_at.isoformat() if row.created_at else None,
                        'ended_at': row.ended_at.isoformat() if row.ended_at else None,
                        'duration_sec': row.duration_sec,
                        'input_mode': row.input_mode,
                        'turns_total': row.turns_total,
                        'turns_user': row.turns_user,
                        'turns_assistant': row.turns_assistant,
                        'user_chars': row.user_chars,
                        'assistant_chars': row.assistant_chars,
                        'user_words': row.user_words,
                        'assistant_words': row.assistant_words,
                        'user_tokens_approx': row.user_tokens_approx,
                        'assistant_tokens_approx': row.assistant_tokens_approx,
                        'resp_latency_avg_ms': row.resp_latency_avg_ms,
                        'resp_latency_p95_ms': row.resp_latency_p95_ms,
                        'errors': row.errors,
                        'notes': row.notes,
                    }
                )
        return out
