from __future__ import annotations

import asyncio
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from google.cloud import bigquery


class MistakeStore:
    """BigQuery-backed session mistake summaries store."""

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
        user_name: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._record_session_summary_sync,
            session_id,
            records,
            counts,
            level or {},
            timestamp,
            user_name,
        )

    def _record_session_summary_sync(
        self,
        session_id: int,
        records: list[dict[str, Any]],
        counts: list[Any],
        level: dict[str, Any],
        timestamp: str | None,
        user_name: str | None,
    ) -> None:
        from datetime import datetime, timezone

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        lvl = level or {}

        # Use parameterized query
        params = [
            bigquery.ScalarQueryParameter('session_id', 'INT64', session_id),
            bigquery.ScalarQueryParameter('user_name', 'STRING', user_name),
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
          (session_id, user_name, created_at, records_json, counts_json, total_mistakes,
           level_cefr, level_confidence, level_method, level_window, level_explanation)
        VALUES (
          @session_id,
          @user_name,
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

    async def get_user_mistakes(
        self,
        user_name: str | None = None,
        *,
        since_days: int | None = 30,
        limit_summaries: int = 10,
    ) -> dict[str, Any]:
        """
        Aggregate mistakes across recent session summaries (no join to sessions).
        Tries to filter by speaker_name or user_name columns if present in session_summaries.
        """
        return await asyncio.to_thread(
            self._get_user_mistakes_sync, user_name, since_days, limit_summaries
        )

    def _get_user_mistakes_sync(
        self,
        user_name: str | None,
        since_days: int | None,
        limit_summaries: int,
    ) -> dict[str, Any]:
        params = [
            bigquery.ScalarQueryParameter('limit', 'INT64', int(limit_summaries)),
        ]
        since_clause = ''
        if since_days is not None:
            params.append(bigquery.ScalarQueryParameter('since_days', 'INT64', int(since_days)))
            since_clause = (
                'AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @since_days DAY)'
            )
        rows = []
        filtered_by_user = False
        if user_name:
            params_user = params + [
                bigquery.ScalarQueryParameter('user_name', 'STRING', str(user_name))
            ]
            q_by_user = f"""
            SELECT session_id, created_at, records_json, level_cefr, level_confidence
            FROM `{self.table_fq}`
            WHERE user_name = @user_name
                {since_clause}
            ORDER BY created_at DESC, session_id DESC
            LIMIT @limit
            """
            rows = list(
                self.client.query(
                    q_by_user,
                    job_config=bigquery.QueryJobConfig(query_parameters=params_user),
                ).result()
            )
            filtered_by_user = True

        # Merge/aggregate
        all_records: list[dict[str, Any]] = []
        type_counts: Counter[str] = Counter()
        examples_by_type: defaultdict[str, list[str]] = defaultdict(list)
        cefr_to_idx = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        levels_seen: list[int] = []
        idx_to_cefr = {v: k for k, v in cefr_to_idx.items()}

        for row in rows:
            lc = getattr(row, 'level_cefr', None)
            if lc:
                levels_seen.append(cefr_to_idx.get(str(lc), 3))

            recs = getattr(row, 'records_json', None)
            if isinstance(recs, str):
                recs = json.loads(recs)
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict):
                    continue
                all_records.append(r)
                rtype = str(r.get('type') or r.get('category') or 'unknown')
                type_counts[rtype] += 1
                example = r.get('text') or r.get('example') or r.get('utterance') or ''
                if example and len(examples_by_type[rtype]) < 3:
                    examples_by_type[rtype].append(str(example)[:200])
        by_type = [
            {'type': t, 'count': c, 'examples': examples_by_type.get(t, [])}
            for t, c in type_counts.most_common()
        ]
        return {
            'filtered_by_user': filtered_by_user,
            'user_name': user_name,
            'total_summaries': len(rows),  # wording: session summaries
            'total_records': len(all_records),
            'by_type': by_type,
            'records': all_records,
            'level': idx_to_cefr.get(round(sum(levels_seen) / len(levels_seen)))
            if levels_seen
            else None,
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
