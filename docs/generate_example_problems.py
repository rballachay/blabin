from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.cloud import bigquery
from langchain_google_genai import ChatGoogleGenerativeAI

# Make "src" imports work when running from repo root
sys.path.append(str(Path(__file__).parents[1]))
from src.llm.converse import ConversationService  # noqa: E402

OUT_PATH = Path(__file__).with_name('practice_problems.txt')
DEFAULT_TABLE = 'session_summaries'


def load_env() -> dict[str, str]:
    load_dotenv(dotenv_path=Path(__file__).parents[1] / '.env')
    env = {
        'PROJECT': os.getenv('GOOGLE_CLOUD_PROJECT', '').strip(),
        'DATASET': os.getenv('BIGQUERY_DATASET', '').strip(),
        'CREDS': os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip(),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', '').strip(),
    }
    missing = [k for k in ('PROJECT', 'DATASET', 'GEMINI_API_KEY') if not env[k]]
    if missing:
        raise RuntimeError(f'Missing required env vars in .env: {", ".join(missing)}')
    if env['CREDS'] and not Path(env['CREDS']).exists():
        raise RuntimeError(f'GOOGLE_APPLICATION_CREDENTIALS not found: {env["CREDS"]}')
    return env


def bq_client() -> bigquery.Client:
    return bigquery.Client()


def fetch_recent_records(
    project: str, dataset: str, table: str, limit_rows: int
) -> list[dict[str, Any]]:
    client = bq_client()
    fq = f'{project}.{dataset}.{table}'
    q = f"""
      SELECT created_at, records_json
      FROM `{fq}`
      ORDER BY created_at DESC
      LIMIT @limit
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter('limit', 'INT64', int(limit_rows))]
    )
    rows = list(client.query(q, job_config=cfg).result())
    records: list[dict[str, Any]] = []
    for row in rows:
        recs = getattr(row, 'records_json', None)
        if isinstance(recs, str):
            try:
                recs = json.loads(recs)
            except Exception:
                recs = []
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            err = (r.get('error') or '').strip()
            cor = (r.get('correction') or '').strip()
            if not (err and cor):
                continue
            records.append(
                {
                    'mistake_type': str(r.get('mistake_type') or r.get('type') or '').strip(),
                    'error': err,
                    'correction': cor,
                    'explanation': (r.get('explanation') or '').strip(),
                    'context': (r.get('context') or '').strip(),
                    'difficulty': (r.get('difficulty') or 'B2').strip()[:2].upper(),
                }
            )
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    # Minimal structure expected by ConversationService.generate_practice_problems
    return {
        'filtered_by_user': False,
        'user_name': None,
        'total_summaries': 0,
        'total_records': len(records),
        'by_type': [],
        'records': records,
        'level': 'B2',
    }


# --- augmentation helpers ---


def _dedupe_by_error(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        err = (it.get('error') or '').strip()
        if not err or err in seen:
            continue
        seen.add(err)
        out.append(it)
    return out


def _build_augment_prompt(
    seed: list[dict[str, Any]], language: str, per_seed: int
) -> list[dict[str, str]]:
    examples = json.dumps(seed, ensure_ascii=False, indent=2)
    system = (
        'You generate short language-learning mistake examples.\n'
        'Return ONLY a JSON array of objects with keys: '
        'mistake_type, error, correction, explanation, context, difficulty.'
    )
    user = f"""
Language: {language}

Seed mistakes (JSON):
{examples}

Task:
- For each seed item, produce {per_seed} realistic new mistakes (same schema).
- Vary the surface form; keep the type and difficulty plausible.
- Keep each example one short sentence.
- Output ONLY a JSON array.
""".strip()
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def _augment_with_gemini(
    seed: list[dict[str, Any]], language: str, per_seed: int
) -> list[dict[str, Any]]:
    if not seed or per_seed <= 0:
        return []
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    msgs = _build_augment_prompt(seed, language, per_seed)
    res = llm.invoke(msgs)
    raw = str(getattr(res, 'content', str(res))).strip()
    try:
        data = json.loads(raw)
    except Exception:
        import re

        m = re.search(r'(\[[\s\S]*\])', raw)
        data = json.loads(m.group(1)) if m else []
    if isinstance(data, dict):
        data = [data]
    out: list[dict[str, Any]] = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        err = (d.get('error') or '').strip()
        cor = (d.get('correction') or '').strip()
        if not (err and cor):
            continue
        out.append(
            {
                'mistake_type': str(d.get('mistake_type') or '').strip(),
                'error': err,
                'correction': cor,
                'explanation': (d.get('explanation') or '').strip(),
                'context': (d.get('context') or '').strip(),
                'difficulty': (d.get('difficulty') or 'A2').strip()[:2].upper(),
            }
        )
    return out


# --- end augmentation helpers ---


async def run(
    count: int,
    limit_rows: int,
    table: str,
    out_path: Path,
    augment: bool,
    per_seed: int,
    max_seed: int,
    language: str,
) -> Path:
    env = load_env()
    records = fetch_recent_records(env['PROJECT'], env['DATASET'], table, limit_rows)

    # Optional augmentation
    if augment and records:
        seeds = records[: max(0, int(max_seed))]
        fake = _augment_with_gemini(seeds, language=language, per_seed=max(0, int(per_seed)))
        merged = _dedupe_by_error([*records, *fake])
        records = merged

    summary = build_summary(records)

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    conv = ConversationService(llm)

    problems = await conv.generate_practice_problems(summary, count=count)
    if not problems:
        problems = ['Ecrivez trois phrases correctes en utilisant le passé composé.']

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    header = [
        f'# Practice Problems ({ts})',
        f'Total problems: {len(problems)}',
    ]
    body = '\n'.join(f'- {p}' for p in problems)
    out_text = '\n'.join(header) + '\n\n' + body + '\n'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding='utf-8')
    print(f'[ok] wrote {out_path} ({len(problems)} problems from {len(records)} mistakes)')
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description='Generate practice problems using ConversationService (with optional Gemini augmentation).'
    )
    ap.add_argument('--count', type=int, default=20, help='Number of problems to generate')
    ap.add_argument(
        '--limit-rows', type=int, default=20, help='Rows to read from session_summaries'
    )
    ap.add_argument('--table', default=DEFAULT_TABLE, help='BigQuery table name')
    ap.add_argument('--out', default=str(OUT_PATH), help='Output text file path')
    ap.add_argument(
        '--augment', action='store_true', default=True, help='Augment mistakes with Gemini'
    )
    ap.add_argument('--per-seed', type=int, default=1, help='Augmented mistakes per seed')
    ap.add_argument('--max-seed', type=int, default=30, help='Max seed records to augment')
    ap.add_argument('--language', default='fr', help='Language hint for augmentation (e.g., fr)')
    args = ap.parse_args()
    asyncio.run(
        run(
            args.count,
            args.limit_rows,
            args.table,
            Path(args.out),
            args.augment,
            args.per_seed,
            args.max_seed,
            args.language,
        )
    )


if __name__ == '__main__':
    main()
