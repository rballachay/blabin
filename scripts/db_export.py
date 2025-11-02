#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db.level import LevelStore  # noqa: E402
from src.db.mistakes import MistakeStore  # noqa: E402
from src.db.news import NewsStore  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description='Export mistakes and level estimates to CSV.')
    parser.add_argument(
        '--db', default='data/mistakes.db', help='Path to the SQLite DB (default: data/mistakes.db)'
    )
    parser.add_argument(
        '--news-db',
        default='data/news.db',
        help='Path to the news SQLite DB (default: data/news.db)',
    )
    parser.add_argument(
        '--session-id', type=int, default=None, help='Filter to a specific session id (optional)'
    )
    parser.add_argument(
        '--mistakes-out',
        default='data/mistakes.csv',
        help='Output CSV for mistakes (default: data/mistakes.csv)',
    )
    parser.add_argument(
        '--levels-out',
        default='data/levels.csv',
        help='Output CSV for levels (default: data/levels.csv)',
    )
    parser.add_argument(
        '--news-out',
        default='data/news.csv',
        help='Output CSV for news articles (default: data/news.csv)',
    )
    args = parser.parse_args()

    store = MistakeStore(db_path=args.db)
    store_level = LevelStore(db_path=args.db)
    store_news = NewsStore(db_path=args.news_db)

    mistakes_path = await store.export_csv(args.mistakes_out, session_id=args.session_id)
    levels_path = await store_level.export_levels_csv(args.levels_out, session_id=args.session_id)
    news_path = await store_news.export_news_csv(
        args.news_out,
    )

    print(f'[ok] Mistakes CSV: {mistakes_path}')
    print(f'[ok] Levels CSV:   {levels_path}')
    print(f'[ok] News CSV:     {news_path}')

    await store.close()
    await store_level.close()
    await store_news.close()


if __name__ == '__main__':
    asyncio.run(main())
