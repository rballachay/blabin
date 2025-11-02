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


async def main() -> None:
    parser = argparse.ArgumentParser(description='Export mistakes and level estimates to CSV.')
    parser.add_argument(
        '--db', default='data/mistakes.db', help='Path to the SQLite DB (default: data/mistakes.db)'
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
    args = parser.parse_args()

    store = MistakeStore(db_path=args.db)
    store_level = LevelStore(db_path=args.db)

    mistakes_path = await store.export_csv(args.mistakes_out, session_id=args.session_id)
    levels_path = await store_level.export_levels_csv(args.levels_out, session_id=args.session_id)

    print(f'[ok] Mistakes CSV: {mistakes_path}')
    print(f'[ok] Levels CSV:   {levels_path}')

    await store.close()


if __name__ == '__main__':
    asyncio.run(main())
