from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _approx_tokens(text: str) -> int:
    # Rough heuristic ~4 chars/token
    return max(0, int(math.ceil(len(text) / 4))) if text else 0


@dataclass
class StatsAccumulator:
    session_id: int
    input_mode: str  # 'audio' | 'text'
    created_at: str = field(default_factory=_now_iso)
    _t0: float = field(default_factory=time.perf_counter)
    _assistant_latencies: list[float] = field(default_factory=list)
    _user_turns: int = 0
    _assistant_turns: int = 0
    _user_chars: int = 0
    _assistant_chars: int = 0
    _user_words: int = 0
    _assistant_words: int = 0
    _user_tokens: int = 0
    _assistant_tokens: int = 0

    def record_user(self, text: str) -> None:
        words = len((text or '').split())
        toks = _approx_tokens(text or '')
        chars = len(text or '')
        self._user_turns += 1
        self._user_chars += chars
        self._user_words += words
        self._user_tokens += toks

    def record_assistant(self, text: str, latency_s: float, model: str | None) -> None:
        words = len((text or '').split())
        toks = _approx_tokens(text or '')
        chars = len(text or '')
        lat_ms = max(0.0, float(latency_s) * 1000.0)
        self._assistant_turns += 1
        self._assistant_chars += chars
        self._assistant_words += words
        self._assistant_tokens += toks
        self._assistant_latencies.append(lat_ms)

    def finish(self) -> dict[str, Any]:
        t1 = time.perf_counter()
        duration_sec = float(t1 - self._t0)
        lats = sorted(self._assistant_latencies)
        avg = (sum(lats) / len(lats)) if lats else 0.0
        p95 = lats[int(math.ceil(0.95 * len(lats))) - 1] if lats else 0.0

        session_row = {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'ended_at': _now_iso(),
            'duration_sec': duration_sec,
            'input_mode': self.input_mode,
            'turns_total': self._user_turns + self._assistant_turns,
            'turns_user': self._user_turns,
            'turns_assistant': self._assistant_turns,
            'user_chars': self._user_chars,
            'assistant_chars': self._assistant_chars,
            'user_words': self._user_words,
            'assistant_words': self._assistant_words,
            'user_tokens_approx': self._user_tokens,
            'assistant_tokens_approx': self._assistant_tokens,
            'resp_latency_avg_ms': avg,
            'resp_latency_p95_ms': p95,
            'errors': 0,
            'notes': None,
        }

        return session_row
