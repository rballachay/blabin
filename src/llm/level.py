from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.prompt import PromptManager

CEFR_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_IDX = {c: i for i, c in enumerate(CEFR_ORDER)}


@dataclass(frozen=True)
class FeatureRow:
    tokens: int
    types: int
    ttr: float
    avg_sent_len: float
    error_rate: float
    cat_counts: dict[str, int]


@dataclass(frozen=True)
class LevelEstimate:
    cefr: str
    confidence: float
    method: str
    window_size: int
    explanation: str


class LevelEstimator:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        prompt_manager: PromptManager,
        window_size: int = 5,
        ema_alpha: float = 0.4,
        prompt_name: str = 'levels_v1',
        prompt_version: str | None = None,
        prompt_local_only: bool = False,
    ):
        self.llm = llm
        self.window_size = max(1, window_size)
        self._ema_alpha = ema_alpha
        self._ema_score: float | None = None
        self._history_texts: deque[str] = deque(maxlen=self.window_size)

        self._pm = prompt_manager
        self._prompt_name = prompt_name
        self._prompt_version = prompt_version
        self._prompt_local_only = prompt_local_only
        self._prompt_cfg: dict[str, Any] = self._load_prompt_cfg()

    def _load_prompt_cfg(self) -> dict[str, Any]:
        return self._pm.get_prompt(
            name=self._prompt_name,
            version=self._prompt_version,
            local_only=self._prompt_local_only,
        )

    def _build_messages(self, texts: list[str]) -> list[dict[str, str]]:
        # Prefer YAML-configured prompt if available

        system_tpl = str(self._prompt_cfg.get('system', '') or '')
        user_tpl = str(self._prompt_cfg.get('user', '') or 'Learner messages:\n{texts}')
        # definitions/examples may be provided in YAML; fall back to built-ins
        defs = str(self._prompt_cfg.get('definitions'))
        exs = str(self._prompt_cfg.get('examples'))
        system_msg = system_tpl.format(
            definitions=defs,
            examples=exs,
            window_size=self.window_size,
        )
        user_msg = user_tpl.format(
            texts='\n\n'.join(texts[-self.window_size :]), window_size=self.window_size
        )
        return [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ]

    async def estimate_window(self, recent_texts: Iterable[str]) -> LevelEstimate | None:
        texts = [t for t in recent_texts if t and t.strip()]
        if not texts:
            return None

        prompt = self._build_messages(texts)

        res = await self.llm.ainvoke(prompt)
        out = str(getattr(res, 'content', str(res))).strip()
        data = self._coerce_json(out)

        cefr = str(data.get('cefr', 'B2')).upper()
        conf = float(data.get('confidence', 0.5))
        expl = str(data.get('explanation', '')).strip()

        return LevelEstimate(
            cefr=cefr,
            confidence=max(0.0, min(conf, 1.0)),
            method='llm',
            window_size=min(len(texts), self.window_size),
            explanation=expl,
        )

    def _coerce_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r'(\{[\s\S]*\})', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {}
        return {}

    def smooth_level(self, new_cefr: str) -> str:
        idx = CEFR_IDX.get(new_cefr, CEFR_IDX['B2'])
        if self._ema_score is None:
            self._ema_score = float(idx)
            return new_cefr
        self._ema_score = self._ema_alpha * idx + (1 - self._ema_alpha) * self._ema_score
        smoothed = int(round(self._ema_score))
        smoothed = max(0, min(smoothed, len(CEFR_ORDER) - 1))
        return CEFR_ORDER[smoothed]
