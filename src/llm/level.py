from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI

CEFR_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_IDX = {c: i for i, c in enumerate(CEFR_ORDER)}

# Concise CEFR definitions (paraphrased)
CEFR_DEFINITIONS: dict[str, str] = {
    'A1': 'Very basic phrases and simple sentences about self and everyday needs; limited range.',
    'A2': 'Simple, routine exchanges on familiar topics; short linked phrases; basic past/present.',
    'B1': 'Connected discourse on familiar matters; can narrate/describe with some subordination.',
    'B2': 'Clear, detailed language; good control of complex sentences and connectors; mostly accurate.',
    'C1': 'Fluent, flexible, and well-structured discourse with nuanced, precise grammar and lexis.',
    'C2': 'Near-native command; precise, idiomatic language across complex topics and registers.',
}

# Examples to anchor judgments on short samples
CEFR_EXAMPLES: dict[str, str] = {
    'A1': "Je m'appelle Marie. J'aime le café. Où est la gare ?",
    'A2': "Hier, je suis allé au marché et j'ai acheté des fruits. Je voudrais une baguette, s'il vous plaît.",
    'B1': "Je pense que le film était intéressant, même si la fin m'a un peu déçu.",
    'B2': 'Bien que je sois fatigué, je continuerai, car il est essentiel de terminer ce projet à temps.',
    'C1': 'Non seulement elle maîtrise la grammaire, mais elle sait aussi nuancer ses propos selon le contexte.',
    'C2': 'Quelles que soient les circonstances, il aurait fallu anticiper les conséquences de cette décision.',
}


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
        window_size: int = 5,
        ema_alpha: float = 0.4,
    ):
        self.llm = llm
        self.window_size = max(1, window_size)
        self._ema_alpha = ema_alpha
        self._ema_score: float | None = None
        self._history_texts: deque[str] = deque(maxlen=self.window_size)

    def _definitions_block(self) -> str:
        return '\n'.join(f'{lvl}: {desc}' for lvl, desc in CEFR_DEFINITIONS.items())

    def _examples_block(self) -> str:
        return '\n'.join(f'{lvl}: {ex}' for lvl, ex in CEFR_EXAMPLES.items())

    async def estimate_window(self, recent_texts: Iterable[str]) -> LevelEstimate | None:
        texts = [t for t in recent_texts if t and t.strip()]
        if not texts:
            return None

        defs = self._definitions_block()
        examples = self._examples_block()
        sys = (
            'You are a French language proficiency rater.\n'
            '- Task: Estimate CEFR level (A1, A2, B1, B2, C1, C2) from SHORT learner samples (often 1–3 sentences).\n'
            '- Important: For short or formulaic texts, be conservative and lower confidence; base decisions on observable grammar and range only.\n'
            '- Ignore minor spelling/accents. Do not reward topic knowledge.\n'
            '- Consider: grammatical accuracy (agreement, tense/aspect, subordination), sentence complexity, connectors/cohesion, vocabulary control/collocations.\n\n'
            'CEFR summaries:\n'
            f'{defs}\n\n'
            'Anchoring examples:\n'
            f'{examples}\n\n'
            'Reply in strict JSON only: {"cefr": "B1", "confidence": 0.72, "explanation": "one short sentence grounded in the sample"}.'
        )
        prompt = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': '\n\n'.join(texts[-self.window_size :])},
        ]

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
