from __future__ import annotations

import json
import re
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.prompt import PromptManager

CEFR_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_IDX = {c: i for i, c in enumerate(CEFR_ORDER)}


# A compact French error taxonomy from L2 literature (morpho‑syntax oriented)
FR_ERROR_CATEGORIES: list[str] = [
    'conjugation',  # verb endings/person
    'tense',  # passé composé vs imparfait, futur vs conditionnel
    'mood/subjunctive',  # subjonctif vs indicatif, imperative forms
    'agreement-gender',  # genre: le/la; adjectif au bon genre
    'agreement-number',  # pluriel/singulier; accords
    'article/determiner',  # défini/indéfini/partitif; de/du/des; contractions (au/du/aux)
    'preposition',  # à/de/en/au/chez/pour/pendant/depuis; verbprep government
    'pronoun-object/clitic',  # me/te/se/le/la/lui/leur/y/en; ordre des pronoms
    'pronoun-relative',  # qui/que/dont/où/lequel
    'word-order/syntax',  # placement adverbes, ordre sujet/verbe/objet, interrogatives
    'negation',  # ne…pas/jamais/plus; double négation manquante
    'infinitive/participle',  # infinitif vs participe; verb chains
    'past-participle-agreement',  # avec être/avoir; COD avant le verbe
    'register/formality',  # tu/vous; formules de politesse
    'word-choice/false-friend',  # faux amis; calques; collocations
    'connector/cohesion',  # connecteurs logiques (mais/donc/or/car, puisque/parce que)
    'other-grammar',
]


@dataclass(frozen=True)
class LevelEstimate:
    cefr: str
    confidence: float
    method: str
    window_size: int
    explanation: str


@dataclass(frozen=True)
class MistakeRecord:
    mistake_type: str
    error: str
    correction: str
    explanation: str
    context: str
    difficulty: str
    timestamp: str  # ISO8601 UTC

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        system_tpl = str(self._prompt_cfg.get('system', '') or '')
        user_tpl = str(self._prompt_cfg.get('user', '') or 'Learner messages:\n{texts}')
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


# Shared builder: extract session-level mistake records from full conversation
async def build_mistake_records(
    history: list[dict[str, str]],
    *,
    llm: ChatGoogleGenerativeAI,
    prompt_manager: PromptManager,
    max_records: int = 10,
    language: str = 'fr',
    prompt_name: str = 'mistakes_v1',
    prompt_version: str | None = None,
    prompt_local_only: bool = False,
    max_chars: int = 16000,
) -> list[MistakeRecord]:
    """
    Produce MistakeRecord list from the entire conversation history.
    Assistant turns are used as context; only USER messages yield mistakes.
    """
    msgs = history or []
    if not msgs:
        return []

    cfg = prompt_manager.get_prompt(
        name=prompt_name, version=prompt_version, local_only=prompt_local_only
    )
    taxonomy = ', '.join(FR_ERROR_CATEGORIES)
    system_tpl = str(cfg.get('system', '') or '')
    system_msg = system_tpl.format(
        taxonomy=taxonomy,
        max_records=max_records,
        language=language,
    )

    lines: list[str] = []
    for m in msgs:
        role = (m.get('role') or '').lower()
        content = (m.get('content') or '').strip()
        if not content:
            continue
        if role in ('user', 'human'):
            lines.append(f'USER: {content}')
        elif role in ('assistant', 'ai'):
            lines.append(f'ASSISTANT: {content}')
        else:
            lines.append(f'SYSTEM: {content}')
    convo = '\n'.join(lines)
    if len(convo) > max_chars:
        convo = convo[-max_chars:]

    user_msg = (
        f'Conversation transcript (USER/ASSISTANT):\n{convo}\n\nExtract mistakes from USER only.'
    )
    prompt = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]

    ts_now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    res = await llm.ainvoke(prompt)
    raw = str(getattr(res, 'content', str(res))).strip()

    # Coerce JSON array/object from raw
    def _coerce_json(raw_txt: str) -> Any:
        try:
            return json.loads(raw_txt)
        except Exception:
            pass
        m = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', raw_txt)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return []
        return []

    data = _coerce_json(raw)

    records: list[MistakeRecord] = []
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for d in data:
            if not isinstance(d, dict):
                continue
            ctx_val = str(d.get('context', '')).strip()
            ctx = ctx_val or str(d.get('error', '')).strip()
            rec = MistakeRecord(
                mistake_type=str(d.get('mistake_type', '')).strip(),
                error=str(d.get('error', '')).strip(),
                correction=str(d.get('correction', '')).strip(),
                explanation=str(d.get('explanation', '')).strip(),
                context=ctx,
                difficulty=str(d.get('difficulty', 'A2')).strip().upper()[:2] or 'A2',
                timestamp=str(d.get('timestamp') or ts_now),
            )
            if rec.error and rec.correction:
                records.append(rec)

    if len(records) > max_records:
        records = records[:max_records]
    return records


class MistakeAnalyzer:
    """
    Conversation-level analyzer for mistakes only.
    - Takes the full history ([{'role': 'user'|'assistant', 'content': str}, ...])
    - Returns mistake records and aggregated counts.
    """

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        prompt_manager: PromptManager,
        max_records: int = 10,
        language: str = 'fr',
        prompt_name: str = 'mistakes_v1',
        prompt_version: str | None = None,
        prompt_local_only: bool = False,
    ) -> None:
        self.llm = llm
        self.max_records = max_records
        self.language = language
        self._prompt_mgr = prompt_manager
        self._prompt_name = prompt_name
        self._prompt_version = prompt_version
        self._prompt_local_only = prompt_local_only
        self._prompt_cfg: dict[str, Any] = self._load_prompt_cfg()

    def _load_prompt_cfg(self) -> dict[str, Any]:
        return self._prompt_mgr.get_prompt(
            name=self._prompt_name,
            version=self._prompt_version,
            local_only=self._prompt_local_only,
        )

    # Kept for compatibility with any callers, but no longer used internally
    def _build_messages(self, text: str, ts: str) -> list[dict[str, str]]:
        taxonomy = ', '.join(FR_ERROR_CATEGORIES)
        system_tpl = str(self._prompt_cfg.get('system', '') or '')
        user_tpl = str(
            self._prompt_cfg.get('user', '') or 'UTC timestamp: {timestamp}\nUtterance: {utterance}'
        )
        system_msg = system_tpl.format(
            taxonomy=taxonomy,
            max_records=self.max_records,
            language=self.language,
        )
        user_msg = user_tpl.format(
            timestamp=ts,
            utterance=text,
            language=self.language,
        )
        return [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ]

    def _build_conversation_messages(
        self,
        history: list[dict[str, str]],
        max_chars: int = 16000,
    ) -> list[dict[str, str]]:
        taxonomy = ', '.join(FR_ERROR_CATEGORIES)
        system_tpl = str(self._prompt_cfg.get('system', '') or '')
        system_msg = system_tpl.format(
            taxonomy=taxonomy,
            max_records=self.max_records,
            language=self.language,
        )
        lines: list[str] = []
        for m in history:
            role = (m.get('role') or '').lower()
            content = (m.get('content') or '').strip()
            if not content:
                continue
            if role in ('user', 'human'):
                lines.append(f'USER: {content}')
            elif role in ('assistant', 'ai'):
                lines.append(f'ASSISTANT: {content}')
            else:
                lines.append(f'SYSTEM: {content}')
        convo = '\n'.join(lines)
        if len(convo) > max_chars:
            convo = convo[-max_chars:]
        user_msg = f'Conversation transcript (USER/ASSISTANT):\n{convo}\n\nExtract mistakes from USER only.'
        return [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]

    async def analyze(
        self,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Returns:
        {
          'records': list[dict],           # aggregated mistake records (session-level)
          'counts': list[tuple[str,int]],  # (mistake_type, count)
        }
        """
        records = await build_mistake_records(
            history,
            llm=self.llm,
            prompt_manager=self._prompt_mgr,
            max_records=self.max_records,
            language=self.language,
            prompt_name=self._prompt_name,
            prompt_version=self._prompt_version,
            prompt_local_only=self._prompt_local_only,
        )

        counts = Counter(r.mistake_type or 'other-grammar' for r in records)
        counts_list = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        return {
            'records': [r.to_dict() for r in records],
            'counts': counts_list,
        }


# Orchestrator: call both record extraction and level estimation in one place
async def analyze_session(
    history: list[dict[str, str]],
    *,
    llm: ChatGoogleGenerativeAI,
    prompt_manager: PromptManager,
    max_records: int = 10,
    language: str = 'fr',
    mistakes_prompt_name: str = 'mistakes_v1',
    mistakes_prompt_version: str | None = None,
    levels_prompt_name: str = 'levels_v1',
    levels_prompt_version: str | None = None,
    prompt_local_only: bool = False,
) -> dict[str, Any]:
    """
    High-level helper to be called at session end.
    Returns { 'records', 'counts', 'level' }.
    """
    # 1) Mistake records for the whole session
    records = await build_mistake_records(
        history,
        llm=llm,
        prompt_manager=prompt_manager,
        max_records=max_records,
        language=language,
        prompt_name=mistakes_prompt_name,
        prompt_version=m_istakes_prompt_version
        if (m_istakes_prompt_version := mistakes_prompt_version)
        else None,
        prompt_local_only=prompt_local_only,
    )
    counts = Counter(r.mistake_type or 'other-grammar' for r in records)
    counts_list = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    # 2) Session-level CEFR estimate from all user turns
    user_texts = [
        m['content']
        for m in (history or [])
        if (m.get('role') or '').lower() in ('user', 'human') and (m.get('content') or '').strip()
    ]
    level_dict: dict[str, Any] | None = None
    if user_texts:
        est = LevelEstimator(
            llm=llm,
            prompt_manager=prompt_manager,
            window_size=max(3, len(user_texts)),
            prompt_name=levels_prompt_name,
            prompt_version=levels_prompt_version,
            prompt_local_only=prompt_local_only,
        )
        lev = await est.estimate_window(user_texts)
        if lev:
            level_dict = {
                'cefr': lev.cefr,
                'confidence': lev.confidence,
                'method': lev.method,
                'window_size': lev.window_size,
                'explanation': lev.explanation,
            }

    return {
        'records': [r.to_dict() for r in records],
        'counts': counts_list,
        'level': level_dict,
    }
