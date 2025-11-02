from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.prompt import PromptManager  # NEW

# A compact French error taxonomy from L2 literature (morpho‑syntax oriented)
FR_ERROR_CATEGORIES: list[str] = [
    'conjugation',  # verb endings/person
    'tense',  # passé composé vs imparfait, futur vs conditionnel
    'mood/subjunctive',  # subjonctif vs indicatif, imperative forms
    'agreement-gender',  # genre: le/la; adjectif au bon genre
    'agreement-number',  # pluriel/singulier; accords
    'article/determiner',  # défini/indéfini/partitif; de/du/des; contractions (au/du/aux)
    'preposition',  # à/de/en/au/chez/pour/pendant/depuis; verb+prep government
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


class MistakeAnalyzer:
    """
    Uses the LLM to extract structured mistake records from a user utterance.
    Prompt text is loaded via PromptManager when available, with a safe fallback.
    Returns a list[dict] ready for DB insertion.
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
        self._prompt_cfg: dict[str, Any] = self._load_prompt_cfg()  # NEW

    def _load_prompt_cfg(self) -> dict[str, Any]:  # NEW
        return self._prompt_mgr.get_prompt(
            name=self._prompt_name,
            version=self._prompt_version,
            local_only=self._prompt_local_only,
        )

    def _build_messages(self, text: str, ts: str) -> list[dict[str, str]]:  # NEW
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

    async def analyze(self, utterance: str, timestamp: str | None = None) -> list[dict[str, Any]]:
        text = (utterance or '').strip()
        if not text:
            return []

        ts = timestamp or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        prompt = self._build_messages(text, ts)  # NEW

        raw = ''
        try:
            res = await self.llm.ainvoke(prompt)
            raw = str(getattr(res, 'content', str(res))).strip()
            data = self._coerce_json(raw)
        except Exception:
            data = []

        records: list[MistakeRecord] = []
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for d in data:
                if not isinstance(d, dict):
                    continue
                rec = MistakeRecord(
                    mistake_type=str(d.get('mistake_type', '')).strip(),
                    error=str(d.get('error', '')).strip(),
                    correction=str(d.get('correction', '')).strip(),
                    explanation=str(d.get('explanation', '')).strip(),
                    context=text,
                    difficulty=str(d.get('difficulty', 'A2')).strip().upper()[:2] or 'A2',
                    timestamp=str(d.get('timestamp') or ts),
                )
                if rec.error and rec.correction:
                    records.append(rec)

        if len(records) > self.max_records:
            records = records[: self.max_records]

        return [r.to_dict() for r in records]

    def _coerce_json(self, raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Fallback: extract first JSON array/object
        import re

        m = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return []
        return []
