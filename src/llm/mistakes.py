from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

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
    Returns a list[dict] ready for DB insertion.
    """

    def __init__(
        self, llm: ChatGoogleGenerativeAI, max_records: int = 10, language: str = 'fr'
    ) -> None:
        self.llm = llm
        self.max_records = max_records
        self.language = language

    async def analyze(self, utterance: str, timestamp: str | None = None) -> list[dict[str, Any]]:
        text = (utterance or '').strip()
        if not text:
            return []

        ts = timestamp or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        taxonomy = ', '.join(FR_ERROR_CATEGORIES)
        sys = (
            'You are a strict French language error detector for L2 learners.\n'
            '- Ignore accents and minor spelling/orthography entirely (e.g., eleve vs élève). Do not report purely orthographic issues.\n'
            '- Only report clear grammatical errors and incorrect word choice/false friends. '
            "Include spelling only if it changes meaning, and classify it under 'word-choice/false-friend' or 'other-grammar' rather than 'spelling'.\n"
            'Keep explanation of errors very brief, and only pick out one to two major errors per turn.\n'
            '- Use the following taxonomy for mistake_type (pick the single best category): '
            f"{taxonomy}. If none fits, use 'other-grammar'.\n"
            '- Keep explanations concise and pedagogical.\n'
            'Output a JSON array of mistake objects with keys exactly: '
            'mistake_type, error, correction, explanation, context, difficulty, timestamp.\n'
            'difficulty must be one of A1, A2, B1, B2, C1, C2.\n'
            "context must repeat the student's utterance verbatim.\n"
            'timestamp must be the provided UTC timestamp.\n'
            'If there are no mistakes, return [].'
        )
        prompt = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': f'UTC timestamp: {ts}\nUtterance: {text}'},
        ]

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
                # keep only meaningful records
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
