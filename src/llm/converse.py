from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


class ConversationService:
    """
    Centralizes application-specific prompts and parsing for the LLM.
    Works with a LangChain ChatGoogleGenerativeAI instance.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self.llm = llm

    async def _resp_text(self, messages: list[dict[str, Any]]) -> str:
        try:
            resp = await self.llm.ainvoke(messages)
            return str(getattr(resp, 'content', str(resp))).strip()
        except Exception:
            return ''

    async def send_request(self, prompt: list[dict[str, Any]]) -> str:
        """
        Send a prompt (list of {'role','content'}) and return text.
        """
        return await self._resp_text(prompt)

    async def get_speaker_name(self, text: str) -> str:
        """
        Extract a name from text, or 'unknown'.
        """
        sys = (
            "Extract the user's stated name from the message. "
            'If a name is provided, reply with the name only. '
            'If not, reply exactly: unknown. No punctuation, no extra words.'
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': text},
        ]
        out = (await self._resp_text(messages)).strip()
        return out.lower() if out else 'unknown'

    async def is_confirmation(self, text: str) -> bool:
        """
        Return True if the user confirms (YES), else False.
        """
        sys = (
            'Determine if the user is confirming with yes. '
            "Reply exactly 'YES' for a confirmation, otherwise reply 'NONE'."
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': text},
        ]
        out = (await self._resp_text(messages)).strip().upper()
        return out == 'YES'

    async def make_confirmation_request(self, user_text: str, proposed_name: str) -> str:
        """
        Ask to confirm the proposed name, while acknowledging the user's message.
        """
        sys = (
            'You are a concise, polite language partner. '
            f"Ask the user to confirm if their name is '{proposed_name}' in one short sentence. "
            'Briefly acknowledge their message if appropriate. '
            'Respond in the same language as the user.'
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': f"Dernier message de l'utilisateur:\n{user_text}"},
        ]
        return await self._resp_text(messages)

    async def make_greeting(self, user_text: str, name: str, returning: bool) -> str:
        """
        Greet the user by name and respond contextually (1–2 sentences).
        """
        sys = (
            'You are a friendly, concise language tutor. '
            f"Craft a natural reply that greets the user by name ('{name}')"
            f'{" as a returning speaker" if returning else ""}, '
            'and addresses their message. Max 2 sentences. '
            'Respond in the same language as the user.'
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': f"Dernier message de l'utilisateur:\n{user_text}"},
        ]
        return await self._resp_text(messages)

    async def ask_for_name(self, user_text: str, reason: str | None = None) -> str:
        """
        Politely ask for their name, acknowledging their last message.
        """
        sys = (
            'You are a concise language partner. '
            "Acknowledge the user's message briefly, then ask their name politely in one short sentence. "
            'Respond in the same language as the user.'
        )
        if reason:
            sys += f' Context: {reason}.'
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': f"Dernier message de l'utilisateur:\n{user_text}"},
        ]
        return await self._resp_text(messages)

    async def clarify_confirmation(self, user_text: str, proposed_name: str) -> str:
        """
        Ask for a yes/no confirmation naturally and briefly.
        """
        sys = (
            'You are a concise, polite language partner. '
            f"Ask the user to confirm with yes/no if they are '{proposed_name}' in one short sentence. "
            'Respond in the same language as the user.'
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': f"Dernier message de l'utilisateur:\n{user_text}"},
        ]
        return await self._resp_text(messages)

    async def generate_practice_problems(
        self, mistake_summary: dict[str, Any], count: int = 20
    ) -> list[str]:
        """
        Generate 'count' short practice problems based on the user's recent mistake summary.
        Returns a list of problem strings. Keeps output concise and suitable for a .txt handout.
        """
        mistakes_records = mistake_summary.get('records', [])
        cefr = (mistake_summary.get('level') or '') if isinstance(mistake_summary, dict) else 'B2'
        level_hint = ''
        if cefr in {'A1', 'A2', 'B1', 'B2', 'C1', 'C2'}:
            level_hint = f' Target CEFR level: {cefr}. Adjust the difficulty of the prompts accordingly, tend to make them more difficult, as we would like to challenge the user.'

        sys = (
            "You are a concise French tutor. Based on the learner's recent mistakes, "
            f'generate exactly {max(1, int(count))} short practice prompts that help them fix those issues. '
            f'{level_hint} '
            'Use simple, clear instructions. Do not include answers. '
            'Output one prompt per line, with no numbering and no extra commentary.'
        )
        mistakes = [
            f'- Error: {i["error"]}, Explanation: {i["explanation"]}\n' for i in mistakes_records
        ]
        user = f'Here are some of my recent mistakes:\n{mistakes}\nPlease return only the prompts.'
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': user},
        ]
        text = await self._resp_text(messages)
        lines = [ll.strip() for ll in (text or '').splitlines() if ll.strip()]
        # Strip any accidental numbering like "1) " or "1. "
        out: list[str] = []
        import re

        for ll in lines:
            l2 = re.sub(r'^\s*\d+[\).\s-]+\s*', '', ll)
            if l2:
                out.append(l2)
        # Ensure exact count if possible
        if len(out) > count:
            out = out[:count]
        return out

    async def summarize_article(self, article: dict[str, Any]) -> str:
        """
        Produce a concise, engaging French summary of the news article that sparks discussion.
        Uses the full article text when available, with safety truncation to avoid overlong prompts.
        Ends with a short question to invite the user to react.
        """
        title = (article.get('title') or '').strip()
        text = (article.get('text') or '').strip()

        # Safety clamp for very long articles (keep ~6k chars)
        if text and len(text) > 6000:
            text = text[:6000] + '...'

        sys = (
            'You are a concise French-speaking assistant. '
            'Summarize the article in an interesting and clear way in no more than two sentences. '
            'Highlight the main issue and explain why it matters to the reader. '
            'Use a natural tone (not telegraphic). '
            'Start by greeting the user and saying you found an interesting article from radio canada to discuss. '
            'End with a short question that invites an opinion or reaction. '
            'Do not add bullet points, tags, or artificial headings.'
        )
        user = f'Titre: {title or "Sans titre"}\nTexte:\n{text or title or ""}'

        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': user},
        ]
        summary = (await self._resp_text(messages)).strip()

        return summary

    async def is_interested(self, text: str) -> bool:
        """
        Return True if the user is engaging (any non-declining reply),
        False if they indicate they don't want to chat now (e.g., 'no', 'pas maintenant', 'plus tard').
        Empty/whitespace-only replies are treated as not engaged.
        """
        if not text or not text.strip():
            return False

        # If not an explicit decline, treat any content as engagement.
        # Optionally, ask LLM for edge cases (comment out if you want zero LLM calls here).
        sys = (
            'Classify if the user is willing to chat right now. '
            'Reply ENGAGE if they are responding or willing to chat (any non-declining reply). '
            'Reply DECLINE if they indicate they do not want to chat now or prefer later. '
            'Output exactly ENGAGE or DECLINE.'
        )
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': text},
        ]
        out = (await self._resp_text(messages)).strip().upper()
        if out == 'DECLINE':
            return False
        return True


def get_time_appropriate_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Bonjour! Bon matin! Comment allez-vous aujourd'hui?"
    elif 12 <= hour < 18:
        return "Bonjour! Comment allez-vous aujourd'hui?"
    else:
        return 'Bonsoir! Comment allez-vous ce soir?'
