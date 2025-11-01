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


def get_time_appropriate_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Bonjour! Bon matin! Comment allez-vous aujourd'hui?"
    elif 12 <= hour < 18:
        return "Bonjour! Comment allez-vous aujourd'hui?"
    else:
        return 'Bonsoir! Comment allez-vous ce soir?'
