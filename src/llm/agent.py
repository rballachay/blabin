import re
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.db.speaker import VoiceIdentifier
from src.llm.prompt import PromptManager


class _ConvState(TypedDict, total=False):
    # Persistent
    system_prompt: str
    current_speaker: str | None
    conversation_started: bool
    history: list[dict[str, str]]  # [{'role': 'user'|'assistant', 'content': str}, ...]
    proposed_name: str | None  # name proposed by voice ID
    awaiting_confirmation: bool  # awaiting yes/no on proposed_name

    # Per-turn input
    user_text: str
    audio_segment: np.ndarray | None  # audio segment at 16 kHz float32

    # Per-turn output
    response: str


class ConversationAgent:
    def __init__(
        self,
        api_key: str,
        prompt_name: str = 'teacher_v1',
        voice_identifier: VoiceIdentifier | None = None,
    ):
        self.current_speaker: None | str = None
        self.conversation_started: bool = False
        self.system_prompt = (
            'You are a helpful language learning assistant, please respond in the language in '
            'which you are addressed. Correct any grammatical errors made by the speaker.'
        )
        pm = PromptManager()
        doc = pm.get_prompt(prompt_name, local_only=True)
        if doc and isinstance(doc, dict):
            self.system_prompt = str(doc.get('system', doc.get('prompt', self.system_prompt)))

        # LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

        # Optional voice identifier for voice-based recognition
        self.voice_identifier = voice_identifier

        # Simple in-class memory for chat history
        self._history: list[dict[str, str]] = []

        # Build a LangGraph for the conversation turn
        self._graph = self._build_graph()

    async def _classify_confirmation(self, text: str) -> str:
        """
        Classify user text as YES / NO / NONE.
        """
        txt = (text or '').strip()
        if not txt:
            return 'NONE'
        # quick heuristic first
        low = txt.lower()
        if any(w in low for w in ['oui', "c'est bien", 'exact', 'tout à fait', "oui c'est moi"]):
            return 'YES'
        if any(w in low for w in ['non', 'pas moi', "ce n'est pas moi", 'pas du tout']):
            return 'NO'

        prompt = [
            {
                'role': 'system',
                'content': (
                    'You are a French assistant. Classify the user reply as YES, NO, or NONE. '
                    'Respond with exactly one token: YES or NO or NONE.'
                ),
            },
            {'role': 'user', 'content': txt},
        ]
        try:
            res = await self.llm.ainvoke(prompt)
            out = str(getattr(res, 'content', str(res))).strip().upper()
            if out in ('YES', 'NO', 'NONE'):
                return out
        except Exception:
            pass
        return 'NONE'

    def _extract_name_from_text(self, text: str) -> str | None:
        """
        Minimal French name extraction: "je m'appelle X", "moi c'est X", or a lone token.
        """
        if not text:
            return None
        m = re.search(r"(?:je m'appelle|moi c['’]est)\s+([A-Za-zÀ-ÿ\-]{2,})", text, flags=re.I)
        if m:
            return m.group(1).strip()
        tokens = re.findall(r'[A-Za-zÀ-ÿ\-]{2,}', text)
        if 1 <= len(tokens) <= 3:
            return tokens[-1].capitalize()
        return None

    def _build_graph(self):
        g = StateGraph(_ConvState)

        async def identify_from_voice(state: _ConvState) -> dict[str, Any]:
            """
            If an audio segment is provided and no current speaker yet, try voice identification.
            If a candidate is found, ask for confirmation rather than setting the speaker directly.
            """
            if state.get('current_speaker'):
                return {}

            seg = state.get('audio_segment')
            if self.voice_identifier is None:
                return {}

            if isinstance(seg, np.ndarray) and seg.size > 0:
                try:
                    name, _score = self.voice_identifier.identify_speaker(seg)
                except Exception:
                    name = 'unknown'

                if name and name != 'unknown':
                    # Ask for confirmation of the proposed name
                    ask = f'Est-ce que vous êtes bien {name} ?'
                    return {
                        'proposed_name': name,
                        'awaiting_confirmation': True,
                        'response': ask,
                        'conversation_started': True,
                    }

            # No audio or unknown => do nothing here; fall through to text handler
            return {}

        async def confirm_identity(state: _ConvState) -> dict[str, Any]:
            """
            If awaiting confirmation, interpret the user's text as YES/NO, or accept a provided name.
            """
            if state.get('current_speaker'):
                return {}

            if not state.get('awaiting_confirmation'):
                return {}

            user_text = state.get('user_text', '').strip()
            proposed = state.get('proposed_name')

            # If user directly provides a different name, accept it
            provided = self._extract_name_from_text(user_text)
            if provided:
                greet = f'Ravi de vous rencontrer, {provided} !'
                return {
                    'current_speaker': provided,
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': greet,
                    'conversation_started': True,
                }

            # Otherwise classify as yes/no/none
            decision = await self._classify_confirmation(user_text)
            if decision == 'YES' and proposed:
                # Returning user confirmed
                greet = f'Ravi de vous revoir, {proposed} !'
                return {
                    'current_speaker': proposed,
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': greet,
                    'conversation_started': True,
                }
            if decision == 'NO':
                # Ask for their name explicitly
                return {
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': "D'accord. Comment vous appelez-vous ?",
                    'conversation_started': True,
                }

            # Not sure yet; ask again without overwriting any prior response
            if not state.get('response'):
                return {
                    'response': "Pouvez-vous confirmer, s'il vous plaît ? Répondez par oui ou non.",
                    'conversation_started': True,
                }
            return {}

        async def extract_or_ask_name(state: _ConvState) -> dict[str, Any]:
            """
            If speaker still unknown after voice step/confirmation, try extracting a name from text.
            If not present, ask for their name (text-only fallback).
            """
            if state.get('current_speaker'):
                return {}
            # If we already produced a response this turn (e.g. confirmation prompt), keep it
            if state.get('response'):
                return {}
            # If still awaiting confirmation, don't ask for name again here
            if state.get('awaiting_confirmation'):
                return {}

            user_text = state.get('user_text', '').strip()
            if not user_text:
                return {
                    'response': 'Je ne crois pas vous connaître. Comment vous appelez-vous ?',
                    'conversation_started': True,
                }

            extracted = self._extract_name_from_text(user_text) or 'NO_NAME'
            if extracted != 'NO_NAME':
                greet = f'Enchanté(e), {extracted} !'
                return {
                    'current_speaker': extracted,
                    'response': greet,
                    'conversation_started': True,
                }

            return {
                'response': "Je ne suis pas sûr d'avoir compris votre nom. "
                "Pourriez-vous me le redire s'il vous plaît ?",
                'conversation_started': True,
            }

        async def generate_response(state: _ConvState) -> dict[str, Any]:
            """
            If we already produced a response (e.g., confirm/greet/ask-name), do not overwrite.
            Otherwise, continue with the tutoring conversation.
            """
            if state.get('response'):
                return {}

            user_text = state.get('user_text', '').strip()
            if not user_text:
                return {}

            if not state.get('current_speaker'):
                return {}

            # Build conversation with system + history + this user turn
            messages: list[dict[str, str]] = [{'role': 'system', 'content': state['system_prompt']}]
            messages.extend(state.get('history', []))
            messages.append({'role': 'user', 'content': user_text})

            try:
                resp = await self.llm.ainvoke(messages)
                out_text = str(getattr(resp, 'content', str(resp))).strip()
            except Exception:
                out_text = "Désolé, j'ai rencontré un problème en générant une réponse."

            # Append assistant reply to history
            new_hist = list(state.get('history', []))
            new_hist.append({'role': 'user', 'content': user_text})
            new_hist.append({'role': 'assistant', 'content': out_text})

            return {
                'response': out_text,
                'history': new_hist,
                'conversation_started': True,
            }

        # Nodes
        g.add_node('identify_from_voice', identify_from_voice)
        g.add_node('confirm_identity', confirm_identity)
        g.add_node('extract_or_ask_name', extract_or_ask_name)
        g.add_node('generate_response', generate_response)

        # Flow: voice-ID -> confirm -> text fallback -> chat
        g.add_edge(START, 'identify_from_voice')
        g.add_edge('identify_from_voice', 'confirm_identity')
        g.add_edge('confirm_identity', 'extract_or_ask_name')
        g.add_edge('extract_or_ask_name', 'generate_response')
        g.add_edge('generate_response', END)

        return g.compile()

    def _get_time_appropriate_greeting(self) -> str:
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Bonjour! Bon matin! Comment allez-vous aujourd'hui?"
        elif 12 <= hour < 18:
            return "Bonjour! Comment allez-vous aujourd'hui?"
        else:
            return 'Bonsoir! Comment allez-vous ce soir?'

    def set_speaker(self, speaker_name: None | str) -> None:
        self.current_speaker = speaker_name

    async def process_message(self, text: str, audio_segment: np.ndarray | None = None) -> str:
        # Prepare state for this turn (audio included)
        state: _ConvState = {
            'system_prompt': self.system_prompt,
            'current_speaker': self.current_speaker,
            'conversation_started': self.conversation_started,
            'history': self._history,
            'user_text': text,
            'audio_segment': audio_segment,
            # keep any pending confirmation across turns
            'awaiting_confirmation': False,
            'proposed_name': None,
        }

        # Seed pending confirmation/proposed_name from last assistant turn if we asked it
        # Pull from last history message if it contained a confirm question with a name.
        # Lightweight approach: if last assistant asked "Est-ce que vous êtes bien X ?"
        if self._history:
            last_assistant = self._history[-1] if self._history[-1]['role'] == 'assistant' else None
            if last_assistant and 'êtes bien' in last_assistant['content']:
                # Try to extract the proposed name from the question
                m = re.search(r'êtes bien\s+([A-Za-zÀ-ÿ\-]{2,})', last_assistant['content'])
                if m:
                    state['proposed_name'] = m.group(1)
                    state['awaiting_confirmation'] = True

        # Run the graph for this turn
        result = await self._graph.ainvoke(state)

        # Persist updates
        self.current_speaker = result.get('current_speaker', self.current_speaker)
        self.conversation_started = bool(
            result.get('conversation_started', self.conversation_started)
        )

        # Update history
        if text:
            self._history.append({'role': 'user', 'content': text})
        if resp := result.get('response'):
            self._history.append({'role': 'assistant', 'content': resp})

        return result.get('response', '')

    def reset_conversation(self) -> None:
        self.conversation_started = False
        self._history = []

    def say_hello(self) -> str:
        return self._get_time_appropriate_greeting()

    def should_speak_response(self, response: str) -> bool:
        if not response or response.startswith('Error') or len(response.split()) > 200:
            return False
        return True
