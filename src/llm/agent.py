import re
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.db.speaker import VoiceIdentifier
from src.llm.client import AsyncLLMClient
from src.llm.prompt import PromptManager


class _ConvState(TypedDict, total=True):
    # Persistent
    system_prompt: str
    current_speaker: str | None
    conversation_started: bool
    history: list[dict[str, str]]  # [{'role': 'user'|'assistant', 'content': str}, ...]
    proposed_name: str | None  # name proposed by voice ID
    awaiting_confirmation: bool  # awaiting yes/no on proposed_name
    name_just_discovered: bool  # whether we just learned a name this turn

    # Per-turn input
    user_text: str
    audio_bytes: bytes | None  # optional audio
    audio_array: np.ndarray | None  # optional audio

    # Per-turn output
    response: str | None


class ConversationAgent:
    def __init__(
        self,
        api_key: str,
        llm_client: AsyncLLMClient,
        voice_identifier: VoiceIdentifier,
        prompt_name: str = 'teacher_v1',
    ):
        self.current_speaker: str | None = None
        self.conversation_started: bool = False
        self.proposed_name: str | None = None
        self.awaiting_confirmation: bool = False
        self.system_prompt = (
            'You are a helpful language learning assistant, please respond in the language in '
            'which you are addressed. Correct any grammatical errors made by the speaker.'
        )
        try:
            pm = PromptManager()
            doc = pm.get_prompt(prompt_name, local_only=True)
            if doc and isinstance(doc, dict):
                self.system_prompt = str(doc.get('system', doc.get('prompt', self.system_prompt)))
        except Exception:
            pass

        # LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

        self.voice_identifier = voice_identifier
        self.llm_client = llm_client

        # Simple in-class memory for chat history
        self._history: list[dict[str, str]] = []

        # Build a LangGraph for the conversation turn
        self._graph = self._build_graph()

    def _extract_name_from_text(self, text: str) -> str | None:
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
            # awaiting confirmation for the voice, don't try again
            if self.awaiting_confirmation:
                return {}

            if state.get('current_speaker'):
                return {}

            # if the name was proposed in a prior turn, we don't try to identify
            # again until we have determined that this is NOT their name
            if state.get('proposed_name'):
                return {}

            seg = state.get('audio_array')
            if self.voice_identifier is None:
                return {}

            if isinstance(seg, np.ndarray):
                if seg.size > 0:
                    try:
                        name, _ = self.voice_identifier.identify_speaker(seg)
                    except Exception:
                        name = 'unknown'

                    if name and name != 'unknown':
                        ask = f'Je connais votre voix! Est-ce que vous êtes bien {name} ?'
                        return {
                            'proposed_name': name,
                            'awaiting_confirmation': True,
                            'response': ask,
                            'conversation_started': True,
                            'name_just_discovered': True,
                        }

            return {}

        async def confirm_identity(state: _ConvState) -> dict[str, Any]:
            """
            If awaiting confirmation, interpret the user's text as YES/NO, or accept a provided name.
            """
            if state.get('current_speaker'):
                return {}

            if not state.get('awaiting_confirmation'):
                return {}

            if state.get('name_just_discovered'):
                return {}

            user_text = (state.get('user_text') or '').strip()
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

            # Otherwise classify as yes/no via LLM client
            try:
                decision = await self.llm_client.is_confirmation(user_text)
            except Exception:
                decision = False

            if decision and proposed:
                greet = f'Ravi de vous revoir, {proposed} !'
                return {
                    'current_speaker': proposed,
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': greet,
                    'conversation_started': True,
                }
            if decision is False:
                return {
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': "D'accord. Comment vous appelez-vous ?",
                    'conversation_started': True,
                }

            # Not sure yet; gently ask again if we haven't replied this turn
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
            if state.get('response'):
                return {}
            if state.get('awaiting_confirmation'):
                return {}

            user_text = (state.get('user_text') or '').strip()
            if not user_text:
                return {
                    'response': 'Je ne crois pas vous connaître. Comment vous appelez-vous ?',
                    'conversation_started': True,
                }

            extracted = self._extract_name_from_text(user_text)

            if extracted:
                greet = f'Enchanté(e), {extracted} !'
                return {
                    'current_speaker': extracted,
                    'response': greet,
                    'conversation_started': True,
                }

            # respond to what they're saying, then ask their name
            state['system_prompt'] += (
                "At the end of your response, mention you don't know them, and need to ask their name"
            )
            state['current_speaker'] = 'unknown'  # temporary to allow response generation

            response = await generate_response(state)
            return {
                'response': response['response'],
                'conversation_started': True,
            }

        async def generate_response(state: _ConvState) -> dict[str, Any]:
            """
            If we already produced a response (e.g., confirm/greet/ask-name), do not overwrite.
            Otherwise, continue with the tutoring conversation.
            """
            if state.get('response'):
                return {}

            user_text = (state.get('user_text') or '').strip()
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

    def set_speaker(self, speaker_name: str | None) -> None:
        self.current_speaker = speaker_name

    async def process_message(
        self,
        text: str,
        audio_bytes: bytes | None = None,
        audio_array: np.ndarray | None = None,
    ) -> str:
        # Prepare state for this turn (audio may be None in text modes)
        state: _ConvState = {
            'system_prompt': self.system_prompt,
            'current_speaker': self.current_speaker,
            'conversation_started': self.conversation_started,
            'name_just_discovered': False,
            'history': self._history,
            'user_text': text,
            'audio_bytes': audio_bytes,
            'audio_array': audio_array,
            'awaiting_confirmation': False,
            'proposed_name': None,
            'response': None,
        }

        # Seed pending confirmation/proposed_name from last assistant turn if we asked it
        if self._history:
            last = self._history[-1]
            if last.get('role') == 'assistant' and 'êtes bien' in last.get('content', ''):
                m = re.search(r'êtes bien\s+([A-Za-zÀ-ÿ\-]{2,})', last['content'])
                if m:
                    state['proposed_name'] = m.group(1)
                    state['awaiting_confirmation'] = True

        # Run the graph for this turn
        result = await self._graph.ainvoke(state)

        # Persist updates
        self.current_speaker = result.get('current_speaker', self.current_speaker)
        self.proposed_name = result.get('proposed_name', self.proposed_name)
        self.awaiting_confirmation = result.get('awaiting_confirmation', self.awaiting_confirmation)
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
