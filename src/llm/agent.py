import functools
from contextlib import contextmanager
from typing import Any, TypedDict

import mlflow
import numpy as np
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.db.news import NewsStore
from src.db.speaker import VoiceIdentifier
from src.llm.converse import ConversationService, get_time_appropriate_greeting
from src.llm.mistakes import MistakeAnalyzer
from src.llm.prompt import PromptManager
from src.llm.tools import build_tools

# Turn on auto tracing for Gemini
mlflow.langchain.autolog()
mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_experiment('blabin-development')


@contextmanager
def start_span(name: str, inputs: dict | None = None):
    class _Dummy:
        def set_outputs(self, *_args, **_kwargs):  # no-op
            pass

    yield _Dummy()


def trace_node(name: str):
    """
    Decorate a LangGraph node (async fn taking `state` and returning dict) to trace with MLflow.
    Captures light inputs and result keys; LLM calls inside are traced by mlflow.langchain.autolog().
    """

    def _decorator(fn):
        @functools.wraps(fn)
        async def _wrapper(state: '_ConvState') -> dict[str, Any]:
            # Keep inputs small to avoid leaking PII; just previews and flags
            inputs = {
                'node': name,
                'current_speaker_set': bool(state.get('current_speaker')),
                'awaiting_confirmation': bool(state.get('awaiting_confirmation')),
                'history_len': len(state.get('history', [])),
                'user_text_preview': (state.get('user_text') or '')[:120],
            }
            with start_span(name=name, inputs=inputs) as span:
                result = await fn(state)
                try:
                    span.set_outputs(
                        {
                            'result_keys': sorted(result.keys()),
                            'set_response': bool(result.get('response')),
                            'mistake_count': len(result.get('mistakes', []))
                            if isinstance(result.get('mistakes'), list)
                            else 0,
                        }
                    )
                except Exception:
                    pass
            return result

        return _wrapper

    return _decorator


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
    mistakes: list[dict]  #: extracted mistake records


class ConversationAgent:
    def __init__(
        self,
        api_key: str,
        voice_identifier: VoiceIdentifier,
        news_store: NewsStore,
        prompt_name: str = 'teacher_v1',
    ):
        self.current_speaker: str | None = None
        self.conversation_started: bool = False
        self.proposed_name: str | None = None
        self.awaiting_confirmation: bool = False
        pm = PromptManager()
        doc = pm.get_prompt(prompt_name, local_only=True)
        self.system_prompt = str(doc['system'])

        # LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

        self.mistake_analyzer = MistakeAnalyzer(self.llm, pm)
        self.conversation_service = ConversationService(self.llm)

        self.voice_identifier = voice_identifier
        self.news_store = news_store

        # Simple in-class memory for chat history
        self._history: list[dict[str, str]] = []
        self.last_mistakes: list[dict] = []  #: expose latest analysis

        # build our llm with tools
        self._tools = build_tools(news_store)
        self._llm_with_tools = self.llm.bind_tools(self._tools)  # NEW
        self._tool_map = {t.name: t for t in self._tools}  # NEW

        # Build a LangGraph for the conversation turn
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(_ConvState)

        @trace_node('identify_from_voice')
        async def identify_from_voice(state: _ConvState) -> dict[str, Any]:
            if state.get('awaiting_confirmation'):
                return {}
            if state.get('current_speaker'):
                return {}
            if state.get('proposed_name'):
                return {}

            seg = state.get('audio_array')
            if isinstance(seg, np.ndarray) and seg.size > 0:
                name, _ = self.voice_identifier.identify_speaker(seg)
                if name and name != 'unknown':
                    ask = await self.conversation_service.make_confirmation_request(
                        state.get('user_text', ''), name
                    )
                    return {
                        'proposed_name': name,
                        'awaiting_confirmation': True,
                        'response': ask,
                        'conversation_started': True,
                        'name_just_discovered': True,
                    }
            return {}

        @trace_node('confirm_identity')
        async def confirm_identity(state: _ConvState) -> dict[str, Any]:
            if state.get('current_speaker'):
                return {}
            if not state.get('awaiting_confirmation'):
                return {}
            if state.get('name_just_discovered'):
                return {}

            user_text = (state.get('user_text') or '').strip()
            proposed = state.get('proposed_name')

            # If user provides a name directly (text), accept it and greet contextually
            provided = await self.conversation_service.get_speaker_name(user_text)
            if provided and provided != 'unknown':
                greet = await self.conversation_service.make_greeting(
                    user_text, provided, returning=False
                )
                return {
                    'current_speaker': provided,
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': greet,
                    'conversation_started': True,
                }

            # Otherwise classify as yes/no
            try:
                decision = await self.conversation_service.is_confirmation(user_text)
            except Exception:
                decision = False

            if decision and proposed:
                greet = await self.conversation_service.make_greeting(
                    user_text, proposed, returning=True
                )
                return {
                    'current_speaker': proposed,
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': greet,
                    'conversation_started': True,
                }
            if decision is False:
                ask_name = await self.conversation_service.ask_for_name(
                    user_text, reason='user declined proposed name'
                )
                return {
                    'proposed_name': None,
                    'awaiting_confirmation': False,
                    'response': ask_name,
                    'conversation_started': True,
                }

            if not state.get('response'):
                clarify = await self.conversation_service.clarify_confirmation(
                    user_text, proposed or ''
                )
                return {
                    'response': clarify,
                    'conversation_started': True,
                }
            return {}

        @trace_node('extract_or_ask_name')
        async def extract_or_ask_name(state: _ConvState) -> dict[str, Any]:
            if state.get('current_speaker'):
                return {}
            if state.get('response'):
                return {}
            if state.get('awaiting_confirmation'):
                return {}

            user_text = (state.get('user_text') or '').strip()
            if not user_text:
                ask_name = await self.conversation_service.ask_for_name(
                    '', reason='no usable text in this turn'
                )
                return {
                    'response': ask_name,
                    'conversation_started': True,
                }

            extracted = await self.conversation_service.get_speaker_name(user_text)
            if extracted != 'unknown':
                greet = await self.conversation_service.make_greeting(
                    user_text, extracted, returning=False
                )
                return {
                    'current_speaker': extracted,
                    'response': greet,
                    'conversation_started': True,
                }

            state['system_prompt'] = (
                " At the end of your response, mention you don't know them, and need to ask their name."
            )
            state['current_speaker'] = 'unknown'
            response = await generate_response(state)
            return {
                'response': response['response'],
                'conversation_started': True,
            }

        @trace_node('analyze_mistakes')
        async def analyze_mistakes(state: _ConvState) -> dict[str, Any]:
            user_text = (state.get('user_text') or '').strip()
            if not user_text:
                return {'mistakes': []}
            if state.get('awaiting_confirmation') and not state.get('name_just_discovered'):
                return {'mistakes': []}

            try:
                records = await self.mistake_analyzer.analyze(user_text)
            except Exception:
                records = []
            return {'mistakes': records}

        @trace_node('generate_response')
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

            # Build a dynamic system prompt that includes topic policy and recent news titles
            dynamic_system = state['system_prompt']

            # Build conversation with system history  this user turn
            messages: list[dict[str, str]] = [{'role': 'system', 'content': dynamic_system}]
            messages.extend(state.get('history', []))
            messages.append({'role': 'user', 'content': user_text})

            # Let the LLM decide whether to call the tool; handle tool-call loop
            out_text = ''
            try:
                result = await self._llm_with_tools.ainvoke(messages)
                # If the model asks to call tools, execute them and send results back
                max_tool_rounds = 2
                rounds = 0
                while (
                    isinstance(result, AIMessage)
                    and getattr(result, 'tool_calls', None)
                    and rounds < max_tool_rounds
                ):
                    rounds += 1
                    for call in result.tool_calls:
                        name = call.get('name')
                        args = call.get('args') or {}
                        call_id = call.get('id') or ''
                        tool_impl = self._tool_map[name]
                        try:
                            tool_output = await tool_impl.ainvoke(args)
                        except Exception as e:
                            tool_output = f'Erreur outil `{name}`: {e}'
                        messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))

                    # Ask the model to continue after tool outputs
                    result = await self._llm_with_tools.ainvoke(messages)
                    print(result)
                out_text = str(getattr(result, 'content', str(result))).strip()
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
        g.add_node('analyze_mistakes', analyze_mistakes)
        g.add_node('generate_response', generate_response)

        # Flow: voice-ID -> confirm -> text fallback -> chat
        g.add_edge(START, 'identify_from_voice')
        g.add_edge('identify_from_voice', 'confirm_identity')
        g.add_edge('confirm_identity', 'extract_or_ask_name')
        g.add_edge('extract_or_ask_name', 'analyze_mistakes')
        g.add_edge('analyze_mistakes', 'generate_response')
        g.add_edge('generate_response', END)

        return g.compile()

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
            'awaiting_confirmation': self.awaiting_confirmation,
            'proposed_name': self.proposed_name,
            'response': None,
            'mistakes': [],
        }

        # Run the graph for this turn
        result = await self._graph.ainvoke(state)

        # Persist updates
        self.current_speaker = result.get('current_speaker', self.current_speaker)
        self.proposed_name = result.get('proposed_name', self.proposed_name)
        self.awaiting_confirmation = result.get('awaiting_confirmation', self.awaiting_confirmation)
        self.conversation_started = bool(
            result.get('conversation_started', self.conversation_started)
        )

        self.last_mistakes = result.get('mistakes', []) or []

        # Update history
        if text:
            self._history.append({'role': 'user', 'content': text})
        if resp := result.get('response'):
            self._history.append({'role': 'assistant', 'content': resp})

        return result.get('response', '')

    def reset_conversation(self) -> None:
        self.conversation_started = False
        self._history = []
        self.last_mistakes = []

    def say_hello(self) -> str:
        return get_time_appropriate_greeting()

    def should_speak_response(self, response: str) -> bool:
        if not response or response.startswith('Error') or len(response.split()) > 200:
            return False
        return True
