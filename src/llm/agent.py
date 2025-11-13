import functools
from contextlib import contextmanager
from typing import Any, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

from src.context.email import EmailClient
from src.db.mistakes import MistakeStore
from src.db.news import NewsStore
from src.db.speaker import VoiceIdentifier
from src.llm.converse import ConversationService, get_time_appropriate_greeting
from src.llm.prompt import PromptManager
from src.llm.tools import build_tools


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
    lc_messages: list


class ConversationAgent:
    def __init__(
        self,
        api_key: str,
        voice_identifier: VoiceIdentifier,
        news_store: NewsStore,
        prompt_manager: PromptManager,
        search_client: TavilyClient,
        mistake_store: MistakeStore,
        email_client: EmailClient,
        prompt_name: str = 'teacher_v1',
    ):
        self.current_speaker: str | None = None
        self.conversation_started: bool = False
        self.proposed_name: str | None = None
        self.awaiting_confirmation: bool = False
        doc = prompt_manager.get_prompt(prompt_name, local_only=True)
        self.system_prompt = str(doc['system'])

        # LLM (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

        self.conversation_service = ConversationService(self.llm)
        self.email_client = email_client

        self.voice_identifier = voice_identifier

        # Simple in-class memory for chat history
        self._history: list[dict[str, str]] = []

        # build our llm with tools
        self._tools = build_tools(
            lambda: self.current_speaker,
            news_store,
            search_client,
            mistake_store,
            self.conversation_service,
            self.email_client,
        )
        self._llm_with_tools = self.llm.bind_tools(self._tools)
        self._tool_node = ToolNode(self._tools)

        # Build a LangGraph for the conversation turn
        self._graph = self._build_graph()

        self.shutdown = False

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
                try:
                    self.voice_identifier.ensure_exists(provided)
                except Exception:
                    pass
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
            decision = await self.conversation_service.is_confirmation(user_text)

            # ...existing yes/no branch...
            if decision and proposed:
                try:
                    self.voice_identifier.ensure_exists(proposed)
                except Exception:
                    pass
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

            extracted = await self.conversation_service.get_speaker_name(user_text)
            if extracted != 'unknown':
                self.voice_identifier.ensure_exists(extracted)
                greet = await self.conversation_service.make_greeting(
                    user_text, extracted, returning=False
                )
                return {
                    'current_speaker': extracted,
                    'response': greet,
                    'conversation_started': True,
                }

            msgs = [
                {
                    'role': 'system',
                    'content': state['system_prompt']
                    + "\nAt the end of your response, mention you don't know them, and need to ask their name.",
                },
                {'role': 'user', 'content': user_text},
            ]
            ai = await self.llm.ainvoke(msgs)
            if isinstance(ai.content, str):
                out_text = ai.content.strip()
            else:
                out_text = str(ai.content).strip()

            return {
                'response': out_text,
                'conversation_started': True,
            }

        # Define the conditional edge that determines whether to continue or not
        async def call_tool(state: _ConvState) -> dict[str, Any]:
            """
            Execute tool calls requested by the last AIMessage and append ToolMessages to lc_messages.
            """
            lc_messages = list(state.get('lc_messages') or [])
            if not lc_messages or not isinstance(lc_messages[-1], AIMessage):
                return {}
            try:
                out = await self._tool_node.ainvoke({'messages': [lc_messages[-1]]})
                tool_msgs = out.get('messages', [])
            except Exception as e:
                tool_msgs = [ToolMessage(content=f'Erreur outil: {e}', tool_call_id='')]
            lc_messages.extend(tool_msgs)
            return {'lc_messages': lc_messages}

        @trace_node('generate_response')
        async def generate_response(state: _ConvState) -> dict[str, Any]:
            """
            LLM step:
            - First call: build system+history+user as LC messages.
            - Loop calls: reuse lc_messages (with ToolMessages) and ask the LLM again.
            - If no tool_calls are returned, finalize response and update history.
            """
            if state.get('response'):
                return {}

            user_text = (state.get('user_text') or '').strip()
            if not user_text or not state.get('current_speaker'):
                return {}

            # Reuse message list if coming back from tool execution; else build fresh
            lc_messages = list(state.get('lc_messages') or [])
            if not lc_messages:
                lc_messages.append(SystemMessage(content=state['system_prompt']))
                for m in state.get('history', []):
                    role = (m.get('role') or '').strip()
                    content = m.get('content') or ''
                    if role == 'user':
                        lc_messages.append(HumanMessage(content=content))
                    else:
                        lc_messages.append(AIMessage(content=content))
                lc_messages.append(HumanMessage(content=user_text))

            # One LLM step with tools advertised
            ai = await self._llm_with_tools.ainvoke(lc_messages)

            lc_messages.append(ai)

            # If LLM requested tools, return lc_messages and let conditional routing continue
            if isinstance(ai, AIMessage) and getattr(ai, 'tool_calls', None):
                return {'lc_messages': lc_messages}

            # Check if output is list or a string
            if isinstance(ai.content, str):
                out_text = ai.content.strip()
            elif isinstance(ai.content, list) and len(ai.content) > 0:
                # Handle list content - get first element
                first_item = ai.content[0]
                if isinstance(first_item, dict) and 'text' in first_item:
                    out_text = str(first_item['text']).strip()
                else:
                    out_text = str(first_item).strip()
            else:
                out_text = str(ai.content).strip()

            # our prompt may request END_SESSION to finish
            if out_text == 'END_SESSION':
                out_text = 'Ok, merci pour cette session ! À la prochaine fois.'
                self.shutdown = True

            # Otherwise finalize: take content as the assistant reply
            new_hist = list(state.get('history', []))
            new_hist.append({'role': 'user', 'content': user_text})
            if out_text:
                new_hist.append({'role': 'assistant', 'content': out_text})

            return {
                'response': out_text,
                'history': new_hist,
                'conversation_started': True,
                'lc_messages': [],  # reset for next turn
            }

        # Nodes
        g.add_node('identify_from_voice', identify_from_voice)
        g.add_node('confirm_identity', confirm_identity)
        g.add_node('extract_or_ask_name', extract_or_ask_name)
        g.add_node('generate_response', generate_response)
        g.add_node('call_tool', call_tool)

        # Edges
        g.set_entry_point('identify_from_voice')
        g.add_edge('identify_from_voice', 'confirm_identity')
        g.add_edge('confirm_identity', 'extract_or_ask_name')
        g.add_edge('extract_or_ask_name', 'generate_response')

        # Routing helpers
        def should_continue(state: _ConvState) -> str:
            msgs = state.get('lc_messages') or []
            last = msgs[-1] if msgs else None
            if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
                return 'continue'
            return 'end'

        g.add_conditional_edges(
            'generate_response', should_continue, {'continue': 'call_tool', 'end': END}
        )
        g.add_edge('call_tool', 'generate_response')

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
            'lc_messages': [],
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
        return get_time_appropriate_greeting()

    def should_speak_response(self, response: str) -> bool:
        if not response or response.startswith('Error') or len(response.split()) > 200:
            return False
        return True
