from datetime import datetime

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from src.llm.prompt import PromptManager


class ConversationAgent:
    def __init__(self, api_key: str, prompt_name: str | None = 'teacher_v1'):
        self.current_speaker: None | str = None
        self.conversation_started: bool = False
        self.system_prompt = (
            'You are a helpful language learning assistant, please respond in the language in '
            'which you are addressed. Correct any grammatical errors made by the speaker.'
        )
        try:
            pm = PromptManager()
            if prompt_name:
                doc = pm.get_prompt(prompt_name, local_only=True)
                if doc and isinstance(doc, dict):
                    # allow either a top-level 'system' string or a 'system' key in metadata
                    self.system_prompt = str(
                        doc.get('system', doc.get('prompt', self.system_prompt))
                    )
        except Exception:
            # keep default system prompt on any error
            pass

        # initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

        # create agent with simple in-memory checkpointer (existing behaviour preserved)
        self.agent_executor = create_agent(self.llm, checkpointer=InMemorySaver())  # type: ignore

    def say_hello(self) -> str:
        """Return a time-appropriate greeting in French"""
        return self._get_time_appropriate_greeting()

    def _get_time_appropriate_greeting(self) -> str:
        """Return appropriate French greeting based on time of day"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Bonjour! Bon matin! Comment allez-vous aujourd'hui?"
        elif 12 <= hour < 18:
            return "Bonjour! Comment allez-vous aujourd'hui?"
        else:
            return 'Bonsoir! Comment allez-vous ce soir?'

    def set_speaker(self, speaker_name: None | str) -> None:
        """Update the current speaker"""
        self.current_speaker = speaker_name

    async def _extract_name_from_response(self, text: str) -> tuple[bool, str | None]:
        """Use LLM to detect if speaker provided their name and extract it."""
        prompt = [
            {
                'role': 'system',
                'content': (
                    'You are a French conversation assistant. '
                    'Analyze the text and extract any name the speaker provides. '
                    "Respond with exactly 'NO_NAME' if no name is given, "
                    'or with just the name if found.'
                ),
            },
            {'role': 'user', 'content': text},
        ]

        response = await self.llm.ainvoke(prompt)
        result = str(response).strip()

        if result == 'NO_NAME':
            return False, None
        return True, result

    async def process_message(self, text: str) -> str:
        """Process a message through the agent and return assistant content"""
        # Build conversation context with speaker awareness
        conversation = [{'role': 'system', 'content': self.system_prompt}]

        # Add initial greeting if conversation just starting
        if not self.conversation_started:
            greeting = ''
            if self.current_speaker:
                greeting += f' Ravi de vous revoir, {self.current_speaker}!'
            else:
                greeting += ' Comment vous appelez-vous?'
            conversation.append({'role': 'assistant', 'content': greeting})
            self.conversation_started = True

        # Handle name detection using LLM
        if not self.current_speaker:
            has_name, name = await self._extract_name_from_response(text)
            if has_name and name:
                self.current_speaker = name
                conversation.append(
                    {
                        'role': 'assistant',
                        'content': f"Ah, enchanté(e) {name}! C'est un plaisir de vous rencontrer.",
                    }
                )
            else:
                # Keep asking for name if not provided
                conversation.append(
                    {
                        'role': 'assistant',
                        'content': "Je ne suis pas sûr d'avoir compris votre nom. Pourriez-vous me le redire s'il vous plaît?",
                    }
                )
                return conversation[-1]['content']

        conversation.append({'role': 'user', 'content': text})

        # Run agent with full conversation context
        response = await self.agent_executor.ainvoke(
            {'messages': conversation},
            {'configurable': {'thread_id': '1'}},
        )

        try:
            return response['messages'][-1].content
        except Exception:
            return str(response)

    def reset_conversation(self) -> None:
        """Reset the conversation state"""
        self.conversation_started = False

    def should_speak_response(self, response: str) -> bool:
        """Determine if response should be spoken"""
        # simple heuristic: avoid speaking very long or error responses
        if not response or response.startswith('Error') or len(response.split()) > 200:
            return False
        return True
