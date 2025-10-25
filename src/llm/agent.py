from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from src.llm.prompt import PromptManager


class ConversationAgent:
    def __init__(self, api_key: str, prompt_name: str | None = 'french_teacher'):
        # load prompt if available (local preferred)
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

    async def process_message(self, text: str) -> str:
        """Process a message through the agent and return assistant content"""
        conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': text},
        ]

        # Run agent (preserve existing call signature)
        response = await self.agent_executor.ainvoke(
            {'messages': conversation},
            {'configurable': {'thread_id': '1'}},
        )

        # response format expected to include messages list; keep original behaviour
        try:
            return response['messages'][-1].content
        except Exception:
            # fallback: return stringified response
            return str(response)

    def should_speak_response(self, response: str) -> bool:
        """Determine if response should be spoken"""
        # simple heuristic: avoid speaking very long or error responses
        if not response or response.startswith('Error') or len(response.split()) > 200:
            return False
        return True
