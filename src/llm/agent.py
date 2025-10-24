from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver


class ConversationAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

        self.agent_executor = create_agent(self.llm, checkpointer=InMemorySaver())

    async def process_message(self, text: str) -> str:
        """Process a message through the agent and update conversation history"""

        conversation = [
            {
                'role': 'system',
                'content': 'You are a helpful language learning assistant, please respond in the language in which you are addressed. Correct any grammatical errors made by the speaker',
            },
            {'role': 'user', 'content': text},
        ]

        # Run agent
        response = await self.agent_executor.ainvoke(
            {'messages': conversation},
            {'configurable': {'thread_id': '1'}},
        )
        return response['messages'][-1].content

    def should_speak_response(self, response: str) -> bool:
        """Determine if response should be spoken"""
        # Add logic here to determine if response should be spoken
        return True
