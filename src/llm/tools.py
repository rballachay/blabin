from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.db.news import NewsStore


class NewsTopicInput(BaseModel):
    source: str = Field(default='radio-canada', description='News source key')
    limit: int = Field(default=5, ge=1, le=5, description='How many titles to fetch (max 5)')
    include_text: bool = Field(
        default=False, description='If true, include a short snippet from the article text'
    )


def build_tools(news_store: NewsStore):
    """
    Define tools the LLM can call. Includes a news topic fetcher that returns a policy header
    followed by a numbered list of titles (and optional snippets).
    """

    @tool('fetch_news_topics', args_schema=NewsTopicInput)
    async def fetch_news_topics(
        source: str = 'radio-canada', limit: int = 5, include_text: bool = True
    ) -> str:
        """
        Fetch recent news titles for conversation topics. Returns:
        - A short policy reminder
        - A numbered list of titles (and optional snippets)
        """

        items = await news_store.recent_titles(limit=limit, source=source)

        lines: list[str] = []
        for i, it in enumerate(items, start=1):
            title = str(it.get('title', '')).strip()
            if include_text:
                art = await news_store.get_article(int(it['id'])) or {}
                text = (art.get('text') or '').strip()
                snippet = (text[:200] + '…') if len(text) > 200 else text
                lines.append(f'{i}) {title}\n   {snippet}')
            else:
                lines.append(f'{i}) {title}')

        return '\n'.join(lines)

    return [fetch_news_topics]
