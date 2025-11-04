from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.db.news import NewsStore


class NewsTopicInput(BaseModel):
    source: str = Field(default='radio-canada', description='News source key')
    limit: int = Field(default=5, ge=1, le=5, description='How many titles to fetch (max 5)')
    include_text: bool = Field(
        default=False, description='If true, include a short snippet from the article text'
    )


class FetchArticleInput(BaseModel):
    index: int = Field(..., ge=1, description='1-based index from the numbered list (e.g., 1..5)')
    source: str = Field(default='radio-canada', description='News source key')
    limit: int = Field(
        default=5, ge=1, le=5, description='Should match the list size previously requested'
    )
    max_chars: int = Field(
        default=2000, ge=200, le=20000, description='Max characters of article text to return'
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
                snippet = (text[:100] + '…') if len(text) > 200 else text
                lines.append(f'{i}) {title}\n   {snippet}')
            else:
                lines.append(f'{i}) {title}')

        return '\n'.join(lines)

    @tool('fetch_news_article', args_schema=FetchArticleInput)
    async def fetch_news_article(
        index: int, source: str = 'radio-canada', limit: int = 5, max_chars: int = 2000
    ) -> str:
        """
        Fetch the full article for the Nth item from the recent list (1-based index).
        Returns JSON: {"id", "title", "link", "published", "text", "source"}.
        To be called after fetch_news_topics.
        """
        items = await news_store.recent_titles(limit=limit, source=source)
        item = items[index - 1]

        art = await news_store.get_article(int(item['id'])) or {}

        text = (art.get('text') or item.get('text') or '').strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + '…'

        payload = {
            'id': int(item['id']),
            'title': (art.get('title') or item.get('title') or '').strip(),
            'link': (art.get('link') or item.get('link') or '').strip(),
            'published': (art.get('published') or item.get('published') or ''),
            'text': text,
            'source': source,
        }
        import json as _json

        return _json.dumps(payload, ensure_ascii=False)

    return [fetch_news_topics, fetch_news_article]
