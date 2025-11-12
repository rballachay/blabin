from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.context.search import synthesize_search_answer
from src.db.news import NewsStore


class NewsTopicInput(BaseModel):
    limit: int = Field(default=5, ge=1, le=5, description='How many titles to fetch (max 5)')
    include_text: bool = Field(
        default=False, description='If true, include a short snippet from the article text'
    )


class FetchArticleInput(BaseModel):
    index: int = Field(..., ge=1, description='1-based index from the numbered list (e.g., 1..5)')
    limit: int = Field(
        default=5, ge=1, le=5, description='Should match the list size previously requested'
    )
    max_chars: int = Field(
        default=2000, ge=200, le=20000, description='Max characters of article text to return'
    )


def build_tools(news_store: NewsStore, search_client: TavilyClient) -> list[Any]:
    """
    Define tools the LLM can call. Includes a news topic fetcher that returns a policy header
    followed by a numbered list of titles (and optional snippets).
    """

    @tool('fetch_news_topics', args_schema=NewsTopicInput)
    async def fetch_news_topics(limit: int = 5, include_text: bool = True) -> str:
        """
        Fetch recent news titles for conversation topics. Returns:
        - A short policy reminder
        - A numbered list of titles (and optional snippets)
        """

        items = await news_store.recent_titles(limit=limit)

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
    async def fetch_news_article(index: int, limit: int = 5, max_chars: int = 2000) -> str:
        """
        Fetch the full article for the Nth item from the recent list (1-based index).
        Returns JSON: {"id", "title", "link", "published", "text", "source"}.
        To be called after fetch_news_topics.
        """
        items = await news_store.recent_titles(limit=limit)
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
        }
        import json as _json

        return _json.dumps(payload, ensure_ascii=False)

    @tool('search_web', return_direct=False)
    def search_web(query: str, max_results: int = 5, deep: bool = True) -> dict[str, Any]:
        """
        Topical web search using Tavily.
        Inputs:
        - query: search query
        - max_results: max number of results (default 5)
        - deep: use deeper search (slower but better coverage)
        Returns JSON with fields:
        {
            "answer": str | null,
            "citations": [{"title": str, "url": str}]  // up to k
        }
        """
        if not query or not query.strip():
            return {'answer': None, 'citations': []}

        resp = search_client.search(
            query, include_answer=True, max_results=max_results, deep=deep
        )  # include_answer helps populate 'answer'

        answer, citations = synthesize_search_answer(resp)

        return {
            'answer': answer,
            'citations': citations,
        }

    @tool('fetch_url', return_direct=False)
    def fetch_url(url: str, max_chars: int = 12000) -> dict[str, Any]:
        """
        Fetch and extract main text from a URL. To be called after search_web if
        more information is needed from the sources.
        Inputs:
        - url: page URL (http/https)
        - max_chars: truncate extracted text to this many characters
        Returns JSON:
        {
            "url": str,
            "title": str | null,
            "text": str
        }
        """
        # Tavily returns: {"results": [{"url", "title", "raw_content", "images": [...] }], ...}
        resp = search_client.extract([url])

        title: str | None = None
        text: str = ''

        results = (resp or {}).get('results') or []
        if results:
            item = results[0] or {}
            title = (item.get('title') or '').strip() or None
            text = (item.get('raw_content') or '').strip()

        if max_chars and len(text) > max_chars:
            text = text[: int(max_chars)] + '…'

        return {'url': url, 'title': title, 'text': text}

    return [fetch_news_topics, fetch_news_article, search_web, fetch_url]
