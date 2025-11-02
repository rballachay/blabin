from __future__ import annotations

from dataclasses import dataclass

import feedparser
import requests
from bs4 import BeautifulSoup, SoupStrainer


@dataclass
class Article:
    title: str
    link: str
    published: str | None
    text: str


class NewsScraper:
    def __init__(
        self, feed_url: str = 'https://ici.radio-canada.ca/rss/4159', timeout: int = 15
    ) -> None:
        self.feed_url = feed_url
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                'User-Agent': (
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
            }
        )

    def get_top_articles(self, limit: int = 5) -> list[Article]:
        feed = feedparser.parse(self.feed_url)
        articles: list[Article] = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, 'title', '').strip()
            link = getattr(entry, 'link', '').strip()
            published = getattr(entry, 'published', None)
            full_text = self._fetch_article_text(link)
            articles.append(Article(title=title, link=link, published=published, text=full_text))
        return articles

    def _fetch_article_text(self, url: str) -> str:
        if not url:
            return ''
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return ''

        # Parse selectively to speed up: we care about article/main areas
        only = SoupStrainer(['article', 'main', 'div', 'section'])
        soup = BeautifulSoup(resp.text, 'html.parser', parse_only=only)

        # Try site-specific-ish containers first
        candidates = [
            'article',
            'main',
            '[role="main"]',
            '.story-content',
            '.c-article-content',
            '.article__content',
            'section#content',
            'div#content',
        ]
        node = None
        for sel in candidates:
            node = soup.select_one(sel)
            if node:
                break
        if node is None:
            node = soup  # fallback to whole parsed tree

        # Remove obvious non-content elements
        for tag in node.select(
            'script, style, nav, header, footer, aside, form, noscript, '
            '.share, .social, .related, .newsletter, .advert, .ad, .promo, .breadcrumbs'
        ):
            tag.decompose()

        # Prefer paragraphs
        parts: list[str] = []
        for p in node.find_all(['p', 'h2', 'h3', 'li']):
            text = p.get_text(separator=' ', strip=True)
            if not text:
                continue
            # Skip boilerplate lines
            if len(text) < 3:
                continue
            parts.append(text)

        # Fallbacks if very short
        if not parts or sum(len(x) for x in parts) < 200:
            og_desc = soup.select_one('meta[property="og:description"]')
            if og_desc and og_desc.get('content'):
                parts.append(og_desc['content'].strip())
            meta_desc = soup.select_one('meta[name=description]')
            if meta_desc and meta_desc.get('content'):
                parts.append(meta_desc['content'].strip())

        # Deduplicate adjacent lines and join
        cleaned: list[str] = []
        last = None
        for line in parts:
            if line != last:
                cleaned.append(line)
                last = line
        return '\n\n'.join(cleaned).strip()
