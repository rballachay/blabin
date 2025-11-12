import re
from urllib.parse import urlparse


def _pick_best_result(results: list[dict]) -> dict | None:
    if not results:
        return None
    preferred = ('wikipedia.org', 'britannica.com', 'biography.com')

    def _score(r: dict) -> tuple[int, float]:
        host = urlparse(r.get('url', '')).netloc
        return (1 if any(p in host for p in preferred) else 0, float(r.get('score') or 0.0))

    return sorted(results, key=_score, reverse=True)[0]


def _clean(text: str) -> str:
    t = re.sub(r'\s+', ' ', text or '').strip()
    return t.replace('###', '').replace('#', '')


def synthesize_search_answer(resp: dict) -> tuple[str, list[dict[str, str]]]:
    # Prefer API-provided direct answer
    if resp.get('answer'):
        ans = _clean(resp['answer'])
        cites = [
            {'title': r.get('title') or r.get('url', ''), 'url': r.get('url', '')}
            for r in resp.get('results', [])[:3]
        ]
        return ans, cites

    best = _pick_best_result(resp.get('results', [])) or {}
    snippet = best.get('content') or best.get('raw_content') or ''
    title = best.get('title') or ''
    url = best.get('url') or ''
    text = _clean(snippet) or title or 'No answer found.'
    sentences = re.split(r'(?<=[.!?])\s+', text)
    summary = ' '.join(sentences[:2]).strip()
    return summary, [{'title': title or url, 'url': url}]
