from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Source:
    title: str
    url: str
    snippet: str
    text: str = ""


class WebLearner:
    """Searches and reads public web pages with standard-library networking."""

    USER_AGENT = "AutonomousAIAssistant/1.0 (+local personal learning tool)"

    def search(self, topic: str, limit: int = 6) -> list[Source]:
        query = urllib.parse.urlencode({"q": topic})
        url = f"https://duckduckgo.com/html/?{query}"
        try:
            raw = self._fetch(url, timeout=15)
            return self._parse_duckduckgo(raw, limit)[:limit]
        except Exception:
            return self._wikipedia_fallback(topic, limit=limit)

    def read_sources(self, sources: Iterable[Source], max_chars_per_source: int = 4500) -> list[Source]:
        enriched: list[Source] = []
        for source in sources:
            try:
                raw = self._fetch(source.url, timeout=12)
                source.text = self._html_to_text(raw)[:max_chars_per_source]
            except Exception:
                source.text = source.snippet
            enriched.append(source)
        return enriched

    def summarize(self, topic: str, sources: list[Source]) -> tuple[str, list[str]]:
        text = "\n".join(s.text or s.snippet for s in sources)
        sentences = self._sentences(text)
        topic_words = {w.lower() for w in re.findall(r"[\wа-яА-ЯёЁ]{4,}", topic)}
        scored: list[tuple[int, int, str]] = []
        for i, sent in enumerate(sentences):
            words = {w.lower() for w in re.findall(r"[\wа-яА-ЯёЁ]{4,}", sent)}
            score = len(words & topic_words) * 4 + min(len(words), 35)
            if 55 <= len(sent) <= 360:
                scored.append((score, -i, sent))
        top = [s for _, _, s in sorted(scored, reverse=True)[:8]] or sentences[:8]
        key_points = [self._normalize_sentence(s) for s in top[:6]]
        summary = [f"Резюме по теме «{topic}»:" ]
        for point in key_points:
            summary.append(f"• {point}")
        if sources:
            summary.append("\nИсточники:")
            for src in sources:
                summary.append(f"- {src.title}: {src.url}")
        return "\n".join(summary), key_points

    def _fetch(self, url: str, timeout: int) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            charset = "utf-8"
            match = re.search(r"charset=([^;]+)", content_type)
            if match:
                charset = match.group(1)
            return response.read(1_500_000).decode(charset, errors="replace")

    def _parse_duckduckgo(self, raw: str, limit: int) -> list[Source]:
        results: list[Source] = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)">(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.S,
        )
        for match in pattern.finditer(raw):
            href = html.unescape(match.group("href"))
            if "uddg=" in href:
                parsed = urllib.parse.urlparse(href)
                href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
            title = self._clean_html(match.group("title"))
            snippet = self._clean_html(match.group("snippet"))
            if href.startswith("http") and title:
                results.append(Source(title=title, url=href, snippet=snippet))
            if len(results) >= limit:
                break
        if not results:
            for href, title in re.findall(r'class="result__a" href="([^"]+)">(.*?)</a>', raw, flags=re.S)[:limit]:
                results.append(Source(self._clean_html(title), html.unescape(href), ""))
        return results

    def _wikipedia_fallback(self, topic: str, limit: int) -> list[Source]:
        encoded = urllib.parse.quote(topic)
        return [Source(title=f"Wikipedia: {topic}", url=f"https://ru.wikipedia.org/wiki/{encoded}", snippet=f"Статья энциклопедии по теме {topic}")][:limit]

    def _html_to_text(self, raw: str) -> str:
        raw = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
        raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
        raw = re.sub(r"(?is)</(p|div|li|h[1-6])>", "\n", raw)
        return self._clean_html(raw)

    def _clean_html(self, value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = html.unescape(value)
        return re.sub(r"\s+", " ", value).strip()

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?。])\s+", re.sub(r"\s+", " ", text)) if len(s.strip()) > 40]

    def _normalize_sentence(self, sentence: str) -> str:
        sentence = sentence.strip(" •-\t\n")
        return sentence[:1].upper() + sentence[1:]
