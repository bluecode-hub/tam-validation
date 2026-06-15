from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from models import PageContent

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


class SimpleBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        document_counts: Counter[str] = Counter()
        for doc in corpus:
            document_counts.update(set(doc))
        total = len(corpus)
        self.idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in document_counts.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for index, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[index] or 1
            for token in query_tokens:
                freq = freqs.get(token, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(token, 0.0) * freq * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def _build_bm25(tokenized_pages: list[list[str]]):
    if not tokenized_pages:
        logger.debug("No tokenized pages available; using built-in empty BM25")
        return SimpleBM25(tokenized_pages)
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        logger.debug("Using rank-bm25 BM25Okapi")
        return BM25Okapi(tokenized_pages)
    except ImportError:
        logger.debug("rank-bm25 unavailable; using built-in BM25")
        return SimpleBM25(tokenized_pages)


@dataclass
class DomainIndex:
    domain: str
    pages: list[PageContent]
    bm25_index: object
    tokenized_pages: list[list[str]]
    fingerprint: str

    def query_scores(self, query: str) -> list[float]:
        tokens = tokenize(query)
        if not tokens:
            return [0.0 for _ in self.pages]
        scores = self.bm25_index.get_scores(tokens)
        query_terms = set(tokens)
        adjusted: list[float] = []
        for score, page_tokens in zip(scores, self.tokenized_pages):
            overlap = len(query_terms.intersection(page_tokens))
            adjusted.append(max(float(score), 0.0) + overlap * 0.15)
        return adjusted


@dataclass
class DomainIndexCache:
    _indexes: dict[str, DomainIndex] = field(default_factory=dict)
    max_content_chars: int = 500_000
    chunk_chars: int = 1_250
    chunk_overlap_chars: int = 775
    min_chunk_chars: int = 300

    def get_or_build(self, domain: str, pages: list[PageContent]) -> DomainIndex:
        fingerprint = self._fingerprint(pages)
        cached = self._indexes.get(domain)
        if cached and cached.fingerprint == fingerprint:
            logger.debug("Reusing cached domain index for %s", domain)
            return cached
        capped_pages = []
        for page in pages:
            if not page.content:
                continue
            capped_pages.extend(
                chunk_page_content(
                    PageContent(page.url, page.content[: self.max_content_chars], page.metadata),
                    chunk_chars=self.chunk_chars,
                    overlap_chars=self.chunk_overlap_chars,
                    min_chunk_chars=self.min_chunk_chars,
                )
            )
        tokenized_pairs = [
            (page, tokens)
            for page in capped_pages
            if (tokens := tokenize(page.content))
        ]
        index_pages = [page for page, _tokens in tokenized_pairs]
        tokenized = [tokens for _page, tokens in tokenized_pairs]
        index = DomainIndex(
            domain=domain,
            pages=index_pages,
            bm25_index=_build_bm25(tokenized),
            tokenized_pages=tokenized,
            fingerprint=fingerprint,
        )
        self._indexes[domain] = index
        logger.info("Built domain index for %s with %d chunks", domain, len(index_pages))
        return index

    @staticmethod
    def _fingerprint(pages: list[PageContent]) -> str:
        digest = hashlib.sha1()
        for page in sorted(pages, key=lambda item: item.url):
            digest.update(page.url.encode("utf-8", "ignore"))
            digest.update(str(len(page.content)).encode("ascii"))
        return digest.hexdigest()


def chunk_page_content(
    page: PageContent,
    chunk_chars: int = 2_500,
    overlap_chars: int = 1_000,
    min_chunk_chars: int = 1_000,
) -> list[PageContent]:
    text = " ".join((page.content or "").split())
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [
            PageContent(
                page.url,
                text,
                {**page.metadata, "chunk_index": 0, "chunk_start": 0, "chunk_end": len(text)},
            )
        ]

    chunks: list[PageContent] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if 0 < len(text) - end < min_chunk_chars:
            end = len(text)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                PageContent(
                    page.url,
                    chunk,
                    {
                        **page.metadata,
                        "chunk_index": chunk_index,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks
