from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

from domain_grouping import root_domain
from domain_index import DomainIndex
from models import CompanyData, PageContent, TopKPage

logger = logging.getLogger(__name__)

FOCUSED_CONTEXT_CHARS = 2_500


BASE_QUERIES_BY_CATEGORY = {
    "smartphone_financing": [
        "smartphone financing",
        "phone financing",
        "smartphone installment payments",
        "buy phone now pay later",
        "device financing",
        "financing options",
        "consumer financing",
        "telephone paiement echelonne",
        "smartphone a credit",
        "smartphones a credit",
        "pret smartphone",
        "pret smartphone tablette",
        "telephone a credit",
        "achat smartphone credit",
        "acheter smartphone a credit",
        "mensualite smartphone",
        "paiement mensuel smartphone",
    ],
    "device_financing": [
        "device financing",
        "equipment financing",
        "installment payments",
        "financing options",
        "consumer financing",
    ],
    "bnpl": [
        "buy now pay later",
        "bnpl",
        "pay in installments",
        "paiement fractionne",
        "paiement echelonne",
    ],
    "embedded_finance": [
        "embedded finance",
        "integrated financing",
        "merchant financing",
        "consumer financing",
    ],
    "leasing": [
        "leasing",
        "lease financing",
        "location avec option achat",
        "credit bail",
    ],
}

TARGET_LABELS_BY_CATEGORY = {
    "smartphone_financing": "smartphone financing",
    "device_financing": "device financing",
    "bnpl": "buy now pay later",
    "embedded_finance": "embedded finance",
    "leasing": "leasing",
}


EVIDENCE_TERMS_BY_CATEGORY = {
    "smartphone_financing": [
        "smartphone",
        "smartphones",
        "telephone",
        "telephones",
        "tablette",
        "tablettes",
        "credit",
        "pret",
        "financement",
        "mensualite",
        "mensualites",
        "echelonne",
        "installment",
        "installments",
        "monthly",
        "pay later",
        "a credit",
        "à crédit",
    ],
}

PRIORITY_TERMS_BY_CATEGORY = {
    "smartphone_financing": [
        "a credit",
        "à crédit",
        "mensualite",
        "mensualites",
        "installment",
        "installments",
        "financing",
        "financement",
    ],
}

URL_HINTS_BY_CATEGORY = {
    "smartphone_financing": [
        "smartphone",
        "smartphones",
        "telephone",
        "telephones",
        "credit",
        "pret",
        "prt-smartphone",
        "pret-smartphone",
        "smartphones-a-credit",
    ],
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = {str(key).lower(): str(value) for key, value in attrs if value is not None}
        if tag == "meta":
            label = (attr_map.get("name") or attr_map.get("property") or "").lower()
            if label in {
                "description",
                "title",
                "og:title",
                "og:description",
                "twitter:title",
                "twitter:description",
                "twitter:message",
                "email:message",
                "etn:elename",
                "etn:pname",
                "etn:cname",
            }:
                self._append_part(attr_map.get("content", ""))
        if tag == "img":
            self._append_part(attr_map.get("alt", ""))
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._append_part(data)

    def _append_part(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    try:
        parser.feed(html or "")
        return " ".join(parser.text().split())
    except AssertionError:
        without_tags = re.sub(r"<[^>]+>", " ", html or "")
        return " ".join(without_tags.split())


def generate_retrieval_queries(company: CompanyData) -> list[str]:
    target_label = target_label_for_category(company.target_category)
    queries = list(BASE_QUERIES_BY_CATEGORY.get(company.target_category, [target_label]))
    name = company.company_name.strip()
    if name:
        queries.extend(
            [
                f"{name} {target_label}",
                f"{name} financing",
                f"{name} installment plans",
                f"{name} pay later",
                f"{name} {company.target_category.replace('_', ' ')}",
            ]
        )
    return list(dict.fromkeys(queries))


def retrieve_top_k(
    domain_index: DomainIndex,
    queries: list[str],
    k: int = 5,
    use_evidence_boost: bool = True,
) -> list[TopKPage]:
    best: dict[str, TopKPage] = {}
    category = _category_from_queries(queries)
    for query in queries:
        for page, score in zip(domain_index.pages, domain_index.query_scores(query)):
            if score <= 0:
                continue
            adjusted_score = score
            if use_evidence_boost:
                adjusted_score += evidence_score(page, category)
            snippet = snippet_for_query(page.content, query, category=category)
            candidate = TopKPage(
                url=page.url,
                score=adjusted_score,
                content_snippet=snippet,
                title=str(page.metadata.get("title", "")),
            )
            current = best.get(page.url)
            if current is None or candidate.score > current.score:
                best[page.url] = candidate
    pages = sorted(best.values(), key=lambda item: item.score, reverse=True)[:k]
    logger.debug("Retrieved %d top pages from %s", len(pages), domain_index.domain)
    return pages


def evidence_score(page: PageContent, category: str = "") -> float:
    lowered_content = _plain_lower(page.content)
    lowered_url = _plain_lower(page.url.replace("-", " ").replace("/", " "))
    terms = EVIDENCE_TERMS_BY_CATEGORY.get(category, [])
    url_hints = URL_HINTS_BY_CATEGORY.get(category, [])
    score = sum(0.25 for term in terms if _plain_lower(term) in lowered_content)
    score += sum(0.5 for hint in url_hints if _plain_lower(hint) in lowered_url)
    if any(phone in lowered_content for phone in ["smartphone", "telephone", "tablette"]):
        if any(finance in lowered_content for finance in ["credit", "pret", "mensualite", "installment"]):
            score += 2.0
    return score


def snippet_for_query(content: str, query: str, window: int = FOCUSED_CONTEXT_CHARS, category: str = "") -> str:
    lowered = _plain_lower(content)
    terms = [_plain_lower(term) for term in query.split() if len(term) > 2]
    terms.extend(EVIDENCE_TERMS_BY_CATEGORY.get(category, []))
    terms = list(dict.fromkeys(term for term in terms if len(term) > 2))
    priority_terms = [_plain_lower(term) for term in PRIORITY_TERMS_BY_CATEGORY.get(category, [])]
    positions = _term_positions(lowered, priority_terms) or _term_positions(lowered, terms)
    if not positions:
        return " ".join(content[:window].split())

    starts = {max(position - window // 4, 0) for position in positions}
    start = max(starts, key=lambda candidate: _window_score(lowered[candidate : candidate + window], terms))
    return " ".join(content[start : start + window].split())


def pages_for_domain(pages: list[PageContent], domain: str) -> list[PageContent]:
    wanted = root_domain(domain)
    return [page for page in pages if root_domain(page.url) == wanted]


def target_label_for_category(target_category: str) -> str:
    return TARGET_LABELS_BY_CATEGORY.get(target_category, target_category.replace("_", " ").strip())


def _category_from_queries(queries: list[str]) -> str:
    query_text = " ".join(queries).lower()
    for category in BASE_QUERIES_BY_CATEGORY:
        if any(query.lower() in query_text for query in BASE_QUERIES_BY_CATEGORY[category]):
            return category
    return ""


def _plain_lower(value: str) -> str:
    replacements = str.maketrans(
        {
            "à": "a",
            "á": "a",
            "â": "a",
            "ä": "a",
            "ç": "c",
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ö": "o",
            "ù": "u",
            "û": "u",
            "ü": "u",
        }
    )
    return value.lower().translate(replacements)


def _window_score(window_text: str, terms: list[str]) -> float:
    score = sum(1.0 for term in terms if term in window_text)
    if any(phone in window_text for phone in ["smartphone", "telephone", "tablette"]):
        if any(finance in window_text for finance in ["credit", "pret", "mensualite", "installment"]):
            score += 5.0
    return score


def _term_positions(text: str, terms: list[str]) -> list[int]:
    positions: list[int] = []
    for term in terms:
        position = text.find(term)
        if position >= 0:
            positions.append(position)
    return positions
