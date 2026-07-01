from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

from domain_grouping import root_domain
from domain_index import DomainIndex
from criteria_config import ValidationCriteria
from models import CompanyData, PageContent, TopKPage

logger = logging.getLogger(__name__)

BASE_QUERIES_BY_CATEGORY = {
    "smartphone_financing": [
       "financiacion movil",
        "financiaciÃ³n mÃ³vil",
        "financiacion smartphone",
        "financiaciÃ³n smartphone",
        "movil a plazos",
        "mÃ³vil a plazos",
        "smartphone a plazos",
        "telefono a plazos",
        "telÃ©fono a plazos",
        "comprar movil a plazos",
        "comprar mÃ³vil a plazos",
        "pago a plazos",
        "pago mensual",
        "financiacion sin intereses",
        "financiaciÃ³n sin intereses",
        "0 intereses",
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


def generate_retrieval_queries(
    company: CompanyData,
    criteria: ValidationCriteria | None = None,
) -> list[str]:
    target_label = criteria.label if criteria else target_label_for_category(company.target_category)
    queries = list(criteria.retrieval_terms if criteria else BASE_QUERIES_BY_CATEGORY.get(company.target_category, [target_label]))
    name = company.company_name.strip()
    if name:
        if criteria and criteria.company_query_templates:
            queries.extend(
                template.format(
                    company=name,
                    label=criteria.label,
                    name=criteria.name.replace("_", " "),
                )
                for template in criteria.company_query_templates
            )
        else:
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
    k: int = 8,
    criteria: ValidationCriteria | None = None,
) -> list[TopKPage]:
    totals: dict[str, float] = {}
    chunk_pages: dict[str, PageContent] = {}
    for query in queries:
        for page, score in zip(domain_index.pages, domain_index.query_scores(query)):
            if score <= 0:
                continue
            key = _chunk_key(page)
            totals[key] = totals.get(key, 0.0) + score
            chunk_pages[key] = page
    candidates = []
    for key, score in totals.items():
        page = chunk_pages[key]
        candidates.append(
            TopKPage(
                url=page.url,
                score=score,
                content_snippet=" ".join(page.content.split()),
                title=str(page.metadata.get("title", "")),
                chunk_index=int(page.metadata.get("chunk_index", 0)),
                chunk_start=int(page.metadata.get("chunk_start", 0)),
                chunk_end=int(page.metadata.get("chunk_end", 0)),
            )
        )
    pages = sorted(candidates, key=lambda item: item.score, reverse=True)[:k]
    logger.debug("Retrieved %d top chunks from %s", len(pages), domain_index.domain)
    return pages


def pages_for_domain(pages: list[PageContent], domain: str) -> list[PageContent]:
    wanted = root_domain(domain)
    return [page for page in pages if root_domain(page.url) == wanted]


def target_label_for_category(target_category: str) -> str:
    return TARGET_LABELS_BY_CATEGORY.get(target_category, target_category.replace("_", " ").strip())


def _plain_lower(value: str) -> str:
    replacements = str.maketrans(
        {
            "Ã ": "a",
            "Ã¡": "a",
            "Ã¢": "a",
            "Ã¤": "a",
            "Ã§": "c",
            "Ã¨": "e",
            "Ã©": "e",
            "Ãª": "e",
            "Ã«": "e",
            "Ã®": "i",
            "Ã¯": "i",
            "Ã´": "o",
            "Ã¶": "o",
            "Ã¹": "u",
            "Ã»": "u",
            "Ã¼": "u",
        }
    )
    return value.lower().translate(replacements)


def _chunk_key(page: PageContent) -> str:
    return f"{page.url}#{page.metadata.get('chunk_index', 0)}"
