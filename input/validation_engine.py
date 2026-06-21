from __future__ import annotations

import csv
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Iterable

from domain_grouping import group_urls_by_domain, root_domain
from domain_index import DomainIndexCache
from llm_validator import LLMValidator, OpenAIHTTPValidator
from models import CompanyData, DiscoveryResult, PageContent, ReferencedCompany, ValidationResult
from retrieval import generate_retrieval_queries, html_to_text, pages_for_domain, retrieve_top_k

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    PdfReader = None

logger = logging.getLogger(__name__)

AGGREGATOR_MANY_ENTITY_THRESHOLD = 5
LOW_EXTRACTION_CONFIDENCE = 0.65


def validate_company_domain(
    company: CompanyData,
    domain_pages: list[PageContent],
    index_cache: DomainIndexCache | None = None,
    validator: LLMValidator | None = None,
    top_k: int = 8,
    use_evidence_boost: bool = True,
    extract_referenced_companies: bool = False,
) -> ValidationResult:
    cache = index_cache or DomainIndexCache()
    evidence_validator = validator or OpenAIHTTPValidator()
    domain = root_domain(company.domain or (domain_pages[0].url if domain_pages else ""))
    logger.info(
        "Validating company=%s domain=%s target=%s company_score=%.4f",
        company.company_name,
        company.domain or domain,
        company.target_category,
        company.company_score,
    )
    if not domain or not domain_pages:
        logger.warning("No domain pages available for %s", company.domain or company.company_name)
        return ValidationResult(
            validated=None,
            confidence=0.0,
            entity_type="unknown",
            evidence_pages=[],
            supporting_snippets=[],
            reasoning="No domain pages were available for validation.",
        )
    index = cache.get_or_build(domain, domain_pages)
    if not index.pages:
        logger.warning("No indexable page content available for %s", company.domain or company.company_name)
        return ValidationResult(
            validated=None,
            confidence=0.0,
            entity_type="unknown",
            evidence_pages=[],
            supporting_snippets=[],
            reasoning="Domain pages were present, but none contained indexable text for validation.",
        )
    queries = generate_retrieval_queries(company)
    logger.info("Generated %d retrieval queries for %s", len(queries), domain)
    logger.debug("Retrieval queries for %s: %s", domain, queries)
    retrieved = retrieve_top_k(index, queries, k=top_k, use_evidence_boost=use_evidence_boost)
    logger.info("Retrieved %d evidence chunks for %s", len(retrieved), domain)
    if not retrieved:
        logger.warning("No retrievable evidence pages found for %s", company.domain or company.company_name)
        return ValidationResult(
            validated=None,
            confidence=0.0,
            entity_type="unknown",
            evidence_pages=[],
            supporting_snippets=[],
            reasoning="No relevant evidence pages were retrieved for validation.",
        )
    for index_number, page in enumerate(retrieved, start=1):
        logger.info(
            "  Evidence chunk %d score=%.4f url=%s chunk_index=%d",
            index_number,
            page.score,
            page.url,
            page.chunk_index,
        )
    logger.info(
        "Extracting provider company names for %s from %d retrieved chunks",
        company.domain or domain,
        len(retrieved),
    )
    referenced_companies = evidence_validator.extract_referenced_companies(
        company,
        retrieved,
        "provider_extraction",
    )
    referenced_companies = filter_extracted_provider_companies(company, referenced_companies, retrieved)
    reasoning = "Extracted provider company names from retrieved BM25 chunks without page/domain classification."
    return ValidationResult(
        validated=None,
        confidence=0.0,
        entity_type="unknown",
        evidence_pages=[page.url for page in retrieved],
        supporting_snippets=[page.content_snippet for page in retrieved],
        reasoning=reasoning,
        page_company=None,
        referenced_entities=referenced_companies,
        extraction_confidence=0.0,
        needs_additional_extraction=False,
        partner_details=[],
        aggregator_company_details=[],
        referenced_companies=referenced_companies,
    )


def filter_extracted_provider_companies(
    company: CompanyData,
    referenced_companies: list[ReferencedCompany],
    retrieved_pages: list | None = None,
) -> list[ReferencedCompany]:
    return [
        referenced_company
        for referenced_company in referenced_companies
        if not _is_editorial_self_extraction(company, referenced_company)
        and not _is_speculative_financing_evidence(referenced_company, retrieved_pages)
        and _has_explicit_target_anchor(referenced_company, retrieved_pages)
    ]


def _is_editorial_self_extraction(company: CompanyData, referenced_company: ReferencedCompany) -> bool:
    company_domain = root_domain(company.domain)
    referenced_domain = root_domain(
        referenced_company.domain
        or referenced_company.url
        or referenced_company.source_url
    )
    if not company_domain or referenced_domain != company_domain:
        return False
    if _normalize_name(referenced_company.name) != _normalize_name(company.company_name):
        return False

    evidence_text = " ".join(
        [
            referenced_company.role,
            referenced_company.financing_responsibility,
            referenced_company.target_relevance,
            referenced_company.quote,
            referenced_company.notes,
        ]
    ).lower()
    editorial_terms = (
        "advis",
        "analys",
        "article",
        "blog",
        "compar",
        "directory",
        "guide",
        "listing",
        "news",
        "rank",
        "recommend",
        "review",
    )
    direct_terms = (
        "underwrite",
        "lender",
        "loan provider",
        "credit provider",
        "lease provider",
        "direct financing",
        "finances purchases",
        "finances smartphones",
        "provides installments",
        "offers installments",
    )
    return any(term in evidence_text for term in editorial_terms) and not any(
        term in evidence_text for term in direct_terms
    )


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _is_speculative_financing_evidence(
    referenced_company: ReferencedCompany,
    retrieved_pages: list | None = None,
) -> bool:
    evidence_text = " ".join(
        [
            referenced_company.quote,
            referenced_company.role,
            referenced_company.financing_responsibility,
            referenced_company.target_relevance,
            referenced_company.notes,
        ]
    ).lower()
    source_url = referenced_company.source_url or referenced_company.url
    if source_url and retrieved_pages:
        evidence_text = " ".join([evidence_text, _source_page_text(source_url, retrieved_pages)]).lower()
    speculative_terms = (
        "potentially",
        "may include",
        "could support",
        "can support",
        "possibly",
        "various payment method",
        "various payment methods",
        "might offer",
    )
    return any(term in evidence_text for term in speculative_terms)


def _has_explicit_target_anchor(
    referenced_company: ReferencedCompany,
    retrieved_pages: list | None = None,
) -> bool:
    if not any(
        [
            referenced_company.quote,
            referenced_company.role,
            referenced_company.financing_responsibility,
            referenced_company.source_url,
            referenced_company.url,
        ]
    ):
        return True
    evidence_text = " ".join(
        [
            referenced_company.quote,
            referenced_company.role,
            referenced_company.financing_responsibility,
        ]
    ).lower()
    source_url = referenced_company.source_url or referenced_company.url
    if source_url and retrieved_pages:
        evidence_text = " ".join([evidence_text, _source_page_text(source_url, retrieved_pages)]).lower()
    target_terms = (
        "cell phone",
        "device",
        "dispositivo",
        "handset",
        "iphone",
        "mobile phone",
        "movil",
        "móvil",
        "phone",
        "smartphone",
        "tablet",
        "wearable",
    )
    return any(term in evidence_text for term in target_terms)


def _source_page_text(source_url: str, retrieved_pages: list) -> str:
    matching_text: list[str] = []
    for page in retrieved_pages:
        page_url = getattr(page, "url", "")
        if page_url != source_url:
            continue
        matching_text.append(getattr(page, "content_snippet", ""))
        matching_text.append(getattr(page, "title", ""))
    return " ".join(matching_text)


def validate_batch(
    companies: Iterable[CompanyData],
    pages: list[PageContent],
    validator: LLMValidator | None = None,
    top_k: int = 8,
    use_evidence_boost: bool = True,
    extract_referenced_companies: bool = False,
) -> dict[str, ValidationResult]:
    cache = DomainIndexCache()
    evidence_validator = validator or OpenAIHTTPValidator()
    results: dict[str, ValidationResult] = {}
    for company in companies:
        domain = root_domain(company.domain)
        domain_pages = pages_for_domain(pages, domain)
        results[company.domain or company.company_name] = validate_company_domain(
            company,
            domain_pages,
            index_cache=cache,
            validator=evidence_validator,
            top_k=top_k,
            use_evidence_boost=use_evidence_boost,
            extract_referenced_companies=extract_referenced_companies,
        )
    logger.info("Validated %d companies", len(results))
    return results


def load_discovery_results(cache_dir: Path) -> list[DiscoveryResult]:
    results: list[DiscoveryResult] = []
    for path in cache_dir.glob("*.json"):
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            data = json.load(handle)
        for row in data:
            url = row.get("href") or row.get("url") or ""
            results.append(
                DiscoveryResult(
                    url=url,
                    title=row.get("title", ""),
                    snippet=row.get("body") or row.get("snippet", ""),
                    domain=root_domain(url),
                )
            )
    return results


def load_page_contents(
    pages_dir: Path,
    domains: Iterable[str] | None = None,
    max_chars_per_page: int = 500_000,
    extract_pdfs: bool = True,
) -> list[PageContent]:
    wanted_domains = {root_domain(domain) for domain in domains or [] if domain}
    pages: list[PageContent] = []
    for path in pages_dir.glob("*.html"):
        url = url_from_cache_filename(path.name)
        if wanted_domains and root_domain(url) not in wanted_domains:
            continue
        if _is_pdf_url(url) and extract_pdfs:
            content = extract_pdf_text(path, max_chars=max_chars_per_page)
            source_type = "pdf"
        else:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    html = handle.read(max_chars_per_page)
            except FileNotFoundError:
                logger.warning("Cached page disappeared before it could be read: %s", path)
                continue
            content = html_to_text(html)
            source_type = "pdf_raw" if _is_pdf_url(url) else "html"
        pages.append(
            PageContent(
                url=url,
                content=content,
                metadata={"source_path": str(path), "source_type": source_type},
            )
        )
    logger.info("Loaded %d cached pages", len(pages))
    return pages


def extract_pdf_text(path: Path, max_chars: int = 500_000) -> str:
    if PdfReader is None:
        logger.warning("pypdf is not installed; unable to extract text from PDF cache file %s", path)
        return ""
    try:
        data = path.read_bytes()
        reader = PdfReader(BytesIO(data))
        parts = []
        extracted_chars = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                clean_text = text.strip()
                remaining_chars = max_chars - extracted_chars
                if remaining_chars <= 0:
                    break
                parts.append(clean_text[:remaining_chars])
                extracted_chars += len(parts[-1])
        return " ".join(" ".join(parts).split())
    except FileNotFoundError:
        logger.warning("Cached PDF disappeared before it could be read: %s", path)
        return ""
    except Exception as exc:  # pragma: no cover - malformed PDFs vary by parser version
        logger.warning("Failed to extract text from PDF cache file %s: %s", path, exc)
        return ""


def _is_pdf_url(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".pdf")


def url_from_cache_filename(filename: str) -> str:
    stem = filename[:-5] if filename.endswith(".html") else filename
    without_hash = stem.rsplit("_", 1)[0]
    if "_" not in without_hash:
        return f"https://{without_hash}/"
    domain, path = without_hash.split("_", 1)
    clean_path = path.replace("_index.html", "").replace("_", "/").strip("/")
    return f"https://{domain}/{clean_path}" if clean_path else f"https://{domain}/"


def load_companies(csv_path: Path, target_category: str) -> list[CompanyData]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CompanyData(
                company_name=row["company_name"],
                company_score=float(row.get("score") or 0.0),
                target_category=target_category,
                domain=row.get("domain", ""),
            )
            for row in reader
        ]


def build_domain_indexes_from_discovery(
    discovery_results: list[DiscoveryResult],
    pages: list[PageContent],
    cache: DomainIndexCache | None = None,
):
    index_cache = cache or DomainIndexCache()
    grouped = group_urls_by_domain(discovery_results)
    return {
        domain: index_cache.get_or_build(domain, pages_for_domain(pages, domain))
        for domain in grouped
        if pages_for_domain(pages, domain)
    }
