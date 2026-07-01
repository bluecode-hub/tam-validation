from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from criteria_config import ValidationCriteria
from models import CompanyData, EvidenceQuote, LLMValidationJudgment, ReferencedCompany, TopKPage

logger = logging.getLogger(__name__)


class LLMValidationError(RuntimeError):
    pass


class LLMValidator(Protocol):
    def validate(
        self,
        company: CompanyData,
        retrieved_pages: list[TopKPage],
    ) -> LLMValidationJudgment:
        ...

    def extract_referenced_companies(
        self,
        company: CompanyData,
        retrieved_pages: list[TopKPage],
        entity_type: str,
    ) -> list[ReferencedCompany]:
        ...


def build_llm_payload(
    company: CompanyData,
    retrieved_pages: list[TopKPage],
    max_chars_per_page: int = 4_000,
    criteria: ValidationCriteria | None = None,
) -> dict:
    target_label = criteria.label if criteria else company.target_category
    criteria_rules = _classification_rules(company, criteria)
    return {
        "company": asdict(company),
        "criteria": _criteria_payload(criteria),
        "task": {
            "goal": "Map the page company and referenced entities, then determine whether this company/domain directly provides the target category.",
            "target_category": target_label,
            "allowed_entity_types": ["provider", "aggregator", "unknown"],
            "rules": [
                "Use only the provided retrieved chunks.",
                "Do not use outside knowledge.",
                "First identify the page_company: the company/domain represented by the retrieved page itself.",
                "Then extract referenced_entities: every company, partner, provider, merchant, bank, telco, marketplace seller, or website/domain mentioned in the chunks that could relate to the target category.",
                "For each referenced entity, capture name, domain, URL, role, match responsibility, target relevance, source URL, and supporting match evidence where present.",
                "Use the entity map to decide entity_type. The entity extraction and page classification are interlinked and should be reasoned about together.",
                "Return aggregator for comparison sites, directories, review sites, affiliate pages, lead generation pages, or marketplaces listing third-party providers.",
                "For aggregator pages, read and analyze the full content. Identify and extract all company names mentioned in the content, especially for blog, listing, comparison, marketplace, directory, review, affiliate, or lead generation pages.",
                "For aggregator pages, extract relevant company-related information available in the page, including company names, described roles/offers, URLs, domains, website references, and any relationship to the target category.",
                "If an aggregator/blog page discusses multiple companies, include all referenced companies and their associated URLs/domains where possible.",
                "The entity_type, validated value, confidence, evidence, and reasoning must always judge the company/domain represented by the retrieved page itself, not the referenced partner or aggregator-listed companies.",
                "Set extraction_confidence to your confidence that referenced_entities is complete for the provided chunks.",
                "Set needs_additional_extraction to true when the page appears to be an aggregator/blog/listing with many company mentions or when referenced entity extraction is likely incomplete.",
                "Return unknown if evidence is missing, ambiguous, unrelated, or insufficient.",
                "Every evidence item must include a URL and a short quote copied from the provided page text.",
            ]
            + criteria_rules,
        },
        "retrieved_pages": [
            {
                "url": page.url,
                "retrieval_score": page.score,
                "title": page.title,
                "chunk_index": page.chunk_index,
                "chunk_start": page.chunk_start,
                "chunk_end": page.chunk_end,
                "content": page.content_snippet[:max_chars_per_page],
            }
            for page in retrieved_pages
        ],
        "output_schema": {
            "page_company": {
                "name": "string",
                "domain": "string",
                "url": "string",
                "role": "string",
                "match_responsibility": "string",
                "match_evidence": "string",
                "llm_confidence_score": "integer from 1 to 5; assigned only by the LLM based on evidence strength",
                "llm_confidence_reasoning": "string explaining why the LLM assigned that 1-5 score from the retrieved evidence",
                "target_relevance": "string",
                "quote": "string",
                "source_url": "string",
                "notes": "string",
            },
            "referenced_entities": [
                {
                    "name": "string",
                    "domain": "string",
                    "url": "string",
                    "role": "string",
                    "match_responsibility": "string",
                    "match_evidence": "string",
                    "llm_confidence_score": "integer from 1 to 5; assigned only by the LLM based on evidence strength",
                    "llm_confidence_reasoning": "string explaining why the LLM assigned that 1-5 score from the retrieved evidence",
                    "target_relevance": "string",
                    "quote": "string",
                    "source_url": "string",
                    "notes": "string",
                }
            ],
            "entity_type": "provider | aggregator | unknown",
            "validated": "true | false | null",
            "confidence": "float between 0 and 1",
            "extraction_confidence": "float between 0 and 1",
            "needs_additional_extraction": "boolean",
            "evidence": [{"url": "string", "quote": "string", "reason": "string"}],
            "reasoning": "string",
            "partner_details": [
                "deprecated; leave empty. Partner-only pages should be classified as unknown."
            ],
            "aggregator_company_details": [
                "string; for aggregator only, company names, roles/offers, URLs, domains, or website references mentioned in the page"
            ],
        },
    }


def build_llm_prompt(payload: dict) -> str:
    return (
        "You are validating company service evidence.\n"
        "Return JSON only. Do not include markdown.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_referenced_company_payload(
    company: CompanyData,
    retrieved_pages: list[TopKPage],
    entity_type: str = "criteria_extraction",
    max_chars_per_page: int = 4_000,
    criteria: ValidationCriteria | None = None,
) -> dict:
    target_label = criteria.label if criteria else company.target_category
    criteria_rules = _extraction_rules(company, criteria)
    return {
        "company": asdict(company),
        "criteria": _criteria_payload(criteria),
        "task": {
            "goal": "Extract the names of every company in the retrieved chunks that provides the target category.",
            "target_category": target_label,
            "rules": [
                "Use only the provided retrieved chunks.",
                "Do not use outside knowledge.",
                "Do not classify the page, domain, or company being validated.",
                "Do not decide any page/domain classification.",
                "Treat the retrieved chunks as one evidence set. The only aim of this call is to extract company names that provide the target category from these chunks.",
                "If the page company itself directly provides the target service, extract the page company name too.",
                "For each company, return company name, domain or URL if present, role or offer, match responsibility, target-category relevance, and supporting match evidence that proves the company matches the target criteria.",
                "For each returned company, include llm_confidence_score as an integer from 1 to 5, where 1 means very weak evidence and 5 means very strong explicit evidence. This score must be your own LLM judgment from the provided chunks.",
                "For each returned company, include llm_confidence_reasoning explaining why you assigned that 1-5 score, grounded only in the retrieved chunks.",
                "Capture names, domains, URLs, website references, roles/offers, and target-category relevance where available.",
                "If multiple matched companies are discussed, include all of them where possible.",
                "Every returned company must include source_url and a short supporting quote copied from the chunk.",
                "If a domain or URL is not present, leave that field empty.",
            ]
            + criteria_rules,
        },
        "retrieved_chunks": [
            {
                "url": page.url,
                "retrieval_score": page.score,
                "title": page.title,
                "chunk_index": page.chunk_index,
                "chunk_start": page.chunk_start,
                "chunk_end": page.chunk_end,
                "content": page.content_snippet[:max_chars_per_page],
            }
            for page in retrieved_pages
        ],
        "output_schema": {
            "referenced_companies": [
                {
                    "name": "string",
                    "domain": "string",
                    "url": "string",
                    "role": "string",
                    "match_responsibility": "string",
                    "match_evidence": "string",
                    "llm_confidence_score": "integer from 1 to 5; assigned only by the LLM based on evidence strength",
                    "llm_confidence_reasoning": "string explaining why the LLM assigned that 1-5 score from the retrieved evidence",
                    "target_relevance": "string",
                    "quote": "string",
                    "source_url": "string",
                    "notes": "string",
                }
            ]
        },
    }


def build_referenced_company_prompt(payload: dict) -> str:
    return (
        "You are extracting company references from evidence chunks.\n"
        "Return JSON only. Do not include markdown.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _criteria_payload(criteria: ValidationCriteria | None) -> dict:
    if criteria is None:
        return {}
    return {
        "name": criteria.name,
        "label": criteria.label,
        "description": criteria.description,
        "complete_match": criteria.complete_match,
        "partial_match": criteria.partial_match,
        "reject": criteria.reject,
    }


def _classification_rules(company: CompanyData, criteria: ValidationCriteria | None) -> list[str]:
    if criteria is None:
        return [
            "Return provider only when the retrieved page content explicitly says the page company/domain itself directly provides smartphone financing, phone installments, device financing, or an equivalent direct financing/leasing offer.",
            "Provider evidence must be for financing offered by the page company/domain itself, not through a named partner, third-party lender, embedded payment processor, marketplace seller, affiliate, or external financing provider.",
            "Do not infer provider status from generic payment method mentions, customer reviews, marketplace listings, comparison articles, blogs, or content saying financing is handled by another company.",
            "Return unknown when the page only says financing is provided, underwritten, processed, or made available by a separate partner or third party and the page company/domain is not itself the direct financing provider.",
            "The key distinction is: provider requires direct smartphone-financing evidence for the page company/domain itself; aggregator identifies all companies referenced within blog/listing/comparison content; third-party or partner-financing-only evidence is unknown.",
        ]
    rules = [
        f"Target criteria: {criteria.label}.",
        f"Criteria description: {criteria.description or criteria.label}.",
        "Return provider only when the retrieved page content explicitly proves the page company/domain itself is a complete match for the target criteria.",
        "Return aggregator when the page lists, compares, reviews, or references other companies that may match the target criteria.",
        "Return unknown when evidence is missing, ambiguous, speculative, unrelated, partner-only, or only a partial match for the page company/domain itself.",
        "Use complete_match examples as positive match evidence.",
        "Use partial_match examples as referenced-entity or uncertain evidence, not as a complete match classification unless the page company itself is explicitly proven.",
        "Use reject examples as negative evidence.",
    ]
    rules.extend(f"Complete match example: {item}" for item in criteria.complete_match)
    rules.extend(f"Partial match example: {item}" for item in criteria.partial_match)
    rules.extend(f"Reject example: {item}" for item in criteria.reject)
    return rules


def _extraction_rules(company: CompanyData, criteria: ValidationCriteria | None) -> list[str]:
    if criteria is None:
        return [
            "Extract every company name that has definite evidence in the chunks showing it provides smartphone financing, mobile phone financing, phone/device installment plans, pay-monthly phones, leasing/renting of smartphones, underwriting, lending, credit agreements, or installment payment financing specifically for smartphones/devices.",
            "Definite evidence means the chunk explicitly connects that company to providing, offering, financing, underwriting, lending, leasing, enabling installment payments, or handling credit agreements for the target service.",
            "Do not treat simple checkout post-payment, pay-after-delivery, invoice payment, deferred payment, generic BNPL, or ordinary payment methods as smartphone financing unless the chunk explicitly describes an installment plan, loan, lease, credit agreement, device financing, or phone-specific monthly financing.",
            "Do not extract companies from generic consumer finance, personal loans, credit products, financial wellness apps, loan origination, purchasing power, cashback, or generic 'finance when needed' language unless the same evidence explicitly ties that financing to smartphones, mobile phones, handsets, tablets, wearables, or device purchases.",
            "Do not infer smartphone/device financing from broad fintech, lending, BNPL, or credit language. The supporting quote must explicitly contain the target product/service connection.",
            "Include companies even when they are mentioned as a partner, third-party lender, bank, telco, BNPL provider, payment provider, or financing provider.",
            "Do not extract merchants, retailers, shops, marketplaces, coupon sites, comparison sites, or page companies whose role is only selling the device, hosting the checkout, listing the offer, or making third-party payment options available.",
            "Do not extract the page company/domain itself when its own page is editorial, advisory, comparison, directory, affiliate, review, news, or blog content about financing; extract only the actual third-party matching companies named in that content.",
            "Writing about, analyzing, recommending, comparing, reviewing, listing, ranking, or advising on smartphone financing is not evidence that the page company provides smartphone financing.",
            "When a quote says a merchant offers flexible payment through named companies, extract the named financing/payment-plan providers and do not extract the merchant unless the quote explicitly says the merchant itself finances, underwrites, lends, leases, or provides the installment credit.",
            "Do not extract a company when the quote only shows it sells phones, publishes an article, hosts a directory, or mentions financing by another unnamed entity.",
        ]
    rules = [
        f"Target criteria: {criteria.label}.",
        f"Criteria description: {criteria.description or criteria.label}.",
        "Extract only companies with definite evidence for a complete or partial match to the target criteria.",
        "Use complete_match examples as strong positive extraction evidence.",
        "Use partial_match examples as weaker extraction evidence and explain the limitation in target_relevance or notes.",
        "Use reject examples as exclusion rules.",
    ]
    rules.extend(criteria.extraction_rules)
    rules.extend(f"Complete match example: {item}" for item in criteria.complete_match)
    rules.extend(f"Partial match example: {item}" for item in criteria.partial_match)
    rules.extend(f"Reject example: {item}" for item in criteria.reject)
    return rules


def parse_llm_judgment(raw: str) -> LLMValidationJudgment:
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise LLMValidationError(f"LLM returned invalid JSON: {exc}") from exc

    entity_type = data.get("entity_type")
    if entity_type == "partner_only":
        entity_type = "unknown"
    if entity_type not in {"provider", "aggregator", "unknown"}:
        raise LLMValidationError(f"Invalid entity_type: {entity_type!r}")

    validated = _normalize_validated(data.get("validated"))
    if validated not in {True, False, None}:
        raise LLMValidationError(f"Invalid validated value: {validated!r}")

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(confidence, 1.0))
    evidence = [
        EvidenceQuote(
            url=str(item.get("url", "")),
            quote=str(item.get("quote", "")),
            reason=str(item.get("reason", "")),
        )
        for item in data.get("evidence", [])
        if isinstance(item, dict)
    ]
    return LLMValidationJudgment(
        entity_type=entity_type,
        validated=validated,
        confidence=confidence,
        evidence=evidence,
        reasoning=str(data.get("reasoning", "")),
        page_company=_parse_referenced_company_item(data.get("page_company")),
        referenced_entities=_parse_referenced_company_list(data.get("referenced_entities", [])),
        extraction_confidence=_clamp_float(data.get("extraction_confidence", 0.0)),
        needs_additional_extraction=_normalize_bool(data.get("needs_additional_extraction")),
        partner_details=_normalize_partner_details(data.get("partner_details")),
        aggregator_company_details=_normalize_details(data.get("aggregator_company_details")),
    )


def parse_referenced_companies(raw: str) -> list[ReferencedCompany]:
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise LLMValidationError(f"LLM returned invalid referenced-company JSON: {exc}") from exc
    return _parse_referenced_company_list(data.get("referenced_companies", []))


class OpenAIHTTPValidator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        endpoint: str = "https://api.openai.com/v1/responses",
        timeout_seconds: int = 60,
        debug_payload_path: Path | None = None,
        log_payload_preview: bool = True,
        criteria: ValidationCriteria | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.debug_payload_path = debug_payload_path
        self.log_payload_preview = log_payload_preview
        self.criteria = criteria
        self.max_retries = max_retries

    def validate(
        self,
        company: CompanyData,
        retrieved_pages: list[TopKPage],
    ) -> LLMValidationJudgment:
        if not self.api_key:
            raise LLMValidationError("OPENAI_API_KEY is not configured")
        payload = build_llm_payload(company, retrieved_pages, criteria=self.criteria)
        self._log_payload(company, payload)
        prompt = build_llm_prompt(payload)
        logger.info(
            "Sending LLM request for %s (%s) with %d retrieved chunks",
            company.company_name,
            company.domain,
            len(retrieved_pages),
        )
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "model": self.model,
                    "input": prompt,
                    "text": {"format": {"type": "json_object"}},
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = self._post_json(request, "LLM request")
        judgment = parse_llm_judgment(_response_text(data))
        logger.info(
            "LLM judgment for %s: validated=%s entity_type=%s confidence=%.3f",
            company.domain,
            judgment.validated,
            judgment.entity_type,
            judgment.confidence,
        )
        return judgment

    def extract_referenced_companies(
        self,
        company: CompanyData,
        retrieved_pages: list[TopKPage],
        entity_type: str,
    ) -> list[ReferencedCompany]:
        if not self.api_key:
            raise LLMValidationError("OPENAI_API_KEY is not configured")
        payload = build_referenced_company_payload(company, retrieved_pages, entity_type, criteria=self.criteria)
        self._log_extraction_payload(company, payload)
        prompt = build_referenced_company_prompt(payload)
        logger.info(
            "Sending referenced-company extraction request for %s (%s) with %d chunks",
            company.company_name,
            company.domain,
            len(retrieved_pages),
        )
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "model": self.model,
                    "input": prompt,
                    "text": {"format": {"type": "json_object"}},
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = self._post_json(request, "Referenced-company extraction")
        companies = parse_referenced_companies(_response_text(data))
        logger.info("Extracted %d referenced companies for %s", len(companies), company.domain)
        return companies

    def _post_json(self, request: urllib.request.Request, operation: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt == self.max_retries:
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = _retry_delay(attempt, retry_after)
                logger.warning("%s failed with HTTP %s; retrying in %.1fs (%d/%d)", operation, exc.code, delay, attempt, self.max_retries)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = _retry_delay(attempt)
                logger.warning("%s failed: %s; retrying in %.1fs (%d/%d)", operation, exc, delay, attempt, self.max_retries)
                time.sleep(delay)
        raise LLMValidationError(f"{operation} failed: {last_exc}") from last_exc

    def _log_payload(self, company: CompanyData, payload: dict) -> None:
        pages = payload.get("retrieved_pages", [])
        if self.log_payload_preview:
            logger.info(
                "LLM payload for %s includes %d chunks; full payload is written to %s",
                company.domain,
                len(pages),
                self.debug_payload_path or "(disabled)",
            )
            for index, page in enumerate(pages, start=1):
                content = " ".join(str(page.get("content", "")).split())
                logger.info(
                    "  LLM chunk %d score=%.4f url=%s chunk_index=%s preview=%s",
                    index,
                    float(page.get("retrieval_score") or 0.0),
                    page.get("url", ""),
                    page.get("chunk_index", 0),
                    content[:300],
                )
        if not self.debug_payload_path:
            return
        self.debug_payload_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "company_name": company.company_name,
            "domain": company.domain,
            "target_category": company.target_category,
            "payload": payload,
        }
        with self.debug_payload_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_extraction_payload(self, company: CompanyData, payload: dict) -> None:
        if not self.debug_payload_path:
            return
        self.debug_payload_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_type": "referenced_company_extraction",
            "company_name": company.company_name,
            "domain": company.domain,
            "target_category": company.target_category,
            "payload": payload,
        }
        with self.debug_payload_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def _normalize_validated(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized in {"null", "none", "uncertain", "unknown", ""}:
            return None
    return value


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(2 ** (attempt - 1), 10.0)


def _clamp_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 1.0))


def _parse_referenced_company_item(item) -> ReferencedCompany | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    return ReferencedCompany(
        name=name,
        domain=str(item.get("domain", "")).strip(),
        url=str(item.get("url", "")).strip(),
        role=str(item.get("role", "")).strip(),
        match_responsibility=str(
            item.get("match_responsibility")
            or ""
        ).strip(),
        match_evidence=str(item.get("match_evidence") or item.get("quote") or "").strip(),
        llm_confidence_score=str(item.get("llm_confidence_score", "")).strip(),
        llm_confidence_reasoning=str(item.get("llm_confidence_reasoning", "")).strip(),
        target_relevance=str(item.get("target_relevance", "")).strip(),
        quote=str(item.get("quote", "")).strip(),
        source_url=str(item.get("source_url", "")).strip(),
        notes=str(item.get("notes", "")).strip(),
    )


def _parse_referenced_company_list(value) -> list[ReferencedCompany]:
    if not isinstance(value, list):
        return []
    companies: list[ReferencedCompany] = []
    for item in value:
        company = _parse_referenced_company_item(item)
        if company is not None:
            companies.append(company)
    return companies


def _normalize_partner_details(value) -> list[str]:
    return _normalize_details(value)


def _normalize_details(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    details: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = "; ".join(f"{key}: {val}" for key, val in item.items() if val)
        else:
            text = str(item)
        text = text.strip()
        if text:
            details.append(text)
    return details


def _response_text(data: dict) -> str:
    if "output_text" in data:
        return str(data["output_text"])
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    if chunks:
        return "\n".join(chunks)
    raise LLMValidationError("LLM response did not contain output text")
