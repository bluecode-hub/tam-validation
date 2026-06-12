from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from models import CompanyData, EvidenceQuote, LLMValidationJudgment, TopKPage

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


def build_llm_payload(
    company: CompanyData,
    retrieved_pages: list[TopKPage],
    max_chars_per_page: int = 4_000,
) -> dict:
    return {
        "company": asdict(company),
        "task": {
            "goal": "Determine whether this company/domain directly provides the target category.",
            "target_category": company.target_category,
            "allowed_entity_types": ["provider", "aggregator", "partner_only", "unknown"],
            "rules": [
                "Use only the provided retrieved pages.",
                "Do not use outside knowledge.",
                "Return provider only if the company/domain directly offers, finances, underwrites, leases, or sells the target service on installments.",
                "Return aggregator for comparison sites, directories, review sites, affiliate pages, lead generation pages, or marketplaces listing third-party providers.",
                "Return partner_only if the company only says financing is provided by a separate partner and the company is not itself the provider.",
                "Return unknown if evidence is missing, ambiguous, unrelated, or insufficient.",
                "Every evidence item must include a URL and a short quote copied from the provided page text.",
            ],
        },
        "retrieved_pages": [
            {
                "url": page.url,
                "retrieval_score": page.score,
                "title": page.title,
                "content": page.content_snippet[:max_chars_per_page],
            }
            for page in retrieved_pages
        ],
        "output_schema": {
            "entity_type": "provider | aggregator | partner_only | unknown",
            "validated": "true | false | null",
            "confidence": "float between 0 and 1",
            "evidence": [{"url": "string", "quote": "string", "reason": "string"}],
            "reasoning": "string",
        },
    }


def build_llm_prompt(payload: dict) -> str:
    return (
        "You are validating company service evidence.\n"
        "Return JSON only. Do not include markdown.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_llm_judgment(raw: str) -> LLMValidationJudgment:
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise LLMValidationError(f"LLM returned invalid JSON: {exc}") from exc

    entity_type = data.get("entity_type")
    if entity_type not in {"provider", "aggregator", "partner_only", "unknown"}:
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
    )


class OpenAIHTTPValidator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        endpoint: str = "https://api.openai.com/v1/responses",
        timeout_seconds: int = 60,
        debug_payload_path: Path | None = None,
        log_payload_preview: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.debug_payload_path = debug_payload_path
        self.log_payload_preview = log_payload_preview

    def validate(
        self,
        company: CompanyData,
        retrieved_pages: list[TopKPage],
    ) -> LLMValidationJudgment:
        if not self.api_key:
            raise LLMValidationError("OPENAI_API_KEY is not configured")
        payload = build_llm_payload(company, retrieved_pages)
        self._log_payload(company, payload)
        prompt = build_llm_prompt(payload)
        logger.info(
            "Sending LLM request for %s (%s) with %d retrieved pages",
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMValidationError(f"LLM request failed: {exc}") from exc
        judgment = parse_llm_judgment(_response_text(data))
        logger.info(
            "LLM judgment for %s: validated=%s entity_type=%s confidence=%.3f",
            company.domain,
            judgment.validated,
            judgment.entity_type,
            judgment.confidence,
        )
        return judgment

    def _log_payload(self, company: CompanyData, payload: dict) -> None:
        pages = payload.get("retrieved_pages", [])
        if self.log_payload_preview:
            logger.info(
                "LLM payload for %s includes %d pages; full payload is written to %s",
                company.domain,
                len(pages),
                self.debug_payload_path or "(disabled)",
            )
            for index, page in enumerate(pages, start=1):
                content = " ".join(str(page.get("content", "")).split())
                logger.info(
                    "  LLM page %d score=%.4f url=%s preview=%s",
                    index,
                    float(page.get("retrieval_score") or 0.0),
                    page.get("url", ""),
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
