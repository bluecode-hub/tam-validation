from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EntityType = Literal["provider", "aggregator", "partner_only", "unknown"]
ValidationDecision = bool | None


@dataclass(frozen=True)
class DiscoveryResult:
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""


@dataclass(frozen=True)
class PageContent:
    url: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompanyData:
    company_name: str
    company_score: float
    target_category: str
    domain: str = ""


@dataclass(frozen=True)
class TopKPage:
    url: str
    score: float
    content_snippet: str
    title: str = ""


@dataclass(frozen=True)
class EntityClassification:
    entity_type: EntityType
    confidence: float
    evidence: list[str]


@dataclass(frozen=True)
class EvidenceQuote:
    url: str
    quote: str
    reason: str


@dataclass(frozen=True)
class LLMValidationJudgment:
    entity_type: EntityType
    validated: ValidationDecision
    confidence: float
    evidence: list[EvidenceQuote]
    reasoning: str


@dataclass(frozen=True)
class ValidationResult:
    validated: ValidationDecision
    confidence: float
    entity_type: EntityType
    evidence_pages: list[str]
    supporting_snippets: list[str]
    reasoning: str
