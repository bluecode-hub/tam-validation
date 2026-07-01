from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EntityType = Literal["provider", "aggregator", "unknown"]
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
    chunk_index: int = 0
    chunk_start: int = 0
    chunk_end: int = 0


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
class ReferencedCompany:
    name: str
    domain: str = ""
    url: str = ""
    role: str = ""
    match_responsibility: str = ""
    match_evidence: str = ""
    llm_confidence_score: str = ""
    llm_confidence_reasoning: str = ""
    target_relevance: str = ""
    quote: str = ""
    source_url: str = ""
    notes: str = ""


@dataclass(frozen=True)
class LLMValidationJudgment:
    entity_type: EntityType
    validated: ValidationDecision
    confidence: float
    evidence: list[EvidenceQuote]
    reasoning: str
    page_company: ReferencedCompany | None = None
    referenced_entities: list[ReferencedCompany] = field(default_factory=list)
    extraction_confidence: float = 0.0
    needs_additional_extraction: bool = False
    partner_details: list[str] = field(default_factory=list)
    aggregator_company_details: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    validated: ValidationDecision
    confidence: float
    entity_type: EntityType
    evidence_pages: list[str]
    supporting_snippets: list[str]
    reasoning: str
    page_company: ReferencedCompany | None = None
    referenced_entities: list[ReferencedCompany] = field(default_factory=list)
    extraction_confidence: float = 0.0
    needs_additional_extraction: bool = False
    partner_details: list[str] = field(default_factory=list)
    aggregator_company_details: list[str] = field(default_factory=list)
    referenced_companies: list[ReferencedCompany] = field(default_factory=list)
