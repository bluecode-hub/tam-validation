from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from domain_grouping import root_domain
from domain_index import DomainIndexCache
from models import ValidationResult
from llm_validator import OpenAIHTTPValidator
from retrieval import pages_for_domain
from validation_engine import load_companies, load_page_contents, validate_company_domain


CSV_FIELDNAMES = [
    "domain",
    "target_category",
    "extraction_status",
    "provider_company_count",
    "provider_company_names",
    "provider_company_details",
    "evidence_quotes",
    "source_urls",
    "top_chunk_urls",
    "reasoning",
    "supporting_snippets",
]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def default_env_file(input_dir: Path) -> Path:
    workspace_env = input_dir.parent / ".env"
    return workspace_env if workspace_env.exists() else input_dir / ".env"


def write_json_results(path: Path, results: dict[str, ValidationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({key: asdict(value) for key, value in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv_results(path: Path, results: dict[str, ValidationResult], target_category: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for domain, result in results.items():
            writer.writerow(csv_row(domain, result, target_category))


def prepare_csv_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES).writeheader()


def append_csv_result(path: Path, domain: str, result: ValidationResult, target_category: str = "") -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writerow(csv_row(domain, result, target_category))
        handle.flush()


def csv_row(domain: str, result: ValidationResult, target_category: str = "") -> dict[str, object]:
    provider_companies = result.referenced_companies or result.referenced_entities
    provider_names = _unique_nonempty(company.name for company in provider_companies)
    source_urls = _unique_nonempty(
        (company.source_url or company.url or company.domain) for company in provider_companies
    )
    evidence_quotes = _unique_nonempty(company.quote for company in provider_companies)
    return {
        "domain": domain,
        "target_category": target_category,
        "extraction_status": extraction_status(result),
        "provider_company_count": len(provider_names),
        "provider_company_names": " | ".join(provider_names),
        "provider_company_details": json.dumps(
            [asdict(company) for company in provider_companies],
            ensure_ascii=False,
        ),
        "evidence_quotes": " | ".join(evidence_quotes),
        "source_urls": " | ".join(source_urls),
        "top_chunk_urls": " | ".join(_unique_nonempty(result.evidence_pages)),
        "reasoning": result.reasoning,
        "supporting_snippets": " | ".join(result.supporting_snippets),
    }


def extraction_status(result: ValidationResult) -> str:
    if result.referenced_companies or result.referenced_entities:
        return "providers_found"
    lowered_reasoning = result.reasoning.lower()
    if "no domain pages" in lowered_reasoning:
        return "no_domain_pages"
    if "indexable text" in lowered_reasoning:
        return "no_indexable_text"
    if "no relevant evidence" in lowered_reasoning:
        return "no_relevant_chunks"
    return "no_providers_found"


def _unique_nonempty(values) -> list[str]:
    seen = set()
    unique: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def prepare_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run domain-aware company validation.")
    parser.add_argument("--input-dir", default=Path(__file__).parent, type=Path)
    parser.add_argument("--target-category", default="smartphone_financing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--llm-log-output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--retrieval-mode",
        choices=["boosted", "bm25"],
        default="boosted",
        help="Use boosted retrieval scoring or plain BM25-only ranking.",
    )
    parser.add_argument(
        "--pdf-mode",
        choices=["extract", "raw"],
        default="extract",
        help="Extract PDF text with pypdf or use the old raw cached-text behavior.",
    )
    parser.add_argument(
        "--extract-referenced-companies",
        action="store_true",
        help="Deprecated; provider-company extraction now always runs after BM25 retrieval.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    input_dir = args.input_dir.resolve()
    load_env_file(args.env_file or default_env_file(input_dir))

    companies = load_companies(input_dir / "companies_scored.csv", args.target_category)
    if args.limit:
        companies = companies[: args.limit]
    logging.info(
        "Starting validation run: companies=%d target_category=%s",
        len(companies),
        args.target_category,
    )
    logging.info("Retrieval mode: %s", args.retrieval_mode)
    logging.info("PDF mode: %s", args.pdf_mode)
    logging.info("Provider-company extraction: enabled")
    pages = load_page_contents(
        input_dir / "pages",
        domains=[company.domain for company in companies],
        extract_pdfs=args.pdf_mode == "extract",
    )
    json_output = args.json_output or input_dir / "validation_results.json"
    csv_output = args.csv_output or input_dir / "validation_results.csv"
    llm_log_output = args.llm_log_output or input_dir / "llm_payloads.jsonl"
    prepare_log_file(llm_log_output)
    prepare_csv_results(csv_output)
    validator = OpenAIHTTPValidator(debug_payload_path=llm_log_output)
    cache = DomainIndexCache()
    results: dict[str, ValidationResult] = {}
    for index, company in enumerate(companies, start=1):
        domain = company.domain or company.company_name
        logging.info("Processing company %d/%d: %s", index, len(companies), domain)
        result = validate_company_domain(
            company,
            pages_for_domain(pages, root_domain(company.domain)),
            index_cache=cache,
            validator=validator,
            top_k=args.top_k,
            use_evidence_boost=args.retrieval_mode == "boosted",
            extract_referenced_companies=args.extract_referenced_companies,
        )
        results[domain] = result
        append_csv_result(csv_output, domain, result, args.target_category)
        write_json_results(json_output, results)
        logging.info("Wrote incremental result for %s to %s", domain, csv_output)
    logging.info("Wrote JSON results to %s", json_output)
    logging.info("Wrote CSV results to %s", csv_output)
    logging.info("Wrote LLM payload debug log to %s", llm_log_output)
    print(json.dumps({key: asdict(value) for key, value in results.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
