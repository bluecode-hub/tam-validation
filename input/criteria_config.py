from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ValidationCriteria:
    name: str
    label: str
    description: str = ""
    complete_match: list[str] = field(default_factory=list)
    partial_match: list[str] = field(default_factory=list)
    reject: list[str] = field(default_factory=list)
    extraction_rules: list[str] = field(default_factory=list)
    retrieval_terms: list[str] = field(default_factory=list)
    company_query_templates: list[str] = field(default_factory=list)
    output_positive_label: str = "matches_found"
    output_negative_label: str = "no_matches_found"


def load_criteria_config(path: Path) -> ValidationCriteria:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Criteria config must be a mapping: {path}")
    criteria_data = data.get("criteria", data)
    if not isinstance(criteria_data, dict):
        raise ValueError(f"Criteria config 'criteria' must be a mapping: {path}")
    name = _required_text(criteria_data, "name", path)
    label = _text(criteria_data, "label") or name.replace("_", " ")
    return ValidationCriteria(
        name=name,
        label=label,
        description=_text(criteria_data, "description"),
        complete_match=_text_list(criteria_data, "complete_match"),
        partial_match=_text_list(criteria_data, "partial_match"),
        reject=_text_list(criteria_data, "reject"),
        extraction_rules=_text_list(criteria_data, "extraction_rules"),
        retrieval_terms=_text_list(criteria_data, "retrieval_terms"),
        company_query_templates=_text_list(criteria_data, "company_query_templates"),
        output_positive_label=_text(criteria_data, "output_positive_label") or "matches_found",
        output_negative_label=_text(criteria_data, "output_negative_label") or "no_matches_found",
    )


def default_criteria_path(input_dir: Path, target_category: str) -> Path:
    return input_dir / "configs" / f"{target_category}.yaml"


def _required_text(data: dict[str, Any], key: str, path: Path) -> str:
    value = _text(data, key)
    if not value:
        raise ValueError(f"Criteria config missing required field '{key}': {path}")
    return value


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return str(value).strip() if value is not None else ""


def _text_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError(f"Criteria field '{key}' must be a string or list of strings")
    return [str(item).strip() for item in value if str(item).strip()]
