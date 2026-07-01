# Config-Driven TAM Validation Architecture

This document describes the current pipeline blueprint. The system validates a
company/domain list against a target criterion using only local cached evidence,
then extracts matching company names from the retrieved chunks.

## Pipeline Blueprint

```text
CLI arguments
   |
   v
load .env and OPENAI_API_KEY
   |
   v
load YAML criteria config
   |
   v
load companies CSV
   |
   v
load cached pages for company domains
   |
   v
reconstruct URLs from cache filenames
   |
   v
HTML/PDF to text extraction
   |
   v
per-company root-domain filtering
   |
   v
DomainIndexCache.get_or_build()
   |
   v
page chunking + tokenization + BM25 index
   |
   v
criteria retrieval terms + company query templates
   |
   v
top BM25 evidence chunks
   |
   v
LLM criteria-extraction payload
   |
   v
ReferencedCompany matches
   |
   v
ValidationResult
   |
   v
incremental CSV + JSON + JSONL payload log
```

## File Map

```text
input/
  new.py                    CLI entry point and output writer
  criteria_config.py        YAML criteria loader
  validation_engine.py      page loading, retrieval, extraction orchestration
  retrieval.py              HTML text extraction, query generation, BM25 ranking
  domain_index.py           chunking, tokenization, BM25, domain index cache
  domain_grouping.py        domain normalization and root-domain matching
  llm_validator.py          LLM payloads, API calls, response parsing
  models.py                 shared dataclasses
  configs/                  reusable validation blueprints
  companies_scored.csv      default company input rows
  pages/                    local cached evidence corpus
```

Tests live under `tests/`.

## Criteria Configs

Each YAML file under `input/configs/` defines a target blueprint:

```yaml
criteria:
  name: pbt_plastic_manufacturer
  label: PBT plastic manufacturer
  description: Companies that manufacture PBT materials.
  complete_match:
    - The company manufactures PBT resin, pellets, or compounds.
  partial_match:
    - The company distributes PBT but manufacturing is unclear.
  reject:
    - The company only uses PBT in finished products.
  extraction_rules:
    - Only complete matches can receive medium or high confidence.
  retrieval_terms:
    - PBT resin manufacturer
  company_query_templates:
    - "{company} {label}"
    - "{company} PBT resin"
  output_positive_label: manufacturers_found
  output_negative_label: no_manufacturers_found
```

Loaded into:

```python
ValidationCriteria(
    name,
    label,
    description,
    complete_match,
    partial_match,
    reject,
    extraction_rules,
    retrieval_terms,
    company_query_templates,
    output_positive_label,
    output_negative_label,
)
```

Default lookup:

```text
input/configs/<target-category>.yaml
```

Explicit override:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml
```

If no YAML config exists, the pipeline falls back to legacy target-category
query defaults where available.

## CLI

Common commands:

```powershell
python input\new.py --target-category smartphone_financing --limit 3
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --limit 3
python input\new.py --companies-csv input\companies_scored.csv --config-yaml input\configs\pbt_plastic_manufacturer.yaml
```

Key arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | directory containing `new.py` | Base folder for configs, pages, default inputs, and default outputs. |
| `--target-category` | `smartphone_financing` | Target name; also selects `input/configs/<target-category>.yaml` when present. |
| `--config-yaml` | auto-detected | Explicit criteria blueprint path. |
| `--companies-csv` | `input/companies_scored.csv` | Company input CSV. |
| `--limit` | `0` | If non-zero, validates only the first N companies. |
| `--env-file` | workspace `.env`, then `input/.env` | API key file. |
| `--json-output` | `input/validation_results.json` | JSON output. |
| `--csv-output` | `input/validation_results.csv` | CSV output. |
| `--llm-log-output` | `input/llm_payloads.jsonl` | Full LLM payload log. |
| `--top-k` | `8` | Number of retrieved chunks sent to extraction. |
| `--pdf-mode` | `extract` | `extract` uses `pypdf`; `raw` preserves older cached-text behavior. |
| `--extract-referenced-companies` | deprecated | Accepted for compatibility; criteria extraction now always runs. |
| `--verbose` | `False` | Enables debug logging. |

## Inputs

### Company CSV

Default path:

```text
input/companies_scored.csv
```

Used columns:

| Column | Model field |
| --- | --- |
| `company_name` | `CompanyData.company_name` |
| `domain` | `CompanyData.domain` |
| `score` | `CompanyData.company_score` |

The active target category comes from the loaded criteria name, or from
`--target-category` when no criteria config is loaded.

### Cached Pages

Default path:

```text
input/pages/
```

The validator does not crawl the live internet during a run. It reads local
cached `.html` files, reconstructs URLs from filenames, filters them to the
company root domains, and extracts text.

For PDF URLs:

- `--pdf-mode extract` uses `pypdf` and marks metadata `source_type=pdf`.
- `--pdf-mode raw` reads cached text and marks metadata `source_type=pdf_raw`.

HTML text extraction keeps visible text, selected meta fields, and image alt
text while skipping script/style/noscript content.

## Retrieval

For each company:

1. Pages are filtered to the company root domain.
2. `DomainIndexCache` builds or reuses an in-memory index.
3. Page text is capped, whitespace-normalized, and split into overlapping
   chunks.
4. Chunks are tokenized and indexed with `rank-bm25` when installed, otherwise
   `SimpleBM25`.
5. Queries are generated from criteria `retrieval_terms` plus
   `company_query_templates`.
6. BM25 scores are summed per chunk across all queries.
7. The top `--top-k` chunks are sent to the LLM extraction call.

The current retrieval path is BM25-only. Older boosted-retrieval documentation
does not apply to this version.

## LLM Extraction

The current runtime calls:

```python
OpenAIHTTPValidator.extract_referenced_companies(...)
```

It does not use the older first-pass page classification flow for the main CLI
result. The extraction payload asks the model to:

- use only the retrieved chunks
- apply the loaded criteria rules
- extract every matching company name
- include role, responsibility, target relevance, source URL, quote, and notes
- assign an LLM confidence score from `1` to `5`
- explain the confidence score from the retrieved evidence

Transient API failures are retried by `OpenAIHTTPValidator`.

## Results

`ValidationResult` is still the shared result container, but the active CLI
outputs focus on criteria extraction fields:

```python
ValidationResult(
    validated=None,
    confidence=0.0,
    entity_type="unknown",
    evidence_pages=[...top chunk URLs...],
    supporting_snippets=[...top chunk text...],
    reasoning="Extracted matched company names from retrieved BM25 chunks without page/domain classification.",
    referenced_entities=[...ReferencedCompany...],
    referenced_companies=[...ReferencedCompany...],
)
```

No-evidence safeguards return `unknown` results without calling the LLM:

- no local domain pages
- pages present but no indexable text
- no relevant BM25 chunks

LLM failures are caught in the CLI and written as an `unknown` result with
reasoning beginning `LLM validation failed:`.

## CSV Output

Default path:

```text
input/validation_results.csv
```

Columns:

```text
domain
target_category
extraction_status
matched_company_count
matched_company_names
matched_company_llm_confidence_scores
matched_company_llm_confidence_reasoning
matched_company_details
evidence_quotes
source_urls
top_chunk_urls
reasoning
supporting_snippets
```

`extraction_status` is:

- the criteria `output_positive_label` when matches are found
- `no_domain_pages`
- `no_indexable_text`
- `no_relevant_chunks`
- the criteria `output_negative_label` when no matches are found

## JSON And Payload Logs

Default JSON path:

```text
input/validation_results.json
```

The JSON file is rewritten after each completed company using `asdict()` on
each `ValidationResult`.

Default payload log:

```text
input/llm_payloads.jsonl
```

The payload log is cleared at run start and receives one record per LLM
extraction call. No payload is logged for no-evidence safeguard paths.

## Pushing Configs

`input/configs/` is not ignored. To include the blueprints in GitHub:

```powershell
git add input/configs/*.yaml input/README.md input/ARCHITECTURE.md
git commit -m "Document config-driven validation pipeline"
git push
```

Keep `.env`, `__pycache__/`, generated result files, SERP caches, and large page
caches out of the commit unless they are intentionally part of the dataset.

## Test Command

```powershell
python -m unittest discover -s tests -v
```

