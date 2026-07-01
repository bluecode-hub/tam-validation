# TAM Validation Pipeline

This pipeline validates a list of company domains against a configurable target
criterion. It uses cached local pages, BM25 chunk retrieval, and an LLM
extraction pass to return matched company names and supporting evidence.

## Current Architecture

The current flow is config-driven:

```text
companies_scored.csv
  -> cached pages in input/pages/
  -> domain filtering
  -> HTML/PDF text extraction
  -> per-domain chunked BM25 index
  -> retrieval queries from input/configs/*.yaml
  -> top evidence chunks
  -> LLM criteria extraction
  -> incremental CSV, JSON, and JSONL payload outputs
```

The YAML files in `input/configs/` define the target criteria, retrieval terms,
query templates, extraction rules, and output status labels. If a config exists
for `--target-category`, it is loaded automatically from:

```text
input/configs/<target-category>.yaml
```

You can also pass a specific config:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml
```

## Important Files

- `new.py` is the CLI entry point and incremental output writer.
- `criteria_config.py` loads YAML criteria into `ValidationCriteria`.
- `validation_engine.py` loads cached pages, builds/reuses indexes, retrieves
  chunks, and runs criteria extraction.
- `retrieval.py` generates criteria-aware queries and ranks BM25 chunks.
- `domain_index.py` chunks page text and builds the per-domain BM25 index.
- `llm_validator.py` builds LLM extraction payloads, calls the OpenAI Responses
  API, retries transient failures, and parses structured company matches.
- `models.py` contains shared dataclasses.
- `configs/` contains the committed reusable target blueprints.

## Run

Put your API key in the workspace `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Run with the default target. This automatically loads
`input/configs/smartphone_financing.yaml` when present:

```powershell
python input\new.py --target-category smartphone_financing --limit 3
```

Run with an explicit blueprint:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --limit 3
```

Override the input company list:

```powershell
python input\new.py --companies-csv input\companies_scored.csv --config-yaml input\configs\pbt_plastic_manufacturer.yaml
```

Override outputs:

```powershell
python input\new.py --limit 3 --json-output output\results.json --csv-output output\results.csv --llm-log-output output\llm_payloads.jsonl
```

## Outputs

The command writes incrementally after each company:

- `input/validation_results.csv`
- `input/validation_results.json`
- `input/llm_payloads.jsonl`

The CSV is optimized for extraction review. Its columns include:

- `target_category`
- `extraction_status`
- `matched_company_count`
- `matched_company_names`
- `matched_company_llm_confidence_scores`
- `matched_company_llm_confidence_reasoning`
- `matched_company_details`
- `evidence_quotes`
- `source_urls`
- `top_chunk_urls`
- `reasoning`
- `supporting_snippets`

## Pushing Configs To GitHub

`input/configs/` is not ignored by the repo. To include the YAML blueprints in
GitHub, stage and commit them:

```powershell
git add input/configs/*.yaml input/README.md input/ARCHITECTURE.md
git commit -m "Document config-driven validation pipeline"
git push
```

Do not add `.env`, `__pycache__/`, cached SERP files, generated outputs, or page
caches unless you intentionally want those large/local artifacts in the repo.

## Test

```powershell
python -m unittest discover -s tests -v
```

