# Domain-Aware Validation Pipeline

This pipeline validates whether a company directly provides a target service by
building one retrieval index per root domain and evaluating evidence at the
company/domain level.

## Modules

- `domain_grouping.py` groups discovery URLs by normalized root domain.
- `domain_index.py` builds and caches one BM25 index per domain. It uses
  `rank-bm25` when installed and falls back to a built-in BM25 implementation.
- `retrieval.py` generates target/category-aware queries and returns top pages
  with snippets.
- `llm_validator.py` builds the LLM evidence-validation prompt, parses the
  structured judgment, and calls an OpenAI-compatible HTTP endpoint.
- `validation_engine.py` orchestrates page loading, indexing, retrieval,
  LLM validation, and batch validation.
- `new.py` is the CLI entry point over the cached `companies_scored.csv` and
  `pages/` corpus.

## Run

Put your API key in the workspace `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Then run a small test batch:

```powershell
python input\new.py --target-category smartphone_financing --limit 3
```

Omit `--limit` to validate all companies. The loader filters cached pages to
the domains being validated and caps per-page reads to keep large cached files
manageable.

The command writes:

- `input/validation_results.json`
- `input/validation_results.csv`
- `input/llm_payloads.jsonl`

The LLM payload log is JSONL: one JSON object per company. It includes company
data, target category, selected evidence page URLs, BM25 scores, and the exact
page text snippets sent to the LLM.

You can override paths:

```powershell
python input\new.py --limit 3 --json-output output\results.json --csv-output output\results.csv
```

You can override the LLM debug log:

```powershell
python input\new.py --limit 3 --llm-log-output output\llm_payloads.jsonl
```

Increase terminal detail:

```powershell
python input\new.py --limit 3 --verbose
```

You can also point at a different env file:

```powershell
python input\new.py --limit 3 --env-file .env.local
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## LLM Validation

BM25 retrieval only selects candidate evidence pages. The validation decision is
made by the LLM validator.

The LLM receives company data, target category, retrieved URLs, retrieval
scores, and page text snippets. It returns `entity_type`, `validated`,
`confidence`, evidence quotes with URLs, and reasoning.

`OPENAI_API_KEY` must be configured for the default CLI/runtime validator.
Tests inject mocked validators so they do not need network access.

## Scores

There is no final scoring formula. BM25 scores are used only to rank retrieved
pages, and `company_score` is sent to the LLM as context. The final validation
decision and confidence come directly from the LLM judgment.
