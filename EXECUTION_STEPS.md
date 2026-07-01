# Step-by-Step Execution Guide

This guide explains how to run the config-driven TAM validation pipeline from a
fresh terminal in the project root.

## 1. Open The Project

```powershell
cd C:\Users\mahek\mahek\tam_validation_refactor
```

## 2. Check The Inputs

Confirm the company input CSV exists:

```powershell
dir input\companies_scored.csv
```

Confirm the cached page corpus exists:

```powershell
dir input\pages
```

Confirm the criteria configs exist:

```powershell
dir input\configs
```

Each config file is a validation blueprint. Examples:

```text
input\configs\smartphone_financing.yaml
input\configs\plastic_mexico.yaml
input\configs\pbt_resin_manufacturer.yaml
input\configs\pbt_plastic_manufacturer.yaml
```

## 3. Add Your API Key

Create or update the workspace `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

The CLI loads `.env` from the project root first. If that file does not exist,
it checks `input\.env`.

Do not commit `.env` to GitHub.

## 4. Pick A Criteria Blueprint

Use automatic config lookup with `--target-category`:

```powershell
python input\new.py --target-category smartphone_financing --limit 3
```

This loads:

```text
input\configs\smartphone_financing.yaml
```

Or pass a config explicitly:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --limit 3
```

## 5. Run A Small Test Batch

Start with a small limit:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --limit 3
```

For more terminal detail:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --limit 3 --verbose
```

## 6. Run The Full Pipeline

After the small batch looks good, remove `--limit`:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml
```

Or use target-category lookup:

```powershell
python input\new.py --target-category smartphone_financing
```

## 7. Use A Different Company CSV

If your company list is somewhere else:

```powershell
python input\new.py --companies-csv input\companies_scored.csv --config-yaml input\configs\pbt_plastic_manufacturer.yaml
```

The CSV should include:

```text
company_name,domain,score
```

## 8. Write Outputs Somewhere Else

Default outputs are written under `input\`.

To write to `output\`:

```powershell
python input\new.py --config-yaml input\configs\plastic_mexico.yaml --json-output output\results.json --csv-output output\results.csv --llm-log-output output\llm_payloads.jsonl
```

## 9. Review The Outputs

Main CSV:

```text
input\validation_results.csv
```

Main JSON:

```text
input\validation_results.json
```

LLM payload debug log:

```text
input\llm_payloads.jsonl
```

The CSV is the easiest review file. Important columns:

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

## 10. Run Tests

```powershell
python -m unittest discover -s tests -v
```

The tests use mocked LLM paths where needed, so they do not require live API
calls for the covered behavior.

## 11. Push Configs And Docs To GitHub

Check what is currently untracked or modified:

```powershell
git status --short
```

Stage the config-driven pipeline files:

```powershell
git add input\configs\*.yaml input\criteria_config.py tests\test_criteria_config.py tests\test_csv_output.py input\README.md input\ARCHITECTURE.md EXECUTION_STEPS.md
```

Commit:

```powershell
git commit -m "Add config-driven validation pipeline docs"
```

Push:

```powershell
git push
```

Do not stage local/private/generated artifacts unless you explicitly want them
in GitHub:

```text
.env
input\__pycache__\
tests\__pycache__\
input\pages\
input\serp_cache\
input\validation_results.csv
input\validation_results.json
input\llm_payloads.jsonl
```

