# Domain-Aware LLM Validation Architecture

This document explains the full current workflow for the TAM validation
pipeline: every main input, every transformation, every decision point, and
every output file that is produced.

The system validates whether a company/domain directly provides a target
service, such as `smartphone_financing`. It does not crawl the live internet
during validation. It only uses the company list and the cached page files that
already exist locally.

Current important behavior:

- The final validation decision comes from the LLM.
- BM25 retrieval selects the best evidence chunks to show the LLM. Pages are
  chunked before indexing, and the LLM receives the full retrieved chunks.
- `company_score` is logged and included in the LLM company payload, but it is
  not used in a local scoring formula.
- Retrieval can run in `boosted` mode or `bm25` mode. `boosted` adds local
  evidence-term boosts after BM25 scoring; `bm25` uses only BM25/overlap
  scores.
- PDF cache files can run in `extract` mode, which uses `pypdf`, or `raw`
  mode, which preserves the older raw cached-text behavior.
- Optional referenced-company extraction can run as a second LLM pass for
  `partner_only` and `aggregator` judgments when
  `--extract-referenced-companies` is enabled.
- The system writes results incrementally after every company.
- If no usable evidence exists, the system returns an `unknown` result without
  calling the LLM.

## Current File Map

```text
input/
  new.py                    CLI entry point and output writer
  validation_engine.py      one-company and batch validation orchestration
  retrieval.py              HTML text extraction, query generation, chunk
                            retrieval, optional evidence boosting
  domain_index.py           tokenization, page chunking, BM25 index,
                            in-memory index cache
  domain_grouping.py        domain normalization and root-domain matching
  llm_validator.py          LLM payload, prompt, API call, response parsing
  models.py                 shared dataclasses
  companies_scored.csv      company input rows
  pages/                    cached HTML pages used as the evidence corpus
  validation_results.csv    default CSV output
  validation_results.json   default JSON output
  llm_payloads.jsonl        default LLM payload debug log
```

Test files live under:

```text
tests/
```

They cover domain grouping, retrieval, LLM payload/parsing, no-evidence
safeguards, and validation orchestration with mocked validators.

## High-Level Flow

```text
CLI arguments
   |
   v
environment loading
   |
   v
companies_scored.csv
   |
   v
CompanyData objects
   |
   v
cached HTML files in input/pages/
   |
   v
URL reconstruction from filenames
   |
   v
domain filtering
   |
   v
HTML/PDF-to-text extraction
   |
   v
PageContent objects
   |
   v
per-company domain page selection
   |
   v
DomainIndexCache.get_or_build()
   |
   v
page chunks + tokenized chunks + BM25 index
   |
   v
target/category retrieval queries
   |
   v
summed BM25 scores across queries + optional local evidence boosts
   |
   v
TopKPage evidence chunks
   |
   v
LLM payload + prompt
   |
   v
OpenAI Responses API
   |
   v
LLMValidationJudgment
   |
   v
ValidationResult
   |
   v
incremental CSV row + rewritten JSON + optional payload log
```

## Runtime Entry Point

The pipeline starts in:

```text
input/new.py
```

Basic command:

```powershell
python input\new.py --target-category smartphone_financing
```

Small run:

```powershell
python input\new.py --target-category smartphone_financing --limit 3
```

Custom output paths:

```powershell
python input\new.py --limit 3 --json-output output\results.json --csv-output output\results.csv --llm-log-output output\llm_payloads.jsonl
```

Verbose logging:

```powershell
python input\new.py --limit 3 --verbose
```

## CLI Inputs

`new.py` defines these arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | directory containing `new.py` | Where input CSV, pages, and default outputs are located. |
| `--target-category` | `smartphone_financing` | Target service the system validates. |
| `--limit` | `0` | If non-zero, only validate the first N companies. |
| `--env-file` | auto-detected | Optional `.env` path. |
| `--json-output` | `input/validation_results.json` | JSON result path. |
| `--csv-output` | `input/validation_results.csv` | CSV result path. |
| `--llm-log-output` | `input/llm_payloads.jsonl` | JSONL debug log containing full LLM payloads. |
| `--top-k` | `8` | Maximum retrieved evidence chunks sent to the LLM per company. |
| `--retrieval-mode` | `boosted` | `boosted` adds evidence boosts after BM25; `bm25` uses BM25/overlap scoring only. |
| `--pdf-mode` | `extract` | `extract` uses `pypdf` for PDF URLs; `raw` uses the older raw cached-text behavior. |
| `--extract-referenced-companies` | `False` | If enabled, runs a second LLM pass for `partner_only` and `aggregator` results to extract referenced companies into `referenced_companies`. |
| `--verbose` | `False` | Enables debug-level logging. |

## Environment Loading

The LLM validator requires:

```text
OPENAI_API_KEY
```

Environment loading happens before validation:

1. If `--env-file` is provided, that file is loaded.
2. Otherwise `default_env_file(input_dir)` is used.
3. The default lookup checks the workspace-level `.env` first:

   ```text
   .env
   ```

4. If the workspace-level `.env` does not exist, it checks:

   ```text
   input/.env
   ```

`load_env_file()` reads the file line by line:

- Blank lines are skipped.
- Lines starting with `#` are skipped.
- Lines without `=` are skipped.
- Each valid line is split once at the first `=`.
- Surrounding spaces are removed from the key and value.
- Surrounding single or double quotes are removed from the value.
- Existing environment variables are not overwritten.

Input example:

```text
OPENAI_API_KEY="sk-..."
```

Output effect:

```python
os.environ["OPENAI_API_KEY"] = "sk-..."
```

## Logging Setup

`configure_logging(verbose)` sets:

- `DEBUG` level when `--verbose` is used.
- `INFO` level otherwise.

Log format:

```text
timestamp level logger_name - message
```

The logs report:

- run start
- number of companies
- target category
- retrieval mode
- PDF mode
- cached pages loaded
- each company being processed
- index creation/reuse
- retrieval query count
- evidence chunk URLs, chunk indexes, and scores
- LLM payload chunk preview
- LLM judgment
- incremental output writes

## Input 1: Company CSV

Loaded from:

```text
input/companies_scored.csv
```

by:

```python
load_companies(input_dir / "companies_scored.csv", args.target_category)
```

Required/used columns:

| CSV column | Used as | Type after loading |
| --- | --- | --- |
| `company_name` | `CompanyData.company_name` | `str` |
| `domain` | `CompanyData.domain` | `str` |
| `score` | `CompanyData.company_score` | `float` |

The target category does not come from the CSV. It comes from the CLI argument
and is copied onto every company.

For each row, `load_companies()` creates:

```python
CompanyData(
    company_name=row["company_name"],
    company_score=float(row.get("score") or 0.0),
    target_category=target_category,
    domain=row.get("domain", ""),
)
```

If `score` is blank or missing, it becomes:

```text
0.0
```

If `--limit N` is passed, the company list is sliced after loading:

```python
companies = companies[:N]
```

So `--limit 3` means only the first three CSV rows are processed.

## Input 2: Cached Pages

Loaded from:

```text
input/pages/
```

by:

```python
load_page_contents(input_dir / "pages", domains=[company.domain for company in companies])
```

Important: these are local cached files under `input/pages/`. They are stored
with `.html` filenames even when the reconstructed URL points to a PDF. The
validator does not fetch new pages from the internet.

### Page Filename to URL

Each cached file name is converted back into a URL by:

```python
url_from_cache_filename(filename)
```

The function does this:

1. Remove `.html` from the filename.
2. Remove the final underscore suffix, which is treated as the cache hash.
3. If the remaining value has no underscore, treat it as a domain homepage.
4. If it has an underscore, split once:

   ```text
   domain_path
   ```

5. Convert underscores in the path into `/`.
6. Remove `_index.html` fragments.
7. Return an HTTPS URL.

Example:

```text
www.orangemali.com_fr_catalogs_smartphones-a-credit.html_d777cfff397e.html
```

becomes approximately:

```text
https://www.orangemali.com/fr/catalogs/smartphones-a-credit.html
```

### Page Domain Filtering

Before reading page content, `load_page_contents()` builds a set of wanted root
domains from the companies:

```python
wanted_domains = {root_domain(domain) for domain in domains or [] if domain}
```

Then each cached page is kept only if:

```python
root_domain(url) in wanted_domains
```

This prevents unrelated cached pages from being loaded when they are not needed
for the current company list.

### Page Read Limit

Each cached file is capped with:

```python
max_chars_per_page = 500_000
```

That means only the first 500,000 characters of HTML text or first 500,000 PDF
bytes are read.

### PDF Mode

The CLI option `--pdf-mode` controls how reconstructed `.pdf` URLs are loaded.

Default:

```powershell
python input\new.py --pdf-mode extract
```

In `extract` mode, `load_page_contents()` detects URLs ending in `.pdf` and
uses `pypdf.PdfReader` to extract text. The resulting `PageContent.metadata`
gets:

```python
{"source_type": "pdf"}
```

If `pypdf` is unavailable or the PDF is malformed, the extracted content is an
empty string. This prevents binary/PDF bytes from leaking into evidence.

Compatibility mode:

```powershell
python input\new.py --pdf-mode raw
```

In `raw` mode, cached PDF URLs are read as text and passed through the HTML
text extractor, matching the older behavior. The metadata gets:

```python
{"source_type": "pdf_raw"}
```

This can produce more non-empty retrieval results, but it may also surface
garbled binary-looking evidence for malformed PDF caches.

### HTML-to-Text Extraction

Raw HTML is converted into plain text by:

```python
html_to_text(html)
```

`html_to_text()` uses `TextExtractor`, a subclass of Python's `HTMLParser`.

It keeps:

- visible page text
- selected meta tag content
- image `alt` text

It skips text inside:

- `<script>`
- `<style>`
- `<noscript>`

The extractor includes meta tag content when the meta name/property is one of:

```text
description
title
og:title
og:description
twitter:title
twitter:description
twitter:message
email:message
etn:elename
etn:pname
etn:cname
```

Image alt text is added because product and category pages often put useful
evidence in image descriptions.

After parsing, whitespace is normalized:

```python
" ".join(parser.text().split())
```

If the HTML parser raises an `AssertionError`, the fallback removes tags with a
regex and normalizes whitespace.

### PageContent Output

For every loaded page, the system creates:

```python
PageContent(
    url=url,
    content=html_to_text(html),
    metadata={"source_path": str(path)},
)
```

Fields:

| Field | Meaning |
| --- | --- |
| `url` | Reconstructed HTTPS URL. |
| `content` | Extracted and normalized text. |
| `metadata["source_path"]` | Local HTML cache path. |

Note: `retrieve_top_k()` reads `page.metadata.get("title", "")`, but the current
loader does not populate a `title` metadata field. So titles are currently
blank unless another caller supplies them.

## Domain Normalization

Domain handling lives in:

```text
input/domain_grouping.py
```

### normalize_domain()

`normalize_domain(value)`:

1. Strips whitespace.
2. Lowercases the value.
3. Adds `https://` before parsing if no scheme exists.
4. Extracts the host.
5. Removes userinfo and port.
6. Removes surrounding dots.
7. Removes leading `www.`.

Examples:

| Input | Output |
| --- | --- |
| `https://www.orangemali.com/fr/page` | `orangemali.com` |
| `www.example.com:443/path` | `example.com` |
| `EXAMPLE.COM` | `example.com` |

### root_domain()

`root_domain(value)` returns the registrable-looking root domain used for
matching company domains to page URLs.

For simple domains:

```text
shop.example.com -> example.com
www.orangemali.com -> orangemali.com
```

For configured second-level TLDs:

```text
service.example.co.uk -> example.co.uk
```

Configured second-level suffixes include:

```text
co.uk, com.au, com.br, com.cn, com.tr, com.pk, com.ng, co.ke,
co.za, co.in, com.sg, com.mx, com.eg, net.au, org.uk
```

## Per-Company Processing Loop

After company and page loading, `new.py` prepares outputs and starts this loop:

```python
for index, company in enumerate(companies, start=1):
    domain = company.domain or company.company_name
    result = validate_company_domain(...)
    results[domain] = result
    append_csv_result(csv_output, domain, result)
    write_json_results(json_output, results)
```

For every company:

1. Choose an output key:

   ```python
   company.domain or company.company_name
   ```

2. Filter loaded pages to the company's root domain.
3. Validate that company/domain.
4. Store the returned `ValidationResult`.
5. Append one CSV row immediately.
6. Rewrite the JSON file with all results completed so far.
7. Continue to the next company.

This is why partial runs still leave useful CSV and JSON output.

## Per-Company Page Selection

Before calling the validation engine, `new.py` filters pages with:

```python
pages_for_domain(pages, root_domain(company.domain))
```

`pages_for_domain()` returns:

```python
[page for page in pages if root_domain(page.url) == wanted]
```

Input:

```python
pages: list[PageContent]
domain: str
```

Output:

```python
list[PageContent]
```

Only pages whose root domain exactly matches the company root domain are passed
into `validate_company_domain()`.

## Validation Engine

The main orchestration function is:

```python
validate_company_domain(
    company: CompanyData,
    domain_pages: list[PageContent],
    index_cache: DomainIndexCache | None = None,
    validator: LLMValidator | None = None,
    top_k: int = 5,
) -> ValidationResult
```

It performs these steps:

1. Create or reuse a `DomainIndexCache`.
2. Create or reuse an LLM validator.
3. Compute the root domain.
4. Handle no-page cases.
5. Build or reuse the domain index.
6. Handle no-indexable-content cases.
7. Generate retrieval queries.
8. Retrieve top evidence chunks.
9. Handle no-retrieved-evidence cases.
10. Call the LLM validator.
11. Map the LLM judgment into a `ValidationResult`.

## Validation Step 1: Domain Choice

Inside `validate_company_domain()`, the domain is computed as:

```python
domain = root_domain(company.domain or (domain_pages[0].url if domain_pages else ""))
```

So:

- If `company.domain` exists, it is used.
- Otherwise, the first page URL is used.
- If neither exists, the domain is blank.

The company score is logged:

```python
company.company_score
```

but no local decision formula uses it.

## Validation Step 2: No Domain Pages Safeguard

If the domain is blank or `domain_pages` is empty, no index is built and no LLM
call is made.

Returned result:

```python
ValidationResult(
    validated=None,
    confidence=0.0,
    entity_type="unknown",
    evidence_pages=[],
    supporting_snippets=[],
    reasoning="No domain pages were available for validation.",
)
```

Meaning:

- The system could not validate the company because it had no local pages.
- This does not prove the company does not offer the service.
- It only means the local evidence corpus had no domain pages.

## Validation Step 3: Domain Index Build or Reuse

The engine calls:

```python
index = cache.get_or_build(domain, domain_pages)
```

This lives in:

```text
input/domain_index.py
```

The cache is in memory for the current run only. It is not written to disk.

### Fingerprint

Before building an index, `DomainIndexCache` computes a fingerprint:

```python
fingerprint = self._fingerprint(pages)
```

The fingerprint is a SHA-1 hash based on:

- each page URL
- each page content length

Pages are sorted by URL before hashing so order changes do not change the
fingerprint.

If a cached index already exists for the domain and the fingerprint matches,
the old index is reused.

### Content Cap And Chunking

Before chunking and tokenization, page text is capped:

```python
page.content[: self.max_content_chars]
```

Current default:

```text
500,000 characters per page
```

Pages with empty content are removed.

After the cap, each page is split into overlapping chunks before BM25 indexing.
At runtime, `DomainIndexCache` currently uses:

```python
chunk_chars = 1_250
chunk_overlap_chars = 775
min_chunk_chars = 300
```

The chunker whitespace-normalizes page text, then creates chunk `PageContent`
objects with the same source URL and these metadata fields:

```python
chunk_index
chunk_start
chunk_end
```

If the remaining tail would be shorter than `min_chunk_chars`, it is folded
into the current chunk instead of creating a tiny final chunk.

### Tokenization

Tokenization uses:

```python
TOKEN_RE = re.compile(r"[\w...]+", re.UNICODE)
```

`tokenize(text)`:

1. Finds word-like Unicode tokens.
2. Lowercases every token.
3. Drops one-character tokens.

Input:

```text
Achetez smartphone a credit en mensualite
```

Output:

```python
["achetez", "smartphone", "credit", "en", "mensualite"]
```

Chunks that produce no tokens are removed from the index.

### BM25 Implementation Choice

The code tries to use:

```python
rank_bm25.BM25Okapi
```

If `rank-bm25` is not installed, it falls back to the built-in `SimpleBM25`.

The fallback computes:

- document length
- average document length
- term frequencies per document
- document frequency per token
- IDF
- BM25 score using `k1=1.5` and `b=0.75`

### DomainIndex Output

The built index is:

```python
DomainIndex(
    domain=domain,
    pages=index_pages,
    bm25_index=_build_bm25(tokenized),
    tokenized_pages=tokenized,
    fingerprint=fingerprint,
)
```

Fields:

| Field | Meaning |
| --- | --- |
| `domain` | Root domain being indexed. |
| `pages` | Indexed chunks with non-empty tokenized content. The field name is historical. |
| `bm25_index` | `BM25Okapi` or `SimpleBM25`. |
| `tokenized_pages` | Token list for each indexed chunk. The field name is historical. |
| `fingerprint` | SHA-1 fingerprint for cache reuse. |

## Validation Step 4: No Indexable Text Safeguard

If the index contains no pages:

```python
if not index.pages:
```

the system returns:

```python
ValidationResult(
    validated=None,
    confidence=0.0,
    entity_type="unknown",
    evidence_pages=[],
    supporting_snippets=[],
    reasoning="Domain pages were present, but none contained indexable text for validation.",
)
```

No LLM call is made.

## Validation Step 5: Query Generation

Queries are generated by:

```python
generate_retrieval_queries(company)
```

Inputs:

```python
CompanyData.company_name
CompanyData.target_category
```

Output:

```python
list[str]
```

### Target Label

The first step is:

```python
target_label = target_label_for_category(company.target_category)
```

Current target labels:

| Category | Label |
| --- | --- |
| `smartphone_financing` | `smartphone financing` |
| `device_financing` | `device financing` |
| `bnpl` | `buy now pay later` |
| `embedded_finance` | `embedded finance` |
| `leasing` | `leasing` |

If a category is not configured, underscores are replaced with spaces:

```text
solar_paygo -> solar paygo
```

### Base Queries

The code starts with category-specific base queries from:

```python
BASE_QUERIES_BY_CATEGORY
```

Current `smartphone_financing` queries:

```text
smartphone financing
phone financing
smartphone installment payments
buy phone now pay later
device financing
financing options
consumer financing
telephone paiement echelonne
smartphone a credit
smartphones a credit
pret smartphone
pret smartphone tablette
telephone a credit
achat smartphone credit
acheter smartphone a credit
mensualite smartphone
paiement mensuel smartphone
```

Current `device_financing` queries:

```text
device financing
equipment financing
installment payments
financing options
consumer financing
```

Current `bnpl` queries:

```text
buy now pay later
bnpl
pay in installments
paiement fractionne
paiement echelonne
```

Current `embedded_finance` queries:

```text
embedded finance
integrated financing
merchant financing
consumer financing
```

Current `leasing` queries:

```text
leasing
lease financing
location avec option achat
credit bail
```

If the target category has no configured base query list, the fallback is:

```python
[target_label]
```

### Company-Specific Queries

If the company name is non-empty, these are added:

```python
f"{name} {target_label}"
f"{name} financing"
f"{name} installment plans"
f"{name} pay later"
f"{name} {company.target_category.replace('_', ' ')}"
```

Example for:

```python
CompanyData(
    company_name="Orange Mali",
    target_category="smartphone_financing",
    ...
)
```

additional queries include:

```text
Orange Mali smartphone financing
Orange Mali financing
Orange Mali installment plans
Orange Mali pay later
Orange Mali smartphone financing
```

Duplicates are removed while preserving order:

```python
list(dict.fromkeys(queries))
```

## Validation Step 6: Chunk Retrieval

Retrieval happens in:

```python
retrieve_top_k(index, queries, k=top_k)
```

Inputs:

| Input | Meaning |
| --- | --- |
| `domain_index` | BM25 index over chunks for the company's root domain. |
| `queries` | Target and company-specific query strings. |
| `k` | Maximum number of evidence chunks to return. |

Output:

```python
list[TopKPage]
```

### Category Inference During Retrieval

Retrieval infers a category from query text:

```python
category = _category_from_queries(queries)
```

It joins all queries into one lowercased string and checks whether any known
base query appears inside it. If matched, it returns that category. Otherwise,
it returns an empty string.

This inferred category controls:

- evidence boost terms
- URL boost hints

### BM25 Query Scoring

For each query, retrieval calls:

```python
domain_index.query_scores(query)
```

`query_scores()`:

1. Tokenizes the query.
2. If there are no query tokens, returns all zero scores.
3. Gets BM25 scores from the index.
4. Computes query-token overlap with each chunk.
5. Adds a small overlap bonus:

   ```python
   overlap * 0.15
   ```

6. Clamps BM25 score to be non-negative:

   ```python
   max(float(score), 0.0)
   ```

The output is one score per indexed chunk.

### Ignoring Zero-Score Chunks

Inside `retrieve_top_k()`, chunks with score `<= 0` for a query are skipped:

```python
if score <= 0:
    continue
```

So a chunk must have at least some BM25/overlap relevance before evidence boosts
matter.

### Score Aggregation Across Queries

Each query scores every indexed chunk. For every chunk, the positive scores
from all queries are summed:

```python
totals[chunk_key] = totals.get(chunk_key, 0.0) + score
```

So the current final BM25 score is:

```text
chunk_final_bm25 = sum(BM25/overlap score from every matching query)
```

This favors chunks that match multiple target/company queries over chunks that
only match one query strongly.

### Retrieval Modes

The CLI controls whether local evidence boosts are used:

```powershell
python input\new.py --retrieval-mode boosted
python input\new.py --retrieval-mode bm25
```

In `bm25` mode:

```python
adjusted_score = summed_bm25_score
```

In `boosted` mode:

```python
adjusted_score = summed_bm25_score + evidence_score(chunk, category)
```

The evidence boost is added once per chunk after summing BM25 scores. It is not
multiplied by the number of matching queries.

`evidence_score()` looks at:

- normalized chunk content
- normalized URL text
- category-specific evidence terms
- category-specific URL hints
- product + finance co-occurrence

For `smartphone_financing`, evidence terms are:

```text
smartphone
smartphones
telephone
telephones
tablette
tablettes
credit
pret
financement
mensualite
mensualites
echelonne
installment
installments
monthly
pay later
a credit
... credit
```

For each evidence term found in content:

```text
+0.25
```

For each URL hint found in the URL:

```text
+0.5
```

Current `smartphone_financing` URL hints:

```text
smartphone
smartphones
telephone
telephones
credit
pret
prt-smartphone
pret-smartphone
smartphones-a-credit
```

If the page content contains at least one product word:

```text
smartphone, telephone, tablette
```

and at least one finance word:

```text
credit, pret, mensualite, installment
```

then the chunk gets:

```text
+2.0
```

This helps chunks about financed phones outrank chunks that only mention phones
or only mention generic finance.

### Accent/Mojibake Normalization

Before matching evidence terms, `_plain_lower()` lowercases text and translates
some encoded French characters into ASCII-like forms.

Examples:

```text
... credit forms are normalized toward "a credit"
```

This is useful because some cached pages contain encoding artifacts.

### Top K Sorting

After all queries are processed:

```python
chunks = sorted(candidates, key=lambda item: item.score, reverse=True)[:k]
```

Default:

```text
top_k = 8
```

The result is a list of `TopKPage` objects.

## Validation Step 7: Retrieved Chunk Objects

The old focused context-window step has been removed. The LLM is sent the full
retrieved chunk text, capped later by the LLM payload builder.

### TopKPage Output

Each candidate chunk is stored as:

```python
TopKPage(
    url=chunk.url,
    score=adjusted_score,
    content_snippet=" ".join(chunk.content.split()),
    title=str(chunk.metadata.get("title", "")),
    chunk_index=int(chunk.metadata.get("chunk_index", 0)),
    chunk_start=int(chunk.metadata.get("chunk_start", 0)),
    chunk_end=int(chunk.metadata.get("chunk_end", 0)),
)
```

Fields:

| Field | Meaning |
| --- | --- |
| `url` | Source page URL for the retrieved chunk. Multiple chunks can share one URL. |
| `score` | Summed BM25/overlap score, optionally plus one local evidence boost. |
| `content_snippet` | Full retrieved chunk text. The field name is historical. |
| `title` | Currently usually blank because loader does not set title metadata. |
| `chunk_index` | Zero-based chunk number within the source page. |
| `chunk_start` | Character offset where this chunk starts in normalized page text. |
| `chunk_end` | Character offset where this chunk ends in normalized page text. |

## Validation Step 8: No Retrieved Evidence Safeguard

If retrieval returns no pages:

```python
if not retrieved:
```

the system returns:

```python
ValidationResult(
    validated=None,
    confidence=0.0,
    entity_type="unknown",
    evidence_pages=[],
    supporting_snippets=[],
    reasoning="No relevant evidence pages were retrieved for validation.",
)
```

No LLM call is made.

This means:

- domain pages existed
- the index had text
- but target-specific retrieval found no relevant evidence

It still does not prove the company definitely does not provide the target
service. It only means the cached corpus did not produce retrievable evidence.

## Validation Step 9: LLM Payload Construction

If retrieved chunks exist, the engine calls:

```python
judgment = evidence_validator.validate(company, retrieved)
```

For the default validator, this is:

```python
OpenAIHTTPValidator.validate()
```

The payload is created by:

```python
build_llm_payload(company, retrieved_pages)
```

### Payload Company Object

The company dataclass is converted with:

```python
asdict(company)
```

Example:

```json
{
  "company_name": "Orange Mali",
  "company_score": 0.762,
  "target_category": "smartphone_financing",
  "domain": "orangemali.com"
}
```

### Unified Entity-Aware Payload Task

The first LLM call is entity-aware. It does not only classify the page; it
first maps the page company and referenced entities, then uses that map to
decide the page-level classification.

Conceptually:

```text
BM25 chunks
  -> unified entity-aware validation prompt
      -> page_company
      -> referenced_entities
      -> provider / aggregator / partner_only / unknown
      -> evidence
  -> optional second pass only if aggregator extraction looks incomplete
```

The task tells the LLM:

- the goal
- the target category
- allowed entity types
- decision rules
- output requirements

Allowed entity types:

```text
provider
aggregator
partner_only
unknown
```

Rules:

- Use only provided retrieved chunks.
- Do not use outside knowledge.
- First identify `page_company`: the company/domain represented by the
  retrieved page itself.
- Then extract `referenced_entities`: companies, partners, providers,
  merchants, banks, telcos, marketplace sellers, websites, or domains mentioned
  in the chunks that could relate to the target category.
- For each referenced entity, capture name, domain, URL, role, financing
  responsibility, target relevance, source URL, and supporting quote where
  present.
- Use the entity map to decide `entity_type`. Entity extraction and page
  classification are treated as interlinked, not separate decisions.
- Return `provider` only if the company/domain directly offers, finances,
  underwrites, leases, or sells the target service on installments.
- Return `aggregator` for comparison sites, directories, review sites,
  affiliate pages, lead-generation pages, blog/listing pages, or marketplaces
  listing third-party providers.
- Return `partner_only` if the company only says financing is provided by a
  separate partner and the company is not itself the provider.
- For `partner_only` pages, focus the output on the company/domain represented
  by the retrieved page itself. If a separate partner company is named, extract
  concise partner details such as partner name, role, financing responsibility,
  and relationship to the page company/domain.
- For `aggregator` pages, identify all companies referenced in the content and
  extract company names, roles/offers, URLs, domains, website references, and
  relationships to the target category where available.
- Return `unknown` if evidence is missing, ambiguous, unrelated, or
  insufficient.
- Set `extraction_confidence` to confidence that `referenced_entities` is
  complete for the provided chunks.
- Set `needs_additional_extraction` to `true` when the page appears to be an
  aggregator/blog/listing with many company mentions or when referenced entity
  extraction is likely incomplete.
- Every evidence item must include a URL and a short quote copied from the
  provided page text.

### Payload Retrieved Chunks

The payload key is still named `retrieved_pages` for compatibility, but each
entry now represents a retrieved chunk. Each retrieved chunk becomes:

```python
{
    "url": page.url,
    "retrieval_score": page.score,
    "title": page.title,
    "chunk_index": page.chunk_index,
    "chunk_start": page.chunk_start,
    "chunk_end": page.chunk_end,
    "content": page.content_snippet[:max_chars_per_page],
}
```

Default LLM content cap:

```text
4,000 characters per retrieved chunk
```

Because chunks are currently around the runtime chunk size, the whole chunk is
normally included.

### Payload Output Schema

The payload asks for:

```json
{
  "page_company": {
    "name": "string",
    "domain": "string",
    "url": "string",
    "role": "string",
    "financing_responsibility": "string",
    "target_relevance": "string",
    "quote": "string",
    "source_url": "string",
    "notes": "string"
  },
  "referenced_entities": [
    {
      "name": "string",
      "domain": "string",
      "url": "string",
      "role": "string",
      "financing_responsibility": "string",
      "target_relevance": "string",
      "quote": "string",
      "source_url": "string",
      "notes": "string"
    }
  ],
  "entity_type": "provider | aggregator | partner_only | unknown",
  "validated": "true | false | null",
  "confidence": "float between 0 and 1",
  "extraction_confidence": "float between 0 and 1",
  "needs_additional_extraction": "boolean",
  "evidence": [
    {
      "url": "string",
      "quote": "string",
      "reason": "string"
    }
  ],
  "reasoning": "string",
  "partner_details": [
    "string; for partner_only only, details about the separate partner company"
  ],
  "aggregator_company_details": [
    "string; for aggregator only, referenced company names, roles, URLs, domains, or website references"
  ]
}
```

### Optional Referenced-Company Extraction

If `--extract-referenced-companies` is enabled, the first validation call still
runs normally and returns `page_company`, `referenced_entities`, classification,
and evidence. After that, the engine checks whether a second extraction pass is
actually needed.

The second LLM extraction call runs only when all of these are true:

```text
--extract-referenced-companies is enabled
entity_type == "aggregator"
and one of:
  needs_additional_extraction == true
  extraction_confidence < 0.65
  len(referenced_entities) >= 5
```

So `partner_only` is handled by the unified first pass. The second pass is a
refinement path for aggregator/blog/listing pages where the first pass either
found many entities or explicitly says extraction may be incomplete.

For `aggregator`, the second-pass extraction prompt asks for every referenced
company in the retrieved chunks, including names, domains, URLs, website
references, roles/offers, target relevance, quotes, and source URLs where
available.

The second-pass output is parsed into:

```python
ReferencedCompany(
    name: str,
    domain: str = "",
    url: str = "",
    role: str = "",
    financing_responsibility: str = "",
    target_relevance: str = "",
    quote: str = "",
    source_url: str = "",
    notes: str = "",
)
```

These are stored in `ValidationResult.referenced_companies`.

## Validation Step 10: LLM Prompt

`build_llm_prompt(payload)` wraps the payload in an instruction:

```text
You are validating company service evidence.
Return JSON only. Do not include markdown.
```

Then it appends the JSON payload.

The final prompt is sent as the `input` field to the OpenAI Responses API.

## Validation Step 11: LLM Payload Debug Log

Before the API call, `_log_payload()` runs.

If `log_payload_preview=True`, logs include:

- number of retrieved chunks
- debug payload path
- each chunk score
- each chunk source URL and `chunk_index`
- first 300 characters of each chunk content

If `debug_payload_path` exists, it appends one JSONL line:

```json
{
  "timestamp": "2026-06-10T...",
  "company_name": "...",
  "domain": "...",
  "target_category": "...",
  "payload": { ... }
}
```

Default path:

```text
input/llm_payloads.jsonl
```

At run startup, `new.py` clears this file:

```python
prepare_log_file(llm_log_output)
```

So each run starts with an empty payload log.

## Validation Step 12: OpenAI API Call

The default validator is:

```python
OpenAIHTTPValidator(
    model="gpt-4.1-mini",
    endpoint="https://api.openai.com/v1/responses",
    timeout_seconds=60,
)
```

It sends a POST request with:

```json
{
  "model": "gpt-4.1-mini",
  "input": "<prompt>",
  "text": {
    "format": {
      "type": "json_object"
    }
  }
}
```

Headers:

```text
Authorization: Bearer <OPENAI_API_KEY>
Content-Type: application/json
```

If `OPENAI_API_KEY` is missing, the validator raises:

```text
LLMValidationError("OPENAI_API_KEY is not configured")
```

If the request fails, times out, or returns invalid JSON at the HTTP response
level, the validator raises:

```text
LLMValidationError("LLM request failed: ...")
```

There is currently no retry loop in the code.

## Validation Step 13: LLM Response Text Extraction

After the API response is decoded as JSON, `_response_text(data)` extracts the
model output text.

It first checks:

```python
data["output_text"]
```

If not present, it walks:

```python
data["output"][*]["content"][*]
```

and collects content entries whose type is:

```text
output_text
text
```

If no output text exists, it raises:

```text
LLMValidationError("LLM response did not contain output text")
```

## Validation Step 14: LLM Judgment Parsing

The raw output string is parsed by:

```python
parse_llm_judgment(raw)
```

### JSON Extraction

`_extract_json(raw)`:

1. Strips whitespace.
2. If the output starts with markdown code fences, removes backticks.
3. Removes a leading `json` label when present.
4. Finds the first `{` and last `}`.
5. Returns that substring.

This allows the parser to recover if the model wraps JSON in code fences even
though it was told not to.

### Entity Type Validation

The parser requires:

```text
provider
aggregator
partner_only
unknown
```

Any other value raises `LLMValidationError`.

### Validated Normalization

`validated` can be:

```python
True
False
None
```

String values are normalized:

| Raw string | Parsed value |
| --- | --- |
| `"true"` | `True` |
| `"false"` | `False` |
| `"null"` | `None` |
| `"none"` | `None` |
| `"uncertain"` | `None` |
| `"unknown"` | `None` |
| `""` | `None` |

Any invalid value raises `LLMValidationError`.

### Confidence Clamping

Confidence is converted to `float`, then clamped:

```python
confidence = max(0.0, min(confidence, 1.0))
```

So values below `0` become `0.0`, and values above `1` become `1.0`.

### Evidence Parsing

Every dictionary inside the returned `evidence` list becomes:

```python
EvidenceQuote(
    url=str(item.get("url", "")),
    quote=str(item.get("quote", "")),
    reason=str(item.get("reason", "")),
)
```

Non-dictionary evidence items are ignored.

### LLMValidationJudgment Output

The parser returns:

```python
LLMValidationJudgment(
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
```

## Validation Step 15: Mapping Judgment to Final Result

Back in `validate_company_domain()`, the LLM judgment is mapped directly into:

```python
ValidationResult(
    validated=judgment.validated,
    confidence=judgment.confidence,
    entity_type=judgment.entity_type,
    evidence_pages=[item.url for item in judgment.evidence] or [page.url for page in retrieved],
    supporting_snippets=[item.quote for item in judgment.evidence] or [page.content_snippet for page in retrieved],
    reasoning=judgment.reasoning or "LLM returned no reasoning.",
    page_company=judgment.page_company,
    referenced_entities=judgment.referenced_entities,
    extraction_confidence=judgment.extraction_confidence,
    needs_additional_extraction=judgment.needs_additional_extraction,
    partner_details=judgment.partner_details,
    aggregator_company_details=judgment.aggregator_company_details,
)
```

Important:

- There is no local threshold after the LLM.
- There is no weighted score formula after the LLM.
- There is no rule that changes `validated` based on confidence.
- If the LLM returns evidence quotes, those quotes are used.
- If the LLM returns no evidence, the retrieved page URLs and snippets are used
  as fallback evidence.
- If the LLM returns no reasoning, the fallback reasoning is:

  ```text
  LLM returned no reasoning.
  ```
- `partner_details` is populated from the LLM only when returned, mainly for
  `partner_only` judgments.
- `aggregator_company_details` is populated from the LLM only when returned,
  mainly for `aggregator` judgments.
- `page_company` and `referenced_entities` come from the unified first pass.
- `referenced_companies` comes only from the optional second pass, and that
  pass is now used only for aggregator pages that look dense or incomplete.

## Final Output Dataclass

The final result type is:

```python
ValidationResult(
    validated: bool | None,
    confidence: float,
    entity_type: EntityType,
    evidence_pages: list[str],
    supporting_snippets: list[str],
    reasoning: str,
    page_company: ReferencedCompany | None,
    referenced_entities: list[ReferencedCompany],
    extraction_confidence: float,
    needs_additional_extraction: bool,
    partner_details: list[str],
    aggregator_company_details: list[str],
    referenced_companies: list[ReferencedCompany],
)
```

Field meanings:

| Field | Meaning |
| --- | --- |
| `validated` | `True`, `False`, or `None` if unknown/uncertain. |
| `confidence` | LLM confidence from `0.0` to `1.0`. |
| `entity_type` | `provider`, `aggregator`, `partner_only`, or `unknown`. |
| `evidence_pages` | URLs supporting the result. |
| `supporting_snippets` | Evidence quotes or fallback retrieved snippets. |
| `reasoning` | LLM reasoning or a safeguard/fallback reason. |
| `page_company` | Entity object for the company/domain represented by the retrieved page itself. |
| `referenced_entities` | Entity objects extracted in the unified first pass. |
| `extraction_confidence` | LLM confidence that first-pass entity extraction is complete for the provided chunks. |
| `needs_additional_extraction` | First-pass signal that an aggregator/listing page likely needs second-pass extraction. |
| `partner_details` | Partner-company details extracted for `partner_only` results. |
| `aggregator_company_details` | Referenced company names/details/URLs extracted for `aggregator` results. |
| `referenced_companies` | Structured second-pass extraction results, only populated when `--extract-referenced-companies` is enabled and an aggregator result has many mentions, low extraction confidence, or `needs_additional_extraction=true`. |

## CSV Output

Default path:

```text
input/validation_results.csv
```

At startup, the file is created/overwritten and the header is written:

```python
prepare_csv_results(csv_output)
```

CSV columns:

```text
domain
validated
confidence
entity_type
evidence_pages
supporting_snippets
reasoning
page_company
referenced_entities
extraction_confidence
needs_additional_extraction
partner_details
aggregator_company_details
referenced_companies
```

For each completed company, `append_csv_result()` appends one row and flushes
the file handle.

CSV row mapping:

```python
{
    "domain": domain,
    "validated": "" if result.validated is None else result.validated,
    "confidence": result.confidence,
    "entity_type": result.entity_type,
    "evidence_pages": " | ".join(result.evidence_pages),
    "supporting_snippets": " | ".join(result.supporting_snippets),
    "reasoning": result.reasoning,
    "page_company": json.dumps(asdict(result.page_company) if result.page_company else None, ensure_ascii=False),
    "referenced_entities": json.dumps([asdict(company) for company in result.referenced_entities], ensure_ascii=False),
    "extraction_confidence": result.extraction_confidence,
    "needs_additional_extraction": result.needs_additional_extraction,
    "partner_details": " | ".join(result.partner_details),
    "aggregator_company_details": " | ".join(result.aggregator_company_details),
    "referenced_companies": json.dumps([asdict(company) for company in result.referenced_companies], ensure_ascii=False),
}
```

Special CSV behavior:

- `None` validation is written as a blank value.
- Multiple evidence pages are joined with ` | `.
- Multiple snippets are joined with ` | `.
- Multiple partner or aggregator details are joined with ` | `.
- `page_company`, `referenced_entities`, and `referenced_companies` are written
  as JSON strings in CSV cells.

## JSON Output

Default path:

```text
input/validation_results.json
```

After each company, the entire completed `results` dictionary is rewritten:

```python
write_json_results(json_output, results)
```

The dictionary key is:

```python
company.domain or company.company_name
```

Each `ValidationResult` is converted with:

```python
asdict(value)
```

Output shape:

```json
{
  "orangemali.com": {
    "validated": true,
    "confidence": 0.86,
    "entity_type": "provider",
    "evidence_pages": ["https://..."],
    "supporting_snippets": ["..."],
    "reasoning": "..."
  }
}
```

The JSON file uses:

```python
indent=2
ensure_ascii=False
```

## LLM Payload JSONL Output

Default path:

```text
input/llm_payloads.jsonl
```

This file is cleared at the start of each run.

One line is appended per LLM call.

No line is written when:

- no domain pages exist
- pages exist but no page is indexable
- retrieval returns no evidence
- validation fails before `_log_payload()`

Each line contains:

- timestamp
- company name
- domain
- target category
- full LLM payload

This file is useful for debugging exactly what evidence the LLM saw.

## Data Model Summary

All shared models are in:

```text
input/models.py
```

### DiscoveryResult

```python
DiscoveryResult(
    url: str,
    title: str = "",
    snippet: str = "",
    domain: str = "",
)
```

Used by older/helper discovery flows, not by the main `new.py` run.

### PageContent

```python
PageContent(
    url: str,
    content: str,
    metadata: dict[str, Any] = {},
)
```

Represents one cached HTML page after URL reconstruction and text extraction.

### CompanyData

```python
CompanyData(
    company_name: str,
    company_score: float,
    target_category: str,
    domain: str = "",
)
```

Represents one company row plus the CLI target category.

### TopKPage

```python
TopKPage(
    url: str,
    score: float,
    content_snippet: str,
    title: str = "",
)
```

Represents one retrieved evidence page selected for the LLM.

### EvidenceQuote

```python
EvidenceQuote(
    url: str,
    quote: str,
    reason: str,
)
```

Represents one LLM-selected quote.

### LLMValidationJudgment

```python
LLMValidationJudgment(
    entity_type: EntityType,
    validated: bool | None,
    confidence: float,
    evidence: list[EvidenceQuote],
    reasoning: str,
)
```

Represents the parsed LLM output.

### ValidationResult

```python
ValidationResult(
    validated: bool | None,
    confidence: float,
    entity_type: EntityType,
    evidence_pages: list[str],
    supporting_snippets: list[str],
    reasoning: str,
    partner_details: list[str],
    aggregator_company_details: list[str],
)
```

Represents the final output written to CSV and JSON.

## Supported Entity Types

The allowed values are:

```text
provider
aggregator
partner_only
unknown
```

Meaning:

| Entity type | Meaning |
| --- | --- |
| `provider` | The company/domain directly offers, finances, underwrites, leases, or sells the target service on installments. |
| `aggregator` | The domain lists, compares, reviews, or generates leads for third-party providers. |
| `partner_only` | The domain mentions the service, but a separate partner is the actual provider. |
| `unknown` | Evidence is missing, ambiguous, unrelated, or insufficient. |

## Supported Validation Decisions

`validated` can be:

| Value | Meaning |
| --- | --- |
| `True` | The LLM judged that the company/domain validates for the target. |
| `False` | The LLM judged that the company/domain does not validate. |
| `None` | The system or LLM could not determine a clear answer. |

Safeguard paths usually return:

```python
validated=None
entity_type="unknown"
confidence=0.0
```

## Current Target Categories

Configured in `retrieval.py`:

```text
smartphone_financing
device_financing
bnpl
embedded_finance
leasing
```

Only `smartphone_financing` currently has all of these target-specific extras:

- evidence terms
- URL hints
- product + finance co-occurrence boost

Other configured categories have base queries and labels, but no category
specific evidence boost terms unless added.

## Current Workflow Example

Assume a company CSV row:

```csv
company_name,domain,score
Orange Mali,orangemali.com,0.762
```

and command:

```powershell
python input\new.py --target-category smartphone_financing --top-k 8 --retrieval-mode boosted --pdf-mode extract
```

The flow is:

1. CLI args are parsed.
2. `.env` is loaded if present.
3. `companies_scored.csv` is read.
4. The row becomes:

   ```python
   CompanyData(
       company_name="Orange Mali",
       company_score=0.762,
       target_category="smartphone_financing",
       domain="orangemali.com",
   )
   ```

5. Cached page filenames under `input/pages/` are converted back to URLs.
6. Only pages whose root domain is `orangemali.com` are loaded.
7. Each cached HTML page is converted to text. Cached PDF URLs are extracted
   with `pypdf` in `--pdf-mode extract`, or read with the older raw behavior in
   `--pdf-mode raw`.
8. `new.py` filters the loaded pages again to `orangemali.com`.
9. `validate_company_domain()` starts.
10. The domain index is built for `orangemali.com`.
11. Page text is split into overlapping chunks.
12. Chunk text is tokenized.
13. BM25 is built over chunks using `rank-bm25` if available, otherwise
    `SimpleBM25`.
14. Retrieval queries are generated, including:

    ```text
    smartphone financing
    phone financing
    smartphone installment payments
    ...
    Orange Mali smartphone financing
    Orange Mali financing
    Orange Mali installment plans
    Orange Mali pay later
    ```

15. Every query is scored against every indexed Orange Mali chunk.
16. Positive BM25/overlap scores are summed per chunk across all queries.
17. In `boosted` mode, one local evidence boost is added per chunk. In `bm25`
    mode, no evidence boost is added.
18. The top 8 chunks are selected.
19. The full retrieved chunk text is sent in the LLM payload, with chunk
    metadata such as `chunk_index`, `chunk_start`, and `chunk_end`.
20. The payload is appended to `input/llm_payloads.jsonl`.
21. The prompt is sent to the OpenAI Responses API.
22. The LLM returns JSON.
23. The JSON is parsed into `LLMValidationJudgment`, including
    `partner_details` and `aggregator_company_details` when returned.
24. The judgment is mapped directly to `ValidationResult`.
25. A row is appended to `input/validation_results.csv`.
26. `input/validation_results.json` is rewritten with all completed results.
27. The loop continues to the next company.

## Error and Edge Behavior

### Missing API Key

If retrieval finds evidence but `OPENAI_API_KEY` is missing, the default
validator raises `LLMValidationError`.

There is no local fallback result for this case in `new.py`; the run will fail
unless the caller catches the exception elsewhere.

### Failed LLM Request

Network, timeout, or invalid API-response JSON errors raise
`LLMValidationError`.

There is currently:

- no retry
- no backoff
- no automatic partial result for that company

### Invalid LLM JSON

If the LLM returns invalid JSON or invalid enum values, parsing raises
`LLMValidationError`.

### Existing Output Files

At startup:

- CSV output is overwritten with a fresh header.
- LLM JSONL log is overwritten with an empty file.
- JSON output is not explicitly cleared first, but it is rewritten after the
  first completed company.

### Duplicate Domains

The `results` dictionary key is the domain string. If multiple CSV rows share
the same domain, later rows overwrite earlier rows in JSON because they use the
same key. CSV still gets one appended row per processed company.

## Main Change Points

| Desired change | File/function |
| --- | --- |
| Add or edit target base queries | `retrieval.py`, `BASE_QUERIES_BY_CATEGORY` |
| Add readable target label | `retrieval.py`, `TARGET_LABELS_BY_CATEGORY` |
| Add evidence boost terms | `retrieval.py`, `EVIDENCE_TERMS_BY_CATEGORY` |
| Add URL boost hints | `retrieval.py`, `URL_HINTS_BY_CATEGORY` |
| Change chunk size/overlap/minimum | `domain_index.py`, `DomainIndexCache` defaults and `chunk_page_content()` |
| Change default top K | `new.py`, `--top-k` default |
| Change retrieval scoring mode options | `new.py`, `--retrieval-mode`; `retrieval.py`, `retrieve_top_k()` |
| Change PDF extraction behavior | `new.py`, `--pdf-mode`; `validation_engine.py`, `load_page_contents()` and `extract_pdf_text()` |
| Change referenced-company extraction behavior | `new.py`, `--extract-referenced-companies`; `llm_validator.py`, `build_referenced_company_payload()` |
| Change LLM model | `llm_validator.py`, `OpenAIHTTPValidator.__init__` |
| Change unified entity-aware validation prompt | `llm_validator.py`, `build_llm_payload()` and `parse_llm_judgment()` |
| Change second-pass extraction trigger thresholds | `validation_engine.py`, `AGGREGATOR_MANY_ENTITY_THRESHOLD` and `LOW_EXTRACTION_CONFIDENCE` |
| Change prompt or task rules | `llm_validator.py`, `build_llm_payload()`, `build_llm_prompt()`, and `build_referenced_company_payload()` |
| Change CSV columns | `new.py`, `CSV_FIELDNAMES` and `csv_row()` |
| Change JSON shape | `new.py`, `write_json_results()` or `models.py` |
| Change no-evidence fallback behavior | `validation_engine.py` |
| Change domain normalization | `domain_grouping.py` |
| Change tokenization or BM25 fallback | `domain_index.py` |

## Adding a New Target Category

For a new target such as `solar_paygo`, add entries in `retrieval.py`.

Minimum useful configuration:

```python
BASE_QUERIES_BY_CATEGORY["solar_paygo"] = [
    "pay as you go solar",
    "solar financing",
    "solar installment payments",
]

TARGET_LABELS_BY_CATEGORY["solar_paygo"] = "pay as you go solar"
```

Recommended full configuration:

```python
EVIDENCE_TERMS_BY_CATEGORY["solar_paygo"] = [
    "solar",
    "paygo",
    "pay as you go",
    "installment",
    "monthly payment",
    "financing",
]

PRIORITY_TERMS_BY_CATEGORY["solar_paygo"] = [
    "pay as you go",
    "installment",
    "monthly payment",
    "financing",
]

URL_HINTS_BY_CATEGORY["solar_paygo"] = [
    "solar",
    "paygo",
    "financing",
]
```

If no category is added, the pipeline still runs because it falls back to the
target category with underscores replaced by spaces. Retrieval quality will
usually be weaker.

## Changing Country or Market

Country is not a single hardcoded setting. It mainly comes from:

- `companies_scored.csv`
- cached HTML pages under `input/pages/`
- language-specific retrieval terms in `retrieval.py`

To change from Mali to another country:

1. Update `companies_scored.csv` with companies/domains for the new country.
2. Put cached HTML pages for those domains under `input/pages/`.
3. Keep or change `--target-category`.
4. Add local-language terms to:

   ```text
   BASE_QUERIES_BY_CATEGORY
   EVIDENCE_TERMS_BY_CATEGORY
   PRIORITY_TERMS_BY_CATEGORY
   URL_HINTS_BY_CATEGORY
   ```

For Francophone countries, the current French terms may help. For other
markets, adding local-language terms will matter more.

## What Is New / Current Now

The current architecture is domain-aware and evidence-first:

- Pages are grouped and validated by root domain.
- The LLM is only shown retrieved chunks from cached pages on the company
  domain.
- Pages are chunked before BM25 indexing, so retrieval works at chunk level
  rather than whole-page level.
- BM25 scores are summed across all matching queries per chunk.
- `--retrieval-mode bm25` uses only BM25/overlap scores.
- `--retrieval-mode boosted` adds one evidence boost per chunk after BM25
  summing.
- `--pdf-mode extract` uses `pypdf` for PDF URLs; `--pdf-mode raw` preserves
  the older raw cached-text behavior.
- Retrieval uses target-aware queries instead of one generic search phrase.
- Smartphone financing has special French/English evidence terms.
- Smartphone financing has URL hints for pages like `pret-smartphone`,
  `prt-smartphone`, and `smartphones-a-credit`.
- The old focused context-window step has been removed; retrieved chunks are
  sent directly to the LLM.
- The first LLM pass is now unified and entity-aware: it returns
  `page_company`, `referenced_entities`, page classification, and evidence in
  one response.
- `partner_only` results can include `partner_details`.
- `aggregator` results can include `aggregator_company_details` for companies,
  URLs, domains, and website references mentioned in the content.
- With `--extract-referenced-companies`, aggregator results get a second
  extraction pass only when the first pass reports many referenced entities,
  low extraction confidence, or `needs_additional_extraction=true`.
- LLM payloads are logged to JSONL for inspection.
- CSV and JSON outputs are written incrementally after every company.
- Empty/no-evidence cases return clear `unknown` results instead of crashing or
  forcing an LLM call.

## Test Command

Run:

```powershell
python -m unittest discover -s tests -v
```

The tests use mocked validators for LLM-dependent behavior, so they do not need
network access or a real OpenAI API key for those mocked paths.
