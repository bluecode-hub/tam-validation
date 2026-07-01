import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from models import CompanyData, PageContent, ReferencedCompany
from validation_engine import extract_pdf_text, load_page_contents, validate_company_domain


class StubValidator:
    def __init__(self, judgment, referenced_companies=None):
        self.judgment = judgment
        self.referenced_companies = referenced_companies or []
        self.validate_calls = []
        self.extract_calls = []

    def validate(self, company, retrieved_pages):
        self.validate_calls.append((company, retrieved_pages))
        return self.judgment

    def extract_referenced_companies(self, company, retrieved_pages, entity_type):
        self.extract_calls.append((company, retrieved_pages, entity_type))
        return self.referenced_companies


class StubPdfPage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class StubPdfReader:
    def __init__(self, stream):
        self.pages = [
            StubPdfPage("Smartphone financing terms."),
            StubPdfPage("Customers can pay monthly installments."),
        ]


class LengthCheckingPdfReader:
    def __init__(self, stream):
        data = stream.read()
        if len(data) != 24:
            raise ValueError("truncated PDF bytes")
        self.pages = [
            StubPdfPage("Smartphone financing terms."),
            StubPdfPage("Customers can pay monthly installments."),
        ]


class ValidationEngineTests(unittest.TestCase):
    def test_validation_extracts_provider_companies_without_classification(self):
        validator = StubValidator(
            None,
            referenced_companies=[
                ReferencedCompany(
                    name="Orange",
                    domain="orange.test",
                    role="smartphone financing provider",
                    source_url="https://orange.test/phones",
                )
            ],
        )
        pages = [
            PageContent(
                "https://orange.test/phones",
                "We offer smartphone financing. Customers can apply for phone credit and pay monthly installments.",
            )
        ]
        result = validate_company_domain(
            CompanyData("Orange", 0.8, "smartphone_financing", "orange.test"),
            pages,
            validator=validator,
        )

        self.assertEqual(validator.validate_calls, [])
        self.assertEqual(len(validator.extract_calls), 1)
        self.assertEqual(validator.extract_calls[0][2], "criteria_extraction")
        self.assertIsNone(result.validated)
        self.assertEqual(result.entity_type, "unknown")
        self.assertEqual(result.evidence_pages, ["https://orange.test/phones"])
        self.assertEqual(result.referenced_companies[0].name, "Orange")
        self.assertEqual(result.referenced_entities[0].name, "Orange")

    def test_validation_extracts_provider_names_from_aggregator_content(self):
        validator = StubValidator(
            None,
            referenced_companies=[
                ReferencedCompany(
                    name="ProviderCo",
                    domain="provider.test",
                    role="installment provider",
                    source_url="https://compare.test/bnpl",
                )
            ],
        )
        pages = [
            PageContent(
                "https://compare.test/bnpl",
                "Compare smartphone financing from ProviderCo at provider.test.",
            )
        ]
        result = validate_company_domain(
            CompanyData("Compare", 0.8, "smartphone_financing", "compare.test"),
            pages,
            validator=validator,
        )

        self.assertEqual(validator.validate_calls, [])
        self.assertEqual(len(validator.extract_calls), 1)
        self.assertEqual(validator.extract_calls[0][2], "criteria_extraction")
        self.assertIsNone(result.validated)
        self.assertEqual(result.entity_type, "unknown")
        self.assertEqual(result.referenced_companies[0].name, "ProviderCo")

    def test_validation_extracts_referenced_companies_even_without_flag(self):
        validator = StubValidator(
            None,
            referenced_companies=[
                ReferencedCompany(
                    name="ProviderCo",
                    domain="provider.test",
                    role="installment provider",
                    source_url="https://compare.test/bnpl",
                )
            ],
        )
        result = validate_company_domain(
            CompanyData("Compare", 0.8, "smartphone_financing", "compare.test"),
            [
                PageContent(
                    "https://compare.test/bnpl",
                    "Compare smartphone financing from ProviderCo at provider.test.",
                )
            ],
            validator=validator,
        )

        self.assertEqual(len(validator.extract_calls), 1)
        self.assertEqual(validator.extract_calls[0][2], "criteria_extraction")
        self.assertEqual(result.referenced_companies[0].name, "ProviderCo")

    def test_validation_deprecated_extraction_flag_does_not_change_behavior(self):
        validator = StubValidator(
            None,
            referenced_companies=[ReferencedCompany(name="ProviderCo")],
        )
        result = validate_company_domain(
            CompanyData("Compare", 0.8, "smartphone_financing", "compare.test"),
            [PageContent("https://compare.test/bnpl", "Compare smartphone financing from ProviderCo.")],
            validator=validator,
            extract_referenced_companies=True,
        )

        self.assertEqual(len(validator.extract_calls), 1)
        self.assertEqual(validator.extract_calls[0][2], "criteria_extraction")
        self.assertEqual(result.referenced_companies[0].name, "ProviderCo")

    def test_validation_does_not_skip_criteria_extraction_for_complete_aggregator_like_content(self):
        validator = StubValidator(
            None,
            referenced_companies=[ReferencedCompany(name="ProviderCo")],
        )
        result = validate_company_domain(
            CompanyData("Compare", 0.8, "smartphone_financing", "compare.test"),
            [PageContent("https://compare.test/bnpl", "Compare smartphone financing from ProviderCo and OtherCo.")],
            validator=validator,
            extract_referenced_companies=True,
        )

        self.assertEqual(len(validator.extract_calls), 1)
        self.assertEqual(result.referenced_companies[0].name, "ProviderCo")

    def test_validation_handles_pages_without_indexable_text(self):
        result = validate_company_domain(
            CompanyData("IMF", 0.05, "smartphone_financing", "imf.org"),
            [PageContent("https://imf.org/report", " ")],
            validator=StubValidator(None),
        )
        self.assertIsNone(result.validated)
        self.assertEqual(result.entity_type, "unknown")
        self.assertIn("indexable text", result.reasoning)

    def test_validation_handles_no_retrieved_evidence(self):
        result = validate_company_domain(
            CompanyData("Unrelated", 0.05, "smartphone_financing", "example.org"),
            [PageContent("https://example.org/report", "macroeconomic fiscal policy")],
            validator=StubValidator(None),
        )
        self.assertIsNone(result.validated)
        self.assertEqual(result.entity_type, "unknown")
        self.assertIn("No relevant evidence", result.reasoning)

    def test_load_page_contents_extracts_pdf_urls_with_pypdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pages_dir = Path(temp_dir)
            cached_pdf = pages_dir / "example.com_docs_terms.pdf_abc123.html"
            cached_pdf.write_bytes(b"%PDF-1.4 fake test bytes")

            with patch("validation_engine.PdfReader", StubPdfReader):
                pages = load_page_contents(pages_dir, domains=["example.com"])

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].url, "https://example.com/docs/terms.pdf")
        self.assertEqual(pages[0].metadata["source_type"], "pdf")
        self.assertIn("Smartphone financing terms", pages[0].content)
        self.assertIn("monthly installments", pages[0].content)

    def test_extract_pdf_text_does_not_truncate_pdf_bytes_before_parsing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cached_pdf = Path(temp_dir) / "cached.pdf.html"
            cached_pdf.write_bytes(b"%PDF-1.4 fake test bytes")

            with patch("validation_engine.PdfReader", LengthCheckingPdfReader):
                content = extract_pdf_text(cached_pdf, max_chars=12)

        self.assertEqual(content, "Smartphone f")

    def test_load_page_contents_can_keep_pdf_cache_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pages_dir = Path(temp_dir)
            cached_pdf = pages_dir / "example.com_docs_terms.pdf_abc123.html"
            cached_pdf.write_text("raw cached pdf text with smartphone financing", encoding="utf-8")

            pages = load_page_contents(
                pages_dir,
                domains=["example.com"],
                extract_pdfs=False,
            )

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].url, "https://example.com/docs/terms.pdf")
        self.assertEqual(pages[0].metadata["source_type"], "pdf_raw")
        self.assertIn("smartphone financing", pages[0].content)


if __name__ == "__main__":
    unittest.main()
