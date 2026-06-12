import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from models import CompanyData, EvidenceQuote, LLMValidationJudgment, PageContent
from validation_engine import load_page_contents, validate_company_domain


class StubValidator:
    def __init__(self, judgment):
        self.judgment = judgment

    def validate(self, company, retrieved_pages):
        return self.judgment


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


class ValidationEngineTests(unittest.TestCase):
    def test_validation_confirms_direct_provider(self):
        pages = [
            PageContent(
                "https://orange.test/phones",
                "We offer smartphone financing. Customers can apply for phone credit and pay monthly installments.",
            )
        ]
        result = validate_company_domain(
            CompanyData("Orange", 0.8, "smartphone_financing", "orange.test"),
            pages,
            validator=StubValidator(
                LLMValidationJudgment(
                    "provider",
                    True,
                    0.9,
                    [
                        EvidenceQuote(
                            "https://orange.test/phones",
                            "We offer smartphone financing.",
                            "Direct financing language",
                        )
                    ],
                    "Direct provider evidence found.",
                )
            ),
        )
        self.assertIs(result.validated, True)
        self.assertEqual(result.entity_type, "provider")
        self.assertEqual(result.evidence_pages, ["https://orange.test/phones"])

    def test_validation_rejects_aggregator(self):
        pages = [
            PageContent(
                "https://compare.test/bnpl",
                "Compare the best BNPL providers in this marketplace directory with independent reviews.",
            )
        ]
        result = validate_company_domain(
            CompanyData("Compare", 0.8, "smartphone_financing", "compare.test"),
            pages,
            validator=StubValidator(
                LLMValidationJudgment(
                    "aggregator",
                    False,
                    0.9,
                    [
                        EvidenceQuote(
                            "https://compare.test/bnpl",
                            "Compare the best BNPL providers",
                            "Comparison directory language",
                        )
                    ],
                    "Aggregator evidence found.",
                )
            ),
        )
        self.assertIs(result.validated, False)
        self.assertEqual(result.entity_type, "aggregator")

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
