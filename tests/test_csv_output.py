import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from models import ReferencedCompany, ValidationResult
from new import CSV_FIELDNAMES, csv_row, extraction_status


class CSVOutputTests(unittest.TestCase):
    def test_csv_row_prioritizes_criteria_extraction_fields(self):
        result = ValidationResult(
            validated=None,
            confidence=0.0,
            entity_type="unknown",
            evidence_pages=["https://example.test/phones", "https://example.test/phones"],
            supporting_snippets=["Example supports phones in installments."],
            reasoning="Extracted provider company names from retrieved BM25 chunks.",
            referenced_companies=[
                ReferencedCompany(
                    name="ProviderCo",
                    domain="provider.test",
                    role="smartphone financing provider",
                    match_responsibility="Offers phone installments",
                    llm_confidence_score="5",
                    llm_confidence_reasoning="Direct explicit phone financing offer.",
                    quote="ProviderCo offers smartphone financing.",
                    source_url="https://example.test/phones",
                ),
                ReferencedCompany(
                    name="BankCo",
                    role="financing partner",
                    llm_confidence_score="3",
                    llm_confidence_reasoning="Named partner evidence but responsibility is less detailed.",
                    quote="BankCo finances mobile phone purchases.",
                    source_url="https://example.test/phones",
                ),
            ],
        )

        row = csv_row("example.test", result, "smartphone_financing")

        self.assertEqual(row["target_category"], "smartphone_financing")
        self.assertEqual(row["extraction_status"], "matches_found")
        self.assertEqual(row["matched_company_count"], 2)
        self.assertEqual(row["matched_company_names"], "ProviderCo | BankCo")
        self.assertEqual(row["matched_company_llm_confidence_scores"], "5 | 3")
        self.assertIn("Direct explicit phone financing offer.", row["matched_company_llm_confidence_reasoning"])
        self.assertIn("ProviderCo offers smartphone financing.", row["evidence_quotes"])
        self.assertEqual(row["source_urls"], "https://example.test/phones")
        self.assertEqual(row["top_chunk_urls"], "https://example.test/phones")
        self.assertNotIn("entity_type", row)
        self.assertNotIn("validated", row)
        self.assertEqual(set(row), set(CSV_FIELDNAMES))

    def test_extraction_status_distinguishes_empty_results(self):
        result = ValidationResult(
            validated=None,
            confidence=0.0,
            entity_type="unknown",
            evidence_pages=[],
            supporting_snippets=[],
            reasoning="No relevant evidence pages were retrieved for validation.",
        )

        self.assertEqual(extraction_status(result), "no_relevant_chunks")


if __name__ == "__main__":
    unittest.main()
