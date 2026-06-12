import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from llm_validator import (
    build_llm_payload,
    parse_llm_judgment,
)
from models import CompanyData, TopKPage


class LLMValidatorTests(unittest.TestCase):
    def test_build_payload_contains_company_and_pages(self):
        payload = build_llm_payload(
            CompanyData("Orange", 0.8, "smartphone_financing", "orange.test"),
            [TopKPage("https://orange.test/phones", 1.5, "We offer smartphone financing.")],
        )
        self.assertEqual(payload["company"]["company_name"], "Orange")
        self.assertEqual(payload["retrieved_pages"][0]["url"], "https://orange.test/phones")
        self.assertIn("output_schema", payload)

    def test_parse_llm_judgment(self):
        judgment = parse_llm_judgment(
            """
            {
              "entity_type": "provider",
              "validated": true,
              "confidence": 0.86,
              "evidence": [
                {
                  "url": "https://orange.test/phones",
                  "quote": "We offer smartphone financing.",
                  "reason": "Direct provider evidence"
                }
              ],
              "reasoning": "The company directly offers financing."
            }
            """
        )
        self.assertEqual(judgment.entity_type, "provider")
        self.assertIs(judgment.validated, True)
        self.assertEqual(judgment.evidence[0].url, "https://orange.test/phones")

    def test_parse_llm_judgment_accepts_string_boolean(self):
        judgment = parse_llm_judgment(
            """
            {
              "entity_type": "provider",
              "validated": "true",
              "confidence": "0.86",
              "evidence": [],
              "reasoning": "The company directly offers financing."
            }
            """
        )
        self.assertIs(judgment.validated, True)
        self.assertEqual(judgment.confidence, 0.86)

if __name__ == "__main__":
    unittest.main()
