import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from llm_validator import (
    build_llm_payload,
    build_llm_prompt,
    build_referenced_company_payload,
    parse_referenced_companies,
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
        self.assertIn("page_company", payload["output_schema"])
        self.assertIn("referenced_entities", payload["output_schema"])
        self.assertIn("partner_details", payload["output_schema"])

    def test_prompt_classifies_partner_only_as_unknown(self):
        payload = build_llm_payload(
            CompanyData("Retailer", 0.8, "smartphone_financing", "retailer.test"),
            [TopKPage("https://retailer.test/phones", 1.5, "Financing provided by BankCo.")],
        )
        prompt = build_llm_prompt(payload)

        self.assertNotIn('"partner_only"', prompt)
        self.assertIn("Return unknown when the page only says financing is provided", prompt)
        self.assertIn("separate partner or third party", prompt)
        self.assertIn("page company/domain is not itself the direct financing provider", prompt)

    def test_prompt_requires_direct_smartphone_financing_for_provider(self):
        payload = build_llm_payload(
            CompanyData("Orange", 0.8, "smartphone_financing", "orange.test"),
            [TopKPage("https://orange.test/phones", 1.5, "We offer smartphone financing.")],
        )
        prompt = build_llm_prompt(payload)

        self.assertIn("directly provides smartphone financing", prompt)
        self.assertIn("page company/domain itself", prompt)
        self.assertIn("not through a named partner, third-party lender", prompt)

    def test_prompt_instructs_aggregator_to_extract_referenced_companies(self):
        payload = build_llm_payload(
            CompanyData("Listing Blog", 0.8, "smartphone_financing", "listing.test"),
            [
                TopKPage(
                    "https://listing.test/best-phone-financing",
                    1.5,
                    "Company A offers phone financing at company-a.test. Company B has installments.",
                )
            ],
        )
        prompt = build_llm_prompt(payload)

        self.assertIn("aggregator pages", prompt)
        self.assertIn("all referenced companies", prompt)
        self.assertIn("URLs, domains, website references", prompt)
        self.assertIn("aggregator_company_details", prompt)
        self.assertIn("First identify the page_company", prompt)
        self.assertIn("referenced_entities", prompt)

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

    def test_parse_llm_judgment_maps_legacy_partner_only_to_unknown(self):
        judgment = parse_llm_judgment(
            """
            {
              "entity_type": "partner_only",
              "validated": false,
              "confidence": 0.74,
              "evidence": [],
              "reasoning": "The retailer page says financing is handled by a separate bank.",
              "partner_details": [
                "BankCo provides the financing.",
                {"name": "BankCo", "role": "underwrites installments"}
              ]
            }
            """
        )

        self.assertEqual(judgment.entity_type, "unknown")
        self.assertEqual(judgment.partner_details[0], "BankCo provides the financing.")
        self.assertIn("name: BankCo", judgment.partner_details[1])

    def test_parse_llm_judgment_keeps_aggregator_company_details(self):
        judgment = parse_llm_judgment(
            """
            {
              "page_company": {"name": "Listing Blog", "domain": "listing.test", "role": "aggregator_operator"},
              "referenced_entities": [
                {"name": "Company A", "domain": "company-a.test", "role": "listed provider"}
              ],
              "entity_type": "aggregator",
              "validated": false,
              "confidence": 0.81,
              "extraction_confidence": 0.7,
              "needs_additional_extraction": true,
              "evidence": [],
              "reasoning": "The page lists third-party providers.",
              "aggregator_company_details": [
                "Company A: phone financing; domain company-a.test",
                {"name": "Company B", "url": "https://company-b.test", "role": "installment provider"}
              ]
            }
            """
        )

        self.assertEqual(judgment.entity_type, "aggregator")
        self.assertEqual(judgment.page_company.name, "Listing Blog")
        self.assertEqual(judgment.referenced_entities[0].name, "Company A")
        self.assertEqual(judgment.extraction_confidence, 0.7)
        self.assertIs(judgment.needs_additional_extraction, True)
        self.assertEqual(judgment.aggregator_company_details[0], "Company A: phone financing; domain company-a.test")
        self.assertIn("name: Company B", judgment.aggregator_company_details[1])
        self.assertIn("url: https://company-b.test", judgment.aggregator_company_details[1])

    def test_build_referenced_company_payload_extracts_target_providers_from_chunks(self):
        payload = build_referenced_company_payload(
            CompanyData("Listing Blog", 0.8, "smartphone_financing", "listing.test"),
            [
                TopKPage(
                    "https://listing.test/post",
                    2.0,
                    "Company A offers installments.",
                    chunk_index=3,
                    chunk_start=100,
                    chunk_end=500,
                )
            ],
        )

        self.assertEqual(payload["retrieved_chunks"][0]["chunk_index"], 3)
        rules = " ".join(payload["task"]["rules"])
        self.assertIn("Do not classify the page, domain, or company", rules)
        self.assertIn("The only aim of this call is to extract company names", rules)
        self.assertIn("definite evidence in the chunks", rules)
        self.assertIn("explicitly connects that company", rules)
        self.assertIn("Include companies even when they are mentioned as a partner", rules)
        self.assertIn("Every returned company must include source_url", rules)
        self.assertIn("target_relevance", payload["output_schema"]["referenced_companies"][0])
        self.assertIn("financing_responsibility", payload["output_schema"]["referenced_companies"][0])

    def test_referenced_company_payload_excludes_simple_post_payment(self):
        payload = build_referenced_company_payload(
            CompanyData("PhoneMarket", 0.8, "smartphone_financing", "phonemarket.test"),
            [
                TopKPage(
                    "https://phonemarket.test/pay",
                    2.0,
                    "Bol.com lets customers pay up to 400 euros post-payment after delivery.",
                )
            ],
        )

        rules = " ".join(payload["task"]["rules"])
        self.assertNotIn("deferred payment for smartphones", rules)
        self.assertNotIn("payment financing for smartphones", rules)
        self.assertIn("Do not treat simple checkout post-payment", rules)
        self.assertIn("unless the chunk explicitly describes an installment plan", rules)
        self.assertIn("loan, lease, credit agreement", rules)

    def test_referenced_company_payload_excludes_merchants_with_named_financing_partners(self):
        payload = build_referenced_company_payload(
            CompanyData("Back Market", 0.8, "smartphone_financing", "backmarket.test"),
            [
                TopKPage(
                    "https://coupons.test/backmarket",
                    2.0,
                    "Back Market has flexible payment options: Klarna divides the purchase, Oney finances orders, and PayPal offers 3 installments.",
                )
            ],
        )

        rules = " ".join(payload["task"]["rules"])
        self.assertIn("Do not extract merchants, retailers, shops, marketplaces", rules)
        self.assertIn("whose role is only selling the device", rules)
        self.assertIn("extract the named financing/payment-plan providers", rules)
        self.assertIn("do not extract the merchant", rules)
        self.assertIn("unless the quote explicitly says the merchant itself finances", rules)

    def test_referenced_company_payload_excludes_editorial_page_company(self):
        payload = build_referenced_company_payload(
            CompanyData("Roams", 0.8, "smartphone_financing", "roams.es"),
            [
                TopKPage(
                    "https://roams.es/actualidad/finanzas/iphone-a-plazos",
                    2.0,
                    "Roams analyzes whether financing an iPhone in installments is a good idea. Lowi lets customers finance phones.",
                )
            ],
        )

        rules = " ".join(payload["task"]["rules"])
        self.assertIn("Do not extract the page company/domain itself", rules)
        self.assertIn("editorial, advisory, comparison", rules)
        self.assertIn("Writing about, analyzing, recommending", rules)

    def test_referenced_company_payload_excludes_generic_finance_without_device_anchor(self):
        payload = build_referenced_company_payload(
            CompanyData("ID Finance", 0.8, "smartphone_financing", "idfinance.com"),
            [
                TopKPage(
                    "https://idfinance.com/",
                    2.0,
                    "Plazo helps customers boost purchasing power by providing finance when they need it.",
                )
            ],
        )

        rules = " ".join(payload["task"]["rules"])
        self.assertIn("generic consumer finance", rules)
        self.assertIn("financial wellness apps", rules)
        self.assertIn("unless the same evidence explicitly ties that financing to smartphones", rules)
        self.assertIn("Do not infer smartphone/device financing from broad fintech", rules)

    def test_parse_referenced_companies(self):
        companies = parse_referenced_companies(
            """
            {
              "referenced_companies": [
                {
                  "name": "BankCo",
                  "domain": "bankco.test",
                  "url": "https://bankco.test/finance",
                  "role": "financing partner",
                  "financing_responsibility": "underwrites installments",
                  "target_relevance": "smartphone financing",
                  "quote": "Financing provided by BankCo",
                  "source_url": "https://retailer.test/phones",
                  "notes": "Provides installment underwriting"
                },
                {"domain": "missing-name.test"}
              ]
            }
            """
        )

        self.assertEqual(len(companies), 1)
        self.assertEqual(companies[0].name, "BankCo")
        self.assertEqual(companies[0].domain, "bankco.test")
        self.assertEqual(companies[0].role, "financing partner")
        self.assertEqual(companies[0].financing_responsibility, "underwrites installments")
        self.assertEqual(companies[0].target_relevance, "smartphone financing")

if __name__ == "__main__":
    unittest.main()
