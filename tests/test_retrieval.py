import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from domain_index import DomainIndexCache
from models import CompanyData, PageContent
from retrieval import FOCUSED_CONTEXT_CHARS, generate_retrieval_queries, html_to_text, retrieve_top_k, snippet_for_query


class RetrievalTests(unittest.TestCase):
    def test_bm25_retrieval_returns_relevant_page_first(self):
        pages = [
            PageContent("https://example.com/about", "We sell accessories and cases."),
            PageContent(
                "https://example.com/finance",
                "Customers can buy a smartphone with monthly installment payments and phone financing.",
            ),
        ]
        index = DomainIndexCache().get_or_build("example.com", pages)
        results = retrieve_top_k(index, ["smartphone financing"], k=1)
        self.assertEqual(results[0].url, "https://example.com/finance")
        self.assertGreater(results[0].score, 0)

    def test_retrieval_can_run_without_evidence_boost(self):
        pages = [
            PageContent(
                "https://example.com/generic",
                "smartphone financing " * 20,
            ),
            PageContent(
                "https://example.com/pret-smartphone",
                "smartphone financing credit pret mensualite telephone tablette",
            ),
        ]
        index = DomainIndexCache().get_or_build("example.com", pages)

        bm25_only = retrieve_top_k(
            index,
            ["smartphone financing"],
            k=1,
            use_evidence_boost=False,
        )
        boosted = retrieve_top_k(
            index,
            ["smartphone financing"],
            k=1,
            use_evidence_boost=True,
        )

        self.assertEqual(bm25_only[0].url, "https://example.com/generic")
        self.assertEqual(boosted[0].url, "https://example.com/pret-smartphone")

    def test_company_queries_follow_target_category(self):
        queries = generate_retrieval_queries(CompanyData("LeaseCo", 0.5, "leasing", "lease.test"))
        self.assertIn("LeaseCo leasing", queries)
        self.assertNotIn("LeaseCo smartphone financing", queries)

    def test_unknown_target_category_gets_generic_query_label(self):
        queries = generate_retrieval_queries(CompanyData("SolarCo", 0.5, "solar_paygo", "solar.test"))
        self.assertIn("solar paygo", queries)
        self.assertIn("SolarCo solar paygo", queries)
        self.assertNotIn("SolarCo smartphone financing", queries)

    def test_extraction_keeps_financing_metadata(self):
        text = html_to_text(
            """
            <html>
              <head>
                <meta name="description" content="Achetez vos smartphones a credit sur 12 mois chez Orange Mali">
              </head>
              <body>Generic navigation Pret smartphone Services Mobile</body>
            </html>
            """
        )
        self.assertIn("smartphones a credit", text)

    def test_snippet_prefers_direct_financing_evidence_over_navigation(self):
        content = (
            "Orange Mali Offres Mobiles Telephones mobiles Pret smartphone Services "
            "Some unrelated header text. "
            "Achetez vos smartphones et tablettes a credit sur plusieurs 12 mois. "
            "Offre disponible uniquement chez Orange Mali."
        )
        snippet = snippet_for_query(content, "pret smartphone", category="smartphone_financing")
        self.assertIn("a credit sur plusieurs 12 mois", snippet)
        self.assertIn("Offre disponible uniquement chez Orange Mali", snippet)

    def test_snippet_uses_larger_focused_context_not_full_page(self):
        content = (
            "before " * 500
            + "Achetez vos smartphones et tablettes a credit sur plusieurs 12 mois. "
            + "Conditions de souscription et mensualite disponibles en agence. "
            + "after " * 500
        )
        snippet = snippet_for_query(content, "smartphone a credit", category="smartphone_financing")
        self.assertIn("mensualite disponibles", snippet)
        self.assertLessEqual(len(snippet), FOCUSED_CONTEXT_CHARS)
        self.assertLess(len(snippet), len(content))


if __name__ == "__main__":
    unittest.main()
