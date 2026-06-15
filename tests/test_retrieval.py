import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from domain_index import DomainIndexCache, chunk_page_content
from models import CompanyData, PageContent
from retrieval import generate_retrieval_queries, html_to_text, retrieve_top_k


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

    def test_retrieval_sums_scores_across_queries(self):
        pages = [
            PageContent("https://example.com/single", "alpha " * 20),
            PageContent("https://example.com/multiple", "alpha beta"),
        ]
        index = DomainIndexCache().get_or_build("example.com", pages)

        results = retrieve_top_k(
            index,
            ["alpha", "beta"],
            k=1,
            use_evidence_boost=False,
        )

        self.assertEqual(results[0].url, "https://example.com/multiple")

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

    def test_index_chunks_content_before_retrieval(self):
        pages = [
            PageContent(
                "https://example.com/long",
                "boring introduction " * 40
                + "smartphone financing monthly installment evidence "
                + "boring footer " * 40,
            )
        ]
        index = DomainIndexCache(chunk_chars=120, chunk_overlap_chars=0).get_or_build("example.com", pages)
        self.assertGreater(len(index.pages), 1)

        results = retrieve_top_k(index, ["smartphone financing"], k=1, use_evidence_boost=False)

        self.assertEqual(results[0].url, "https://example.com/long")
        self.assertIn("smartphone financing", results[0].content_snippet)
        self.assertLess(len(results[0].content_snippet), len(pages[0].content))

    def test_chunk_page_content_preserves_overlap_metadata(self):
        chunks = chunk_page_content(
            PageContent("https://example.com/page", "word " * 900),
            chunk_chars=100,
            overlap_chars=10,
            min_chunk_chars=40,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[1].metadata["chunk_index"], 1)
        self.assertEqual(chunks[0].url, chunks[1].url)

    def test_chunk_page_content_avoids_tiny_trailing_chunks(self):
        chunks = chunk_page_content(
            PageContent("https://example.com/page", "x" * 2600),
            chunk_chars=2000,
            overlap_chars=500,
            min_chunk_chars=1000,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].content), 2600)


if __name__ == "__main__":
    unittest.main()
