import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from domain_grouping import group_urls_by_domain, root_domain


class DomainGroupingTests(unittest.TestCase):
    def test_root_domain_normalizes_common_hosts(self):
        self.assertEqual(root_domain("https://www.apple.com/ml/iphone"), "apple.com")
        self.assertEqual(root_domain("shop.example.co.uk/path"), "example.co.uk")

    def test_group_urls_by_domain_deduplicates_urls(self):
        grouped = group_urls_by_domain(
            [
                {"url": "https://www.apple.com/financing", "domain": "www.apple.com"},
                {"url": "https://apple.com/iphone", "domain": "apple.com"},
                {"url": "https://apple.com/iphone", "domain": "apple.com"},
                {"url": "https://shop.samsung.com/offers", "domain": ""},
            ]
        )
        self.assertEqual(
            grouped,
            {
                "apple.com": ["https://www.apple.com/financing", "https://apple.com/iphone"],
                "samsung.com": ["https://shop.samsung.com/offers"],
            },
        )


if __name__ == "__main__":
    unittest.main()
