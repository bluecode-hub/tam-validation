import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "input"))

from criteria_config import load_criteria_config
from llm_validator import build_referenced_company_payload
from models import CompanyData, TopKPage
from retrieval import generate_retrieval_queries


class CriteriaConfigTests(unittest.TestCase):
    def test_loads_yaml_criteria_and_drives_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pbt_resin.yaml"
            path.write_text(
                """
criteria:
  name: pbt_resin_manufacturer
  label: PBT resin or pellets manufacturer
  description: Companies that manufacture PBT resin or pellets.
  retrieval_terms:
    - PBT resin
    - polybutylene terephthalate
  company_query_templates:
    - "{company} {label}"
    - "{company} PBT pellets"
  complete_match:
    - Company manufactures PBT resin.
  partial_match:
    - Company distributes PBT resin but manufacturing is unclear.
  reject:
    - Company only uses PBT in finished products.
""",
                encoding="utf-8",
            )

            criteria = load_criteria_config(path)
            queries = generate_retrieval_queries(
                CompanyData("ChemCo", 0.8, criteria.name, "chemco.test"),
                criteria,
            )

        self.assertIn("PBT resin", queries)
        self.assertIn("ChemCo PBT resin or pellets manufacturer", queries)
        self.assertIn("ChemCo PBT pellets", queries)

    def test_referenced_company_prompt_uses_criteria_rules(self):
        criteria = load_criteria_config(
            Path(__file__).resolve().parents[1] / "input" / "configs" / "smartphone_financing.yaml"
        )

        payload = build_referenced_company_payload(
            CompanyData("PhoneCo", 0.8, criteria.name, "phoneco.test"),
            [TopKPage("https://phoneco.test", 1.0, "PhoneCo offers installments for phones.")],
            criteria=criteria,
        )
        rules = "\n".join(payload["task"]["rules"])

        self.assertIn("Target criteria: smartphone financing", rules)
        self.assertIn("Definite evidence means", rules)
        self.assertEqual(payload["criteria"]["name"], "smartphone_financing")


if __name__ == "__main__":
    unittest.main()
