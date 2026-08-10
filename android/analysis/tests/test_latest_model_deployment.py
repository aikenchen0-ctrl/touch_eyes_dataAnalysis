import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "full_data_clean_v2_old13_all_118"
EXPERIMENTAL_MODEL_IDS = (
    "deduplicated_clean_v3_probe_patch21_domain_lemon-xiaomi-2308cpxd0c",
    "deduplicated_clean_v3_probe_patch21_domain_tao-oneplus-le2110",
    "deduplicated_clean_v3_probe_patch21_domain_tao-oneplus-phk110",
    "deduplicated_clean_v3_probe_patch21_domain_zhuo-oneplus-le2110",
    "deduplicated_clean_v3_probe_patch21_domain_zhuo-oneplus-phk110",
)
ASSET_DIRECTORY = (
    REPOSITORY_ROOT
    / "android"
    / "aispect-core"
    / "src"
    / "main"
    / "assets"
    / "aispect_impact_models"
)


class LatestModelDeploymentTests(unittest.TestCase):
    def test_cross_device_app_defaults_to_latest_model(self):
        build_file = REPOSITORY_ROOT / "android" / "app" / "build.gradle"

        build_text = build_file.read_text(encoding="utf-8")

        self.assertIn(
            f'buildConfigField "String", "DEFAULT_MODEL_ID", \'"{MODEL_ID}"\'',
            build_text,
        )

    def test_latest_model_is_first_catalog_entry_with_assets(self):
        catalog_path = ASSET_DIRECTORY / "AispectImpactCNNModelCatalog.json"

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        model = catalog[0]

        self.assertEqual(model["id"], MODEL_ID)
        self.assertEqual(
            model["scalerResourceName"],
            f"AispectImpactCNNScaler_{MODEL_ID}",
        )
        self.assertEqual(
            model["weightsResourceName"],
            f"AispectImpactCNNWeights_{MODEL_ID}",
        )
        self.assertTrue(
            (ASSET_DIRECTORY / f"AispectImpactCNNScaler_{MODEL_ID}.json").is_file()
        )
        self.assertTrue(
            (ASSET_DIRECTORY / f"AispectImpactCNNWeights_{MODEL_ID}.json").is_file()
        )

    def test_clean_v3_patch21_models_are_selectable_with_assets(self):
        catalog_path = ASSET_DIRECTORY / "AispectImpactCNNModelCatalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        models_by_id = {model["id"]: model for model in catalog}

        for model_id in EXPERIMENTAL_MODEL_IDS:
            self.assertIn(model_id, models_by_id)
            model = models_by_id[model_id]
            self.assertEqual(
                f"AispectImpactCNNScaler_{model_id}",
                model["scalerResourceName"],
            )
            self.assertEqual(
                f"AispectImpactCNNWeights_{model_id}",
                model["weightsResourceName"],
            )
            self.assertTrue(
                (ASSET_DIRECTORY / f"AispectImpactCNNScaler_{model_id}.json").is_file()
            )
            self.assertTrue(
                (ASSET_DIRECTORY / f"AispectImpactCNNWeights_{model_id}.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
