import pandas as pd

from agritech.data import preprocess
from agritech.data.preprocess import build_master_dataset, normalize_area_key, save_master_dataset


def test_normalize_area_key_handles_accents_and_aliases() -> None:
    # Des orthographes differentes d'un meme pays doivent produire une meme cle de fusion.
    assert normalize_area_key("Cote d'Ivoire") == "cote d ivoire"
    assert normalize_area_key("United States of America") == "united states"


def test_build_master_dataset_returns_expected_columns() -> None:
    # La table finale doit contenir les colonnes attendues par la pipeline de modele.
    dataset, report = build_master_dataset()
    expected_columns = {
        "Area",
        "Item",
        "Year",
        "hg_ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    }
    assert expected_columns.issubset(set(dataset.columns))
    assert not dataset.empty
    assert dataset["Year"].min() >= 1990
    assert report["rows_after_merge"] == len(dataset)
    assert isinstance(dataset, pd.DataFrame)


def test_save_master_dataset_writes_expected_artifacts(tmp_path) -> None:
    # Le helper doit persister le CSV consolide et le rapport JSON associe.
    dataset = pd.DataFrame(
        [
            {
                "Area": "Albania",
                "Item": "Wheat",
                "Year": 2013,
                "hg_ha_yield": 30000.0,
                "average_rain_fall_mm_per_year": 1000.0,
                "pesticides_tonnes": 100.0,
                "avg_temp": 18.0,
            }
        ]
    )
    report = {
        "rows_after_merge": 1,
        "year_min": 2013,
        "year_max": 2013,
        "areas": 1,
        "items": 1,
        "duplicates_after_merge": 0,
        "missing_share_after_drop": {
            "hg_ha_yield": 0.0,
            "average_rain_fall_mm_per_year": 0.0,
            "pesticides_tonnes": 0.0,
            "avg_temp": 0.0,
        },
    }

    original_builder = preprocess.build_master_dataset
    preprocess.build_master_dataset = lambda: (dataset, report)
    try:
        dataset_path, report_path = save_master_dataset(output_dir=tmp_path)
    finally:
        preprocess.build_master_dataset = original_builder

    assert dataset_path.exists()
    assert report_path.exists()
    written = pd.read_csv(dataset_path)
    assert len(written) == 1
