import pandas as pd

from agritech.data.preprocess import build_master_dataset, normalize_area_key


def test_normalize_area_key_handles_accents_and_aliases() -> None:
    assert normalize_area_key("Cote d'Ivoire") == "cote d ivoire"
    assert normalize_area_key("United States of America") == "united states"


def test_build_master_dataset_returns_expected_columns() -> None:
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
