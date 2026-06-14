import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from agritech.config import ARTIFACTS_DATA_DIR, DATA_DIR

# Certains pays ont des orthographes legerement differentes selon les fichiers.
# Cette table d'alias aide les jointures a retrouver un meme pays sous une cle unique.
AREA_ALIASES = {
    "bolivia plurinational state of": "bolivia",
    "cote divoire": "cote d ivoire",
    "democratic republic of the congo": "democratic republic of congo",
    "iran islamic republic of": "iran",
    "lao peoples democratic republic": "laos",
    "republic of moldova": "moldova",
    "syrian arab republic": "syria",
    "the former yugoslav republic of macedonia": "north macedonia",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "united republic of tanzania": "tanzania",
    "united states of america": "united states",
    "venezuela bolivarian republic of": "venezuela",
    "viet nam": "vietnam",
}


def normalize_area_key(value: str) -> str:
    # On retire les accents et la ponctuation pour que les noms de pays
    # provenant de plusieurs fichiers puissent etre rapproches lors de la fusion.
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return AREA_ALIASES.get(normalized, normalized)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # On travaille sur une copie pour eviter de modifier le dataframe d'origine.
    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def load_historical_yield(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Ce fichier contient la cible a predire : le rendement par zone et par annee.
    frame = _prepare_frame(pd.read_csv(data_dir / "yield.csv"))
    frame = frame.rename(columns={"Value": "hg_ha_yield"})
    frame = frame[["Area", "Item", "Year", "hg_ha_yield"]].copy()
    frame["area_key"] = frame["Area"].map(normalize_area_key)
    return frame


def load_rainfall(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # On conserve seulement les colonnes utiles a la jointure finale.
    frame = _prepare_frame(pd.read_csv(data_dir / "rainfall.csv"))
    frame["area_key"] = frame["Area"].map(normalize_area_key)
    frame["average_rain_fall_mm_per_year"] = pd.to_numeric(
        frame["average_rain_fall_mm_per_year"], errors="coerce"
    )
    return frame[["area_key", "Year", "average_rain_fall_mm_per_year"]]


def load_pesticides(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # On renomme la colonne generique "Value" en un nom metier plus explicite.
    frame = _prepare_frame(pd.read_csv(data_dir / "pesticides.csv"))
    frame["area_key"] = frame["Area"].map(normalize_area_key)
    frame = frame.rename(columns={"Value": "pesticides_tonnes"})
    frame["pesticides_tonnes"] = pd.to_numeric(frame["pesticides_tonnes"], errors="coerce")
    return frame[["area_key", "Year", "pesticides_tonnes"]]


def load_temperature(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Ce fichier utilise des noms de colonnes differents, donc on les aligne.
    frame = _prepare_frame(pd.read_csv(data_dir / "temp.csv"))
    frame = frame.rename(columns={"year": "Year", "country": "Area"})
    frame["area_key"] = frame["Area"].map(normalize_area_key)
    frame["avg_temp"] = pd.to_numeric(frame["avg_temp"], errors="coerce")
    return frame[["area_key", "Year", "avg_temp"]]


def build_master_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    # Chaque source est chargee separement pour garder une logique simple a suivre.
    yield_frame = load_historical_yield(data_dir)
    rainfall_frame = load_rainfall(data_dir)
    pesticide_frame = load_pesticides(data_dir)
    temperature_frame = load_temperature(data_dir)

    # La cle de fusion est composee du pays normalise et de l'annee.
    merged = yield_frame.merge(rainfall_frame, on=["area_key", "Year"], how="left")
    merged = merged.merge(pesticide_frame, on=["area_key", "Year"], how="left")
    merged = merged.merge(temperature_frame, on=["area_key", "Year"], how="left")

    # On garde la periode recente qui recoupe bien les fichiers climatiques externes.
    merged = merged[merged["Year"] >= 1990].copy()
    required_columns = [
        "hg_ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
    missing_share_after_drop = {
        column: float(merged[column].isna().mean()) for column in required_columns
    }
    # Les lignes sans variable critique ne peuvent pas servir a l'entrainement final.
    merged = merged.dropna(subset=required_columns)
    merged = merged.drop(columns=["area_key"]).sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    # Ce rapport permet de justifier les choix de fusion sans relire le code.
    merge_report = {
        "rows_after_merge": int(len(merged)),
        "year_min": int(merged["Year"].min()) if not merged.empty else None,
        "year_max": int(merged["Year"].max()) if not merged.empty else None,
        "areas": int(merged["Area"].nunique()) if not merged.empty else 0,
        "items": int(merged["Item"].nunique()) if not merged.empty else 0,
        "duplicates_after_merge": int(merged.duplicated().sum()),
        "missing_share_after_drop": missing_share_after_drop,
    }
    return merged, merge_report


def save_master_dataset(output_dir: Path = ARTIFACTS_DATA_DIR) -> tuple[Path, Path]:
    # On sauvegarde a la fois le dataset nettoye et un resume JSON de documentation.
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dataset, merge_report = build_master_dataset()
    dataset_path = output_dir / "consolidated_yield.csv"
    report_path = output_dir / "merge_report.json"
    master_dataset.to_csv(dataset_path, index=False)
    report_path.write_text(json.dumps(merge_report, indent=2), encoding="utf-8")
    return dataset_path, report_path


if __name__ == "__main__":
    dataset_path, report_path = save_master_dataset()
    print(f"dataset={dataset_path}")
    print(f"report={report_path}")
