import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agritech.config import ARTIFACTS_REPORTS_DIR, DATA_DIR


def build_crop_factor_summary(output_dir: Path = ARTIFACTS_REPORTS_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(DATA_DIR / "crop_yield.csv")
    sample_size = min(len(frame), 50000)
    sample = frame.sample(n=sample_size, random_state=42).copy()

    sample["Fertilizer_Used"] = sample["Fertilizer_Used"].astype(int)
    sample["Irrigation_Used"] = sample["Irrigation_Used"].astype(int)

    numeric_features = [
        "Rainfall_mm",
        "Temperature_Celsius",
        "Fertilizer_Used",
        "Irrigation_Used",
        "Days_to_Harvest",
        "Yield_tons_per_hectare",
    ]
    categorical_features = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    pipeline.fit(sample)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    components = pipeline.named_steps["pca"].components_
    top_features = []
    for index, component in enumerate(components, start=1):
        top_indices = component.argsort()[-8:][::-1]
        top_features.append(
            {
                "component": f"PC{index}",
                "top_features": [feature_names[item] for item in top_indices],
            }
        )

    summary = {
        "sample_size": int(sample_size),
        "explained_variance_ratio": pipeline.named_steps["pca"].explained_variance_ratio_.round(4).tolist(),
        "top_features": top_features,
    }

    summary_path = output_dir / "crop_factor_pca_summary.json"
    loadings_path = output_dir / "crop_factor_pca_loadings.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    loadings = pd.DataFrame(components.T, index=feature_names, columns=["PC1", "PC2"])
    loadings = loadings.sort_values("PC1", key=lambda series: series.abs(), ascending=False)
    loadings.to_csv(loadings_path)

    return summary_path, loadings_path


if __name__ == "__main__":
    summary_path, loadings_path = build_crop_factor_summary()
    print(f"summary={summary_path}")
    print(f"loadings={loadings_path}")
