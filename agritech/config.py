from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DATA"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DATA_DIR = ARTIFACTS_DIR / "data"
ARTIFACTS_MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACTS_REPORTS_DIR = ARTIFACTS_DIR / "reports"
MLRUNS_DIR = ROOT_DIR / "mlruns"
