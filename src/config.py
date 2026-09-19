from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "gas_sensor_array_drift_dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "gas_sensor_drift.csv"