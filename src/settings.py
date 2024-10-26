import os.path
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data2"
BREEDS_LIST = os.listdir(DATA_DIR)
MODEL_DIR = BASE_DIR / "model"
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)


# create local MLflow URI
TRACKING_URI = "http://127.0.0.1:5000"

EPOCHS = 20
