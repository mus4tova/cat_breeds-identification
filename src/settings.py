import os.path
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data2"
# BREEDS_LIST = os.listdir(DATA_DIR)
BREEDS_LIST = ["Abyssinian", "Bengal", "Bombay"]
MODEL_DIR = BASE_DIR / "model"
PREDICT_DIR = BASE_DIR / "predict"
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

EPOCHS = 2
