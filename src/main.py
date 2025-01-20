from loguru import logger
import mlflow
from data import DataLoader
from test import ModelTester
from model import ModelBuilder
from datetime import datetime

# from predict import ModelPredictor
from src import settings

import warnings

warnings.filterwarnings("ignore")
mlflow.set_tracking_uri(settings.TRACKING_URI)
mlflow.set_experiment("Cat Breeds Identification")


def main():
    with mlflow.start_run(
        run_name="Train" + "/" + datetime.now().strftime("%Y-%m-%d %H:%M")
    ):
        train, test, y_test = DataLoader().get_data()
        logger.info(f"train:{train}")
        logger.info(f"test:{test}")
        model = ModelBuilder(train).train()
        ModelTester(test, y_test, model).test()
        # ModelPredictor().predict()


if __name__ == "__main__":
    main()
