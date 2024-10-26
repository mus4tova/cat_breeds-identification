import mlflow
import argparse
from loguru import logger
from datetime import datetime

from data import DataLoader
from model import ModelBuilder

from settings import TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict with antifraud models."
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    # if args.train:
    experiment_name = "Cat_breeds_identification_train"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=datetime.now().strftime("%Y-%m-%d %H:%M")):
        train, test = DataLoader().load_data()
        ModelBuilder(train, test).train()

    # elif args.predict:
    # experiment_name = "Cat_breeds_identification_predict"
    # mlflow.set_experiment(experiment_name)
    # ModelPredictor(data_type).predict_all_models()
    # else:
    #   parser.print_help()


if __name__ == "__main__":
    main()
