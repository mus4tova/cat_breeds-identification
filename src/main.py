from loguru import logger

from data import DataLoader
from test import ModelTester
from model import ModelBuilder

# from predict import ModelPredictor

import warnings

warnings.filterwarnings("ignore")


def main():
    train, test, y_test = DataLoader().get_data()
    model = ModelBuilder(train).train()
    ModelTester(test, y_test, model).test()
    # ModelPredictor().predict()


if __name__ == "__main__":
    main()
