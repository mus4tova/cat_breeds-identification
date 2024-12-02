from loguru import logger
from datetime import datetime

from data import DataLoader
from model import ModelBuilder
from predict import ModelPredictor
import warnings

warnings.filterwarnings("ignore")


def main():
    train, test = DataLoader().get_data()
    ModelBuilder(train, test).train()
    # ModelPredictor().predict()


if __name__ == "__main__":
    main()
