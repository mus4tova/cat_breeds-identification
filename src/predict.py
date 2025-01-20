import os
from loguru import logger
from PIL import Image, ImageOps


from settings import PREDICT_DIR, DATA_DIR
from data import DataLoader


class ModelPredictor:
    def __init__(self):
        pass

    def predict(self):
        im_list = []
        for file in os.listdir(str(PREDICT_DIR)):
            im = Image.open(str(DATA_DIR) + "\\" + file)
            im_list.append(im)

        x = DataLoader().standardize_images(im_list)
        X = DataLoader().from_list_to_4Dtensor(x)
        X = DataLoader().image_data_generator(X, "test")

        # model = model.load()
        # y_pred = model.predict(X)
