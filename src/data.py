import os
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from settings import BREEDS_LIST, DATA_DIR

#print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices("GPU")))


class DataLoader:
    def __init__(self):
        self.breeds_list = BREEDS_LIST
        logger.info(f"breeds_dir: {self.breeds_list}")

    def load_data(self):
        for breed in self.breeds_list:
            logger.info(f"breeds: {breed}")
            x, y = self.from_file_to_list(breed)

        X = self.from_list_to_4Dtensor(x)
        y = self.transform_target(y)



        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train = datagen.flow(X_train, y_train, batch_size=1)
        test = test_datagen.flow(X_test, y_test, batch_size=1)
        return train, test

    def transform_target(self, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y

    def from_list_to_4Dtensor(self, x: list[list[list[float]]]) -> np.ndarray[np.ndarray[np.ndarray[np.ndarray[float]]]]:
        X = np.empty((0, 256, 256, 3))
        for i in range(len(x)):
            X = np.append(X, np.array([x[i]]), axis=0)
            logger.info(f"X.shape: {X.shape}")
        return X

    def load_all_files(self, breed):
        im_list = []
        for file in os.listdir(str(DATA_DIR) + "\\" + breed):
            im = Image.open(str(DATA_DIR) + "\\" + breed + "\\" + file)
            im_list.append(im)
        logger.info(f"Number of images in directory: {len(im_list)}")
        return im_list

    def standardize_images(self, im_list,breed):
        x, y = [], []
        for el in im_list:
            y.append(str(breed))

            width, height = el.size
            if width == height:
                el = el.resize((256, 256), Image.ANTIALIAS)
            else:
                if width > height:
                    left = width / 2 - height / 2
                    right = width / 2 + height / 2
                    top = 0
                    bottom = height
                    el = el.crop((left, top, right, bottom))
                    el = el.resize((256, 256), Image.ANTIALIAS)
                else:
                    left = 0
                    right = width
                    top = 0
                    bottom = width
                    el = el.crop((left, top, right, bottom))
                    el = el.resize((256, 256), Image.ANTIALIAS)
            el_arr = np.asarray(el)
            el.close()
            x.append(el_arr)
        return x,y

    def from_file_to_list(self, breed):
        im_list = self.load_all_files(breed)
        x, y = self.standardize_images(im_list, breed)
        return x, y
