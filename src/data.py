import os
import numpy as np
from PIL import Image
from loguru import logger
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

from settings import BREEDS_LIST, DATA_DIR


class DataLoader:
    def __init__(self):
        self.breeds_list = BREEDS_LIST

    def get_data(self) -> tuple[NumpyArrayIterator, NumpyArrayIterator, np.ndarray]:
        """Main method in file. Collect all images of all breeds and consolidate them in 4D numpy array"""
        x, y = [], []
        logger.info(f"Breeds list: {self.breeds_list}")
        for breed in self.breeds_list:
            logger.info(f"Breed: {breed}")
            x, y = self.add_breed(breed, x, y)
        x, y = np.array(x), np.array(y)
        y = self.label_transform(y)
        X_train, X_test, y_train, y_test = self.train_test_split(x, y)
        train = self.image_data_generation(X_train, y_train)
        test = self.image_data_generation(X_test, y_test)
        return train, test, y_test

    def add_breed(
        self, breed: str, x: list, y: list
    ) -> tuple[list[np.ndarray], list[str]]:
        """Collect all images in breeds directory, convert them to arrays and add to general list"""
        im_list = self.load_breed_files(breed)
        for image in im_list:
            y.append(str(breed))
            el_arr = self.convert_to_array(image)
            x.append(el_arr)
        return x, y

    def load_breed_files(self, breed: str) -> list[Image]:
        """Read all images in breed directory and add them to a list"""
        im_list = []
        for file in os.listdir(str(DATA_DIR) + "\\" + breed):
            im = Image.open(str(DATA_DIR) + "\\" + breed + "\\" + file)
            im_list.append(im)
        logger.info(f"Number of images in directory: {len(im_list)}")
        return im_list

    def convert_to_array(self, image: Image) -> np.ndarray:
        """Standardizes and convert element from picture to array"""
        width, height = image.size
        if width == height:
            image = image.resize((256, 256), Image.ANTIALIAS)
        else:
            if width > height:
                left = width / 2 - height / 2
                right = width / 2 + height / 2
                top = 0
                bottom = height
                image = image.crop((left, top, right, bottom))
                image = image.resize((256, 256), Image.ANTIALIAS)
            else:
                left = 0
                right = width
                top = 0
                bottom = width
                image = image.crop((left, top, right, bottom))
                image = image.resize((256, 256), Image.ANTIALIAS)
        arr_el = np.asarray(image)  # convert to array
        return arr_el

    def label_transform(self, y: np.ndarray):
        """Encode element from string to integer"""
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        return y

    def image_data_generation(self, X: np.ndarray, y: np.ndarray) -> NumpyArrayIterator:
        """Convert from array to ImageDataGenerator"""
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        data = datagen.flow(X, y, batch_size=1)
        return data

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train and test samples"""
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
