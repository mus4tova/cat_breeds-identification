import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from loguru import logger

from settings import MODEL_DIR, EPOCHS, BREEDS_LIST


class ModelBuilder:
    def __init__(
        self,
        train_ds: pd.DataFrame,
    ):
        self.train_ds = train_ds

    def train(self) -> tf.keras.Model:
        """Train and save model"""
        model = self.model_architecture()

        # Проверим доступность GPU
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            print("GPU Available: ", physical_devices)
        else:
            print("GPU not available, using CPU instead.")

        # Установим устройство для вычислений
        device_cpu = "/CPU:0"
        device_gpu = "/GPU:0" if physical_devices else "/CPU:0"

        with tf.device(device_gpu):
            model.fit(self.train_ds, epochs=EPOCHS, validation_split=0.2)
        self.save_model(model)
        y_pred_prob = model.predict(self.train_ds)
        return model

    def model_architecture(self) -> tf.keras.Model:
        """Define models architecture"""
        model = Sequential()
        model.add(Conv2D(28, (3, 3), activation="relu", input_shape=(256, 256, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(56, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(112, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(224, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(448, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(896, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(30, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(BREEDS_LIST), activation="softmax"))

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=0.0001),
            metrics=["Accuracy", "Precision", "Recall"],
        )
        logger.info(f"model:{model.summary()}")
        return model

    def save_model(self, model: tf.keras.Model):
        model.save(f"{MODEL_DIR}/breeds_idntf.keras")
