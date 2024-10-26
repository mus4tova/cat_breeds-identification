import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from settings import MODEL_DIR, EPOCHS


class ModelBuilder:
    def __init__(
        self,
        train_ds: pd.DataFrame,
        test_ds: pd.DataFrame,
    ):
        self.train_ds = train_ds
        self.test_ds = test_ds

    def train(self) -> tf.keras.Model:
        model = self.model_architecture()
        history = model.fit(self.train_ds, epochs=EPOCHS) #validation_split=0.2
        #self.save_model(model)
        return model

    def model_architecture(self) -> tf.keras.Model:
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
        # model.add(Conv2D(1792, (3,3), activation='relu'))
        # model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(30, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate=0.0001),
            metrics=["accuracy"],
        )
        model.summary()
        return model

    def save_model(self, model: tf.keras.Model):
        model.save(f"{MODEL_DIR}/1st_version")
