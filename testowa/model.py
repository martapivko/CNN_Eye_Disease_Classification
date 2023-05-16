import tensorflow as tf
from keras import layers
from keras.models import Sequential

class Net():
    def __init__(self) -> None:
        self.nOfClasses = 2
        self.model = Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(96, 96, 1)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPool2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPool2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPool2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(64, activation='sigmoid'),
                layers.Dense(self.nOfClasses, activation='sigmoid')
        ])
