import os
from tkinter.constants import NO

import keras
import matplotlib.pyplot as plt
import numpy
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils


class model_MNIST:

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self) -> None:
        self.model = None
        self.model_name = 'model_MNIST'
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self, seed=7):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        keras.backend.set_image_data_format('channels_first')

        numpy.random.seed(seed)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Reshape to be samples*pixels*width*height
        X_train = X_train.reshape(
            X_train.shape[0], 1, 28, 28).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

        # Normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255

        # One Hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]

        # Create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(
            1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=10, batch_size=200)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def test(self):
        image = (self.X_test[1]).reshape(1, 1, 28, 28)  # 1->'2';
        model_pred = self.model.predict_classes(image, verbose=0)
        print('Prediction of model: {}'.format(model_pred[0]))

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()

        with open(os.path.join(model_MNIST.BASE_DIR, 'SudokuSolve', 'models', f'{self.model_name}.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join(
            model_MNIST.BASE_DIR, 'SudokuSolve', 'models', f'{self.model_name}.h5'))
        print("Saved model to disk")


def main():

    model = model_MNIST()

    model.train()
    model.test()
    model.save_model()


if __name__ == "__main__":
    main()
