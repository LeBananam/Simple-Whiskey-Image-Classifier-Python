import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.regularizers import l1
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import keras

# Relearning
def Relearn3():
    # Load pickle model
    x = pickle.load(open("Data/x.pickle", "rb"))
    y = pickle.load(open("Data/y.pickle", "rb"))

    # normalizing data
    x = np.asarray(x) / 255.0
    y = np.asarray(y)

    # Building the model
    model = Sequential()
    # 3 convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2 hidden layers
    model.add(Flatten())
    model.add(Dense(32, activity_regularizer=l1(0.001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    #model.add(Dense(128))
    #model.add(Activation("relu"))

    # The output layer with 5 neurons, for 5 classes
    model.add(Dense(5))
    model.add(Activation("softmax"))

    # Compiling the model using some basic parameters
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # validation_split corresponds to the percentage of images used for the validation phase compared to all the images
    history = model.fit(x, y, batch_size=50, epochs=55, validation_split=0.05)

    # Saving the model as JSON and .model
    model_json = model.to_json()
    with open("Data/modeljson.json", "w") as file:
        file.write(model_json)
    model.save_weights("Data/modelweight.h5")
    model.save("Data/CNN.model")

if __name__ == "__main__":
    Relearn3()