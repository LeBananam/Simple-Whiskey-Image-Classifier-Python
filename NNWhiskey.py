import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D
import pickle
from tensorflow.keras.regularizers import l1
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
import ImageProcessing
import ImageArrayConverter
from ImageProcessing import Relearn1
from ImageArrayConverter import Relearn2
import os

# Transfer weights from old to new model
def change_model(model, res):
    # replace input shape of first layer
    model._layers[0].batch_input_shape = (None, res, res, 3)
    input_layer = Conv2D(32, (3, 3), input_shape=(None, res, res, 3))
    model._layers.pop(0)
    model.summary()
    model._layers[0] = input_layer
    #model = delete_layer(model, model.layers[0])
    #model = insert_layer(model, input_layer)
    model._layers.pop()
    model = Model(model.input, model.layers[-6].output)

    # rebuild model architecture by exporting and importing
    model.save("Data/CNNrebuild.h5")
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "Data/CNNrebuild.h5")
    new_model = tf.keras.models.load_model(filename)

    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

# Autoencoder Test
def Autoencoder():
    # Load pickle model
    a = pickle.load(open("Data/x.pickle", "rb"))
    b = pickle.load(open("Data/y.pickle", "rb"))

    # normalizing data
    a = np.asarray(a) / 255.0
    b = np.asarray(b)

    input_img = Input(shape=a.shape[1:])  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    encoded = MaxPooling2D((2, 2))(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(64, (3, 3), activation='relu')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Dense(5, activation='sigmoid')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy")

    print(a.shape)
    print(b.shape)

    model.fit(a, b, epochs=50, batch_size=50, shuffle=True, validation_split=0.05)

    # Saving the model as .h5 and .model
    model.save("Data/modelweightauto.h5")
    model.save("Data/CNNauto.model")

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
    # 3 convolutional layersy
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

    # 1 hidden layers
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
    history = model.fit(x, y, batch_size=50, epochs=50, validation_split=0.05)

    # Saving the model as .h5 and .model
    model.save("Data/modelweight.h5")
    model.save("Data/CNN.model")

# CNN Progressive Resizing Test
def Relearn4(res):

    Relearn1(res)
    Relearn2(res)

    # Load pickle model
    x = pickle.load(open("Data/x.pickle", "rb"))
    y = pickle.load(open("Data/y.pickle", "rb"))

    # normalizing data
    x = np.asarray(x) / 255.0
    y = np.asarray(y)

    model = Sequential()

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "Data/CNN.model")
    prior = tf.keras.models.load_model(filename)

    prior = change_model(prior, res)

    model.add(prior)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(5))
    model.add(Activation("softmax"))

    model.layers[0].trainable = False

    model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x, y, batch_size=50, epochs=1, validation_split=0.05)

    model.save("Data/CNN2.model")


if __name__ == "__main__":
    userinres = input("Resolution: \n")
    try:
        userinres = int(userinres)
    except ValueError:
        pass
    while not isinstance(userinres, int):
        userinres = input("Resolution: \n")
        try:
            userinres = int(userinres)
        except ValueError:
            pass
    Relearn3()
    #Relearn4(userinres)
