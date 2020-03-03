import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Loading Data
data = keras.datasets.fashion_mnist

#Creating trsin and test data for keras
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Scale data down for easy manipulation and optimize speed
train_images = train_images/255.0
test_images = test_images/255.0









#Create layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #Flatten input layer to simplify data and create input nodes
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#Compile model with these functions and metrics
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train model with epoches defined. Epoches are how many number of times you seeing the same image in a different order to increase accuracy of model hopefully
model.fit(train_images, train_labels, epochs=1)

#Create predictions and use the model by passing in an array
predictions = model.predict(test_images)

#test if the predict works and give prediction
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predictions " + class_names[np.argmax(predictions[i])])
    plt.show()













#Create accuracy evaluations using test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test acc: ", test_acc)

