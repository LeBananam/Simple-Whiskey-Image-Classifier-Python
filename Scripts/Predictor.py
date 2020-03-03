import cv2
import tensorflow as tf
import numpy as np

categories = ["Hakushu12", "Hibiki", "Yamazaki12"]

# Creating method to process input image
def preparePrediction(file):
    imgsize = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (imgsize, imgsize))
    return new_array.reshape(-1, imgsize, imgsize, 1)

model = tf.keras.models.load_model("/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/Data/CNN.model")


# Ask for user input of test path
userinput = input("Enter image path: ")
image = userinput

# Process input image adn normalize
image = preparePrediction(image)
image = np.asarray(image) / 255.0

# Make predictions and print
prediction = model.predict([image])
prediction = list(prediction[0])
print(categories[prediction.index(max(prediction))])
