import cv2
import tensorflow as tf
import numpy as np
import os
import shutil

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "Data/CNN.model")

categories = ["Hakushu12", "Hibiki", "Yamazaki12", "Toki", "NikkaFTB"]

# Creating method to process input image
def preparePrediction(file):
    imgsize = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (imgsize, imgsize))
    return new_array.reshape(-1, imgsize, imgsize, 1)

model = tf.keras.models.load_model(filename)


while True:
    # Ask for user input of test path
    userinput = input("Enter image path: ")
    userinput = userinput.strip()
    image = userinput

    if userinput=="q":
        break


    # Process input image adn normalize
    image = preparePrediction(image)
    image = np.asarray(image) / 255.0

    # Make predictions and print
    prediction = model.predict([image])
    prediction = list(prediction[0])
    print(categories[prediction.index(max(prediction))])

    print("Correct? Y/N")
    useryn = input()
    useryn = useryn.upper()
    if useryn == "N":
        print("Answer is?")
        userresponse = input()

        destination = os.path.join(dirname, "photos/", userresponse)
        if not os.path.exists(destination):
            print("No category found!")
            break

        shutil.copy(userinput, destination)
