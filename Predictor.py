import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
import PIL
from PIL import Image
from ImageProcessing import Relearn1
from ImageArrayConverter import Relearn2
from NNWhiskey import Relearn3, Relearn4

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "Data/CNN.model")

categories = ["Hakushu12", "Hibiki", "Yamazaki12", "Toki", "NikkaFTB"]

# Creating method to process input image
def preparePrediction(file):
    imgsize = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (imgsize, imgsize))
    return new_array.reshape(-1, imgsize, imgsize, 1)

# Prompt to see if user want to retrain model based on previous results
relearnqry = input("Retrain model? Y/N \n")
relearnqry = relearnqry.upper()
while relearnqry != "Y" and relearnqry != "N":
    relearnqry = input("Retrain model? Y/N \n")
    relearnqry = relearnqry.upper()
if relearnqry == "Y":
    """
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
    Relearn1(int(userinres/2))
    Relearn2(int(userinres/2))
    Relearn3()
    Relearn4(userinres)
    """
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
    Relearn1(userinres)
    Relearn2(userinres)
    Relearn3()

model = tf.keras.models.load_model(filename)

if __name__ == "__main__":
    while True:
        # Ask for user input of test path
        userinput = input("Enter image path: \n")
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
        result = categories[prediction.index(max(prediction))]
        print(result)

        # Handle input
        useryn = input("Correct? Y/N \n")
        useryn = useryn.upper()
        while useryn != "Y" and useryn != "N":
            useryn = input("Correct? Y/N \n")
            useryn = useryn.upper()
        if useryn == "N":
            userresponse = input("Answer is? \n")
            destination = os.path.join(dirname, "photos/", userresponse)
            while not os.path.exists(destination):
                print("No category found! Please input an existing category \n")
                userresponse = input("Answer is? \n")
                destination = os.path.join(dirname, "photos/", userresponse)
            if userresponse == result:
                print("The result was correct. No images are added to relearn. \n")
            else:
                im = Image.open(userinput).convert("RGB")
                userinput = userinput.split("/")
                userinput = userinput[-1]
                userinput = userinput.split(".")
                userinput = userinput[0]
                userresponse = "Data/" + userresponse + "/"
                file_id = 0
                while os.path.exists(destination + "/" + userinput + "_" + str(file_id) + "JPEG"):
                    file_id += 1
                im.save(destination + "/" + userinput + str(file_id) + ".JPEG", "JPEG")
