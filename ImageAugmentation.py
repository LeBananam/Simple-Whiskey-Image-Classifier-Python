import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import PIL
from PIL import Image
import sklearn

dirname = os.path.dirname(__file__)

while True:
    userin = input("Name of Folder: ")

    if userin == "q":
        break

    # Path of input folder and output
    path1 = os.path
    path = os.path.join(dirname, "processed/Resized/")

    path = path + userin + "/"

    # If can't find input path, break
    if not os.path.exists(path):
        break

    # Loop through each image in input path to process each image and output to output path

    listing = os.listdir(path)
    for file in listing:
        if not file.startswith('.'):
            print(path + file)
            im = cv2.imread(os.path.join(path, file))

            new_im = np.fliplr(im)

            cv2.imwrite(path + file + "flip.JPEG", new_im)

    rotationangle = 45
    for file in listing:
        if not file.startswith('.'):
            print(path + file)
            im = Image.open(path + file)

            new_im = im.rotate(rotationangle)

            new_im.save(path + file + "rotate", "JPEG")