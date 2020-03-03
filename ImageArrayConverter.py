import PIL
from PIL import Image
import os
import numpy as np
from numpy import asarray
import sys
import csv

new_list = []
results_list = []
namestonumber = {"Hakushu12": 0, "Hibiki": 1, "Yamazaki12": 2}

while(True):
    userin = input("Name of Folder to Put in Array: ")
    if userin == "q":
        new_array = np.array(new_list)
        new_array = new_array / 255.0
        new_array = new_array.flatten()
        np.savetxt(path2 + "array.csv", new_array)

        results_array = np.array(results_list)
        np.savetxt(path2 + "results_array.csv", results_array)

        print("Done")
        break

    path1 = "/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/Resized/Resized/"
    path2 = "/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/Resized/Array/"

    path1 = path1 + userin + "/"

    listing = os.listdir(path1)
    for file in listing:
        if not file.startswith('.'):
            print(path1 + file)
            im = Image.open(path1 + file)

            data = np.array(im)

            new_list.append(data)
            results_list.append(namestonumber[userin])

    print(new_list)

