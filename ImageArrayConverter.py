import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

dirname = os.path.dirname(__file__)
Data_directory = os.path.join(dirname, "processed/Resized/")

# All the categories the NN can detect
categories = ["Hakushu12", "Hibiki", "Yamazaki12", "Toki", "NikkaFTB"]

# Define data creation method
training_data = []
def create_data():
    for categor in categories :
        path = os.path.join(Data_directory, categor)
        class_num = categories.index(categor)
        for img in os.listdir(path):
            if not img.startswith('.'):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = img_array
                training_data.append([img_array, class_num])

# Creating data
create_data()
random.shuffle(training_data)

# Dividing data into labels and training data
x = []
y = []
for trainingdat, labels in training_data:
    x.append(trainingdat)
    y.append(labels)

x = np.array(x).reshape(-1, 50, 50, 1)

# Creating pickle for training data
pickle_out = open("Data/x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

# Creating pickle for labels
pickle_out = open("Data/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
