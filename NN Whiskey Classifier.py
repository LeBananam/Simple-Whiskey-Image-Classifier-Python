import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

image = cv2.imread("/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/Resized/Yam12/images1.jpg")

data = asarray(image)

data = data/255.0

print(data)



