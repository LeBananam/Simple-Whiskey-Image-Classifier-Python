import PIL
from PIL import Image
import os

while True:
    userin = input("Name of Folder: ")

    if userin == "q":
        break

    # Path of input folder and output

    path1 = "/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/photos/"
    path2 = "/Users/bananam/PycharmProjects/tensorENV/DOTA/Simple-Whiskey-Image-Classifier-Python/Resized/"

    path1 = path1 + userin + "/"
    path2 = path2 + userin + "/"

    # If can't find input path, break
    if not os.path.exists(path1):
        break

    # if can't find output path, create path
    if not os.path.exists(path2):
        os.makedirs(path2)

    # Loop through each image in input path to process each image and output to output path
    desired_image_size = 50
    listing = os.listdir(path1)
    for file in listing:
        if not file.startswith('.'):
            print(path1 + file)
            im = Image.open(path1 + file)
            old_size = im.size

            ratio = float(desired_image_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            im = im.resize(new_size, Image.ANTIALIAS)

            new_im = Image.new("RGB", (desired_image_size, desired_image_size))
            new_im.paste(im, ((desired_image_size - new_size[0]) // 2,
                              (desired_image_size - new_size[1]) // 2))

            new_im.save(path2 + file, "JPEG")

