import PIL
from PIL import Image, ImageOps
import os
import numpy as np
import cv2

dirname = os.path.dirname(__file__)
Data_directory = os.path.join(dirname, "photos/")

augment = True

# Pad and Resize function
def padnresize(im, desired_image_size):
    old_size = im.size

    ratio = float(desired_image_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_image_size, desired_image_size))
    new_im.paste(im, ((desired_image_size - new_size[0]) // 2,
                      (desired_image_size - new_size[1]) // 2))

    return new_im

def Relearn1(res):
    desired_image_size = res

    categories = ["Hakushu12", "Hibiki", "Yamazaki12", "Toki", "NikkaFTB"]

    # For each category folder
    for categor in categories:
        path = os.path.join(Data_directory, categor)
        # For each image in folder
        for file in os.listdir(path):
            if not file.startswith('.'):
                path2 = os.path.join(dirname, "processed/Resized/")

                path2 = path2 + categor + "/"

                if not file.startswith('.'):
                    print(path + file)
                    im = Image.open(path + "/" + file).convert("RGB")

                    new_im = padnresize(im, res)

                    new_im.save(path2 + file, "JPEG")

                    # If augment is true, create a flip and a rotate
                    if augment == True:
                        file = file.split(".")
                        file = file[0]
                        print(file)

                        file = file + "flip.JPEG"
                        new_im = ImageOps.flip(im)
                        new_im = padnresize(new_im, res)
                        new_im.save(path2 + file, "JPEG")

                        file = file + "rotateleft.JPEG"
                        rotationangle = 45
                        new_im = im.rotate(rotationangle, expand=False)
                        new_im = padnresize(new_im, res)
                        new_im.save(path2 + file, "JPEG")

                        file = file + "rotateright.JPEG"
                        rotationangle = -45
                        new_im = im.rotate(rotationangle, expand=False)
                        new_im = padnresize(new_im, res)
                        new_im.save(path2 + file, "JPEG")

if __name__ == "__main__":
    while True:
        userin = input("Name of Folder: \n")

        if userin == "q":
            break

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

        # Path of input folder and output

        path1 = os.path.join(dirname, "photos/")
        path2 = os.path.join(dirname, "processed/Resized/")

        path1 = path1 + userin + "/"
        path2 = path2 + userin + "/"

        # If can't find input path, break
        if not os.path.exists(path1):
            break

        # if can't find output path, create path
        if not os.path.exists(path2):
            os.makedirs(path2)

        # Loop through each image in input path to process each image and output to output path
        listing = os.listdir(path1)
        for file in listing:
            if not file.startswith('.'):
                print(path1 + file)
                im = Image.open(path1 + file).convert("RGB")

                new_im = padnresize(im, userinres)

                new_im.save(path2 + file, "JPEG")

                #If augment is true, create a flip and a rotate
                if augment == True:

                    file = file.split(".")
                    file = file[0]
                    print(file)

                    file = file + "flip.JPEG"
                    new_im = ImageOps.flip(im)
                    new_im = padnresize(new_im, userinres)
                    new_im.save(path2 + file, "JPEG")


                    file = file + "rotateleft.JPEG"
                    rotationangle = 45
                    new_im = im.rotate(rotationangle, expand=False)
                    new_im = padnresize(new_im, userinres)
                    new_im.save(path2 + file, "JPEG")

                    file = file + "rotateright.JPEG"
                    rotationangle = -45
                    new_im = im.rotate(rotationangle, expand=False)
                    new_im = padnresize(new_im, userinres)
                    new_im.save(path2 + file, "JPEG")
