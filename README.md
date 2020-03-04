# Simple-Whiskey-Image-Classifier-Python
A simple image classifier made in Python using Tensorflow and Keras to identify and distinguish between 5 Japanese Whiskey bottles (Yamazaki12, Hakushu12, Hibiki, Toki, NikkaFTB) with CNN architecture.

The plugins used for this project are:
- CV2
- PIL
- Tensorflow
- Teras
- Numpy
- Pickle
- Random

Step 1: Collecting images to create dataset
- I did this by downloading 100 images of each whiskey from Google using Bulk Image Downloader extension on Google Chrome which is available here: https://chrome.google.com/webstore/detail/bulk-image-downloader/lamfengpphafgjdgacmmnpakdphmjlji?hl=en

Step 2: Preprocess images to standardize data
- In order to efficiently train a model, we need images of the same sizes. I rescaled the images down to 50x50 and pad the images to keep its original aspect ratio. This was done by my ImageProcessing.py script.

Step 3: Convert images to grayscale, convert them into arrays of pixels and ready them for model training
- Using the ImageArrayConverter.py, I loop through each image of each whiskey and converting them into grayscale images using cv2 plugin.
- Afterwards, for each image, I also convert it into an array of pixels (Let's say array A) using numpy before putting this array into another array (B). This B array will contain 2 members, the A array and the type of whiskey of this image.
- B array will then be randomized.
- This B array will be appended into a list which will be split into a feature list(x) and a label list(y).
- X list will be reshaped to be 4-dimensional to be compatible with conv2d layer during training.
- X and y will be pickled out to be imported later for training.

Step 4: Creation of CNN, model and training NN_Whiskey_Classifier.py
- X and y pickles are loaded in and converted from list to arrays.
- X is normalized by diving by 255.0
- CNN is created with 3 convolutional layers, 2 hidden layers and 1 output layer with default settings (loss, activation, etc)
- Model is then compiled and fit.
- Model after training will be saved.

Step 5: 
- Images can now be predicted. User just need to run Predictor.py and put the path of the query image in to get prediction result.
