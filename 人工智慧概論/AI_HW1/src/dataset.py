import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    load face & non-face images through the followings steps
    1. create an empty tuple 
    2. store all the name of the pictures in folder 'face' into the variable 'images'
    3. for all the images, read the image through cv2.imread(), put it into a np array, and append it into dataset with label 1 
    4. do the same thing for non-face as it was done in 2&3 
    """
    dataset = tuple()                                                            # step 1
    images = os.listdir(dataPath+'/face')                                        # step 2
    for image in images:                                                         # step 3
      image_read = cv2.imread(dataPath+'/face/' + image, cv2.IMREAD_GRAYSCALE)   # read through imread() with the name of the file
      image_array = np.array(image_read)                                         # store the image into np array
      dataset = dataset + ((image_array,1),)                                     # add label 1 and put it into the dataset

    images = os.listdir(dataPath+'/non-face')                                    # step 4
    for image in images:
      image_read = cv2.imread(dataPath+'/non-face/' + image, cv2.IMREAD_GRAYSCALE)
      image_array = np.array(image_read)
      dataset = dataset + ((image_array,0),)                                     # with label 0
    # End your code (Part 1)
    return dataset
