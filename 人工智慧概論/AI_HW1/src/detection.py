import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
import classifier

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    '''
    read the file, find the corresponding image of face & other data as described in the .txt.
    read the images, choose the portion needed, resize it, and convert into gray-scale.
    use the classifier to predict whether the image is a face.
    '''
    file= open(dataPath, 'r')                                                                 # open .txt
    contents = file.read()                                                                    
    elements = contents.split()
    i = 0
    while(i < len(elements)):
        imgName = elements[i]
        img = cv2.imread('data/detect/'+imgName)                                              # read image
        i += 1 
        numberOfSquares = int(elements[i])
        i += 1
        for j in range(numberOfSquares):
            x1 , x2 = int(elements[i+4*j]), int(elements[i+4*j]) + int(elements[i+4*j+2])     # area needed
            y1 , y2 = int(elements[i+4*j+1]), int(elements[i+4*j+1]) + int(elements[i+4*j+3])
            cutImg = img[y1:y2,x1:x2]                                                         # cut the area
            imgfordetect = cv2.resize(cutImg, (19, 19), interpolation=cv2.INTER_NEAREST)      # resize it
            imgfordetect = cv2.cvtColor(imgfordetect, cv2.COLOR_BGR2GRAY)                  
            image_array = np.array(imgfordetect)
            isFace = clf.classify(image_array)                                                # use classifier to predict
            if(isFace):                                                                       # draw rectangle
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)                
            else:
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
        i += numberOfSquares*4
        cv2.imwrite('data/img after detect/'+imgName[:-4]+'.jpg', img)                                                 # save the drawn img
    # End your code (Part 4)
