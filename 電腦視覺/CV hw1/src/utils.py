import numpy as np
#from HW1 import read_object
from queue import Queue
import math
import sys

def preprocess_gaussian_filter(image):
    new_image = np.zeros(image.shape)    
    padded = np.pad(image, 1, mode='edge')
    padded = padded.astype(np.float64)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            new_image[i-1, j-1] = np.sum(padded[i-1:i+2, j-1:j+2] * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])) / 16
    return new_image



def preprocess_low_pass(image):
    new_image = np.zeros(image.shape)
    padded = np.pad(image, 1, mode='edge')
    padded = padded.astype(np.float32)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            new_image[i-1, j-1] = np.sum(padded[i-1:i+2, j-1:j+2]) / 9
    return new_image

def normal_low_pass(image):
    new_img = np.zeros(image.shape)
    # only pad x and y axis, not the channel axis
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    padded = padded.astype(np.float32)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            if np.isnan(padded[i, j][0]):
                new_img[i-1, j-1] = np.nan
                continue
            new_img[i-1, j-1] = np.nansum(padded[i-1:i+2, j-1:j+2], axis=(0, 1)) / np.sum(~np.isnan(padded[i-1:i+2, j-1:j+2]))
    return new_img

def preprocess_low_pass2(image):
    new_image = np.zeros(image.shape)
    padded = np.pad(image, 2, mode='edge')
    padded = padded.astype(np.float32)
    for i in range(2, padded.shape[0]-2):
        for j in range(2, padded.shape[1]-2):
            new_image[i-2, j-2] = np.sum(padded[i-2:i+3, j-2:j+3]) / 25

def mask_image_pixel(images):
    mask = np.zeros(images[0].shape)
    for image in images:
        mask += image
    mask = np.where(mask < len(images)*15, 0, 1)
    for image in images:
        image *= mask
    return images

def mask_normal(image):
    mask = np.zeros(image.shape[:2])
    image = image.astype(np.float32)
    q = Queue()
    q.put((int(image.shape[0]/2), int(image.shape[1]/2)))
    while not q.empty():
        x, y = q.get()
        if mask[x, y] == 1:
            continue
        mask[x, y] = 1
        if x > 0 and not np.isnan(image[x-1, y][0]):
            q.put((x-1, y))
        if x < image.shape[0]-1 and not np.isnan(image[x+1, y][0]):
            q.put((x+1, y))
        if y > 0 and not np.isnan(image[x, y-1][0]):
            q.put((x, y-1))
        if y < image.shape[1]-1 and not np.isnan(image[x, y+1][0]):
            q.put((x, y+1))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 0:
                image[i, j] = math.nan
    return image

     
   
        