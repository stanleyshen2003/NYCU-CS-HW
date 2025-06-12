import numpy as np


def get_histogram_function(image):
    count = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            count[image[i][j]] += 1
    sum = np.zeros(256).astype(np.float32)
    for i in range(256):
        sum[i] = sum[i-1] + count[i]
    # new_value = (L-1)/MN * sum
    sum = 255*sum/image.shape[0]/image.shape[1]
    sum = np.round(sum).astype(np.uint8)
    return sum

def histogram_equalization(image):
    func = get_histogram_function(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = func[image[i][j]]
    return image

def get_inverse_mapping(func):
    inverse = np.zeros(256).astype(np.uint8)
    for i in range(255, -1, -1):
        inverse[func[i]] = i
    for i in range(255, 0, -1):
        if inverse[i] == 0:
            inverse[i] = inverse[i+1]
    return inverse
    
def histogram_specification(image, reference):
    func1 = get_histogram_function(image)
    func2 = get_histogram_function(reference)
    func2 = get_inverse_mapping(func2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = func2[func1[image[i][j]]]
    return image

