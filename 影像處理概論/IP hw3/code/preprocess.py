import cv2
import numpy as np


def box_filter(image, filter_size=5):
    '''
    apply box filter to blur the image
    '''
    new_img = np.zeros(image.shape)
    pad_size = filter_size // 2
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            new_img[i, j] = np.mean(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1], axis=(0, 1))
    return new_img.astype(np.uint8)

if __name__ == '__main__':
    # Load the image
    image = cv2.imread('img_src/IU.jpg', cv2.IMREAD_GRAYSCALE)
    # convert to np array
    image = np.array(image)
    image = box_filter(image, 5)
    image = box_filter(image, 5)
    image = box_filter(image, 5)
    cv2.imwrite('img_srd/IU_new.jpg', image)