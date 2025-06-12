import cv2
import numpy as np
from numpy.fft import fft2, ifft2

def Laplacian_filter_saptial(image, filter):
    image = image.astype(np.float64)
    pad_size = 1
    new_img = np.zeros(image.shape)
    # same padding
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')

    for i in range(pad_size, image.shape[0] - pad_size):
        for j in range(pad_size, image.shape[1] - pad_size):
            new_img[i, j] = np.clip(np.sum(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1] * filter, axis=(0, 1)), 0, 255)
    return new_img.astype(np.uint8)


def Laplacian_filter_frequency(image, filter):
    image = image.astype(np.float64)
    
    # pad to avoid weird edge values
    padded_image = np.pad(image, [(1, 1), (1, 1)], mode='constant')
    
    image_fft = fft2(padded_image)
    filter_fft = fft2(filter, s=padded_image.shape)
    new_img_fft = image_fft * filter_fft
    new_img = ifft2(new_img_fft)
    
    # deal with complex number
    new_img = np.abs(new_img)
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)

    
    


if __name__ == '__main__':
    # Load the image
    image = cv2.imread('img_src/IU_new.jpg', cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    
    laplacian_filter1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]).astype(np.float64)
    laplacian_filter2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).astype(np.float64)
    
    image_spatial = Laplacian_filter_saptial(image, laplacian_filter1)
    cv2.imwrite('result/IU_spatial_1.jpg', image_spatial)
    
    image_spatial = Laplacian_filter_saptial(image, laplacian_filter2)
    cv2.imwrite('result/IU_spatial_2.jpg', image_spatial)
    
    image_frequency = Laplacian_filter_frequency(image, laplacian_filter1)
    cv2.imwrite('result/IU_frequency_1.jpg', image_frequency)
    
    image_frequency = Laplacian_filter_frequency(image, laplacian_filter2)
    cv2.imwrite('result/IU_frequency_2.jpg', image_frequency)