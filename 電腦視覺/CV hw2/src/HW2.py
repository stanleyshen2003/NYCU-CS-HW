import cv2
import numpy as np
import random
import math
import sys
import os
from skimage.exposure import match_histograms
from scipy.ndimage import median_filter

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def create_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_matches(keypoints1, descriptor1, keypoints2, descriptor2, lowes_ratio=0.8, debugging=False):
    good_matches = []
    points1 = []
    points2 = []
    
    if debugging:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)
        for m,n in matches:
            if m.distance < lowes_ratio*n.distance:
                good_matches.append(m)
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        
    else:
        for i in range(len(keypoints1)):
            min_dist = np.inf
            second_min_dist = np.inf
            min_dist_index = -1
            
            # compute the dinstance between two descriptors and update
            for j in range(len(keypoints2)):
                dist = np.linalg.norm(descriptor1[i] - descriptor2[j]).item()
                if dist < min_dist:
                    second_min_dist = min_dist
                    min_dist = dist
                    min_dist_index = j
                elif dist < second_min_dist:
                    second_min_dist = dist
            # lowe's ratio test   
            if min_dist  < lowes_ratio * second_min_dist:
                points1.append(keypoints1[i].pt)
                points2.append(keypoints2[min_dist_index].pt)
    return np.float32(points1), np.float32(points2)

def get_Homography(points1, points2):
    # construct A
    A = []
    for i in range(len(points1)):
        A.append([points1[i, 0], points1[i, 1], 1, 0, 0, 0, -points1[i, 0] * points2[i, 0], -points1[i, 1] * points2[i, 0], -points2[i, 0]])
    for i in range(len(points1)):
        A.append([0, 0, 0, points1[i, 0], points1[i, 1], 1, -points1[i, 0] * points2[i, 1], -points1[i, 1] * points2[i, 1], -points2[i, 1]])
    
    # solve using SVD
    _, _, vt = np.linalg.svd(A)
    
    # pick smallest number & normalization
    H = np.reshape(vt[-1], (3, 3))
    H = H / (H[2, 2]+1e-8)
    return H

def RANSAC_for_H(points1, points2, RANSAC_n_iter=1000, RANSAC_threshold=5, debugging=False):
    if debugging:
        best_H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    else:
        maximum_inliers = 0
        best_H = None
        
        for i in range(RANSAC_n_iter):
            random_index = random.sample(range(len(points1)), 4)
            random_points1 = points1[random_index]
            random_points2 = points2[random_index]
            H = get_Homography(random_points1, random_points2)
            inliers = 0
            for j in range(len(points1)):
                p1 = np.array([points1[j, 0], points1[j, 1], 1])
                p2 = np.array([points2[j, 0], points2[j, 1], 1])
                p2_hat = np.dot(H, p1)
                p2_hat = p2_hat / p2_hat[2]
                if np.linalg.norm(p2 - p2_hat) < RANSAC_threshold:
                    inliers += 1
            if inliers > maximum_inliers:
                maximum_inliers = inliers
                best_H = H
    return best_H

def blend(img1, img2, detect_threshold=6):
    height, width, _ = img1.shape
    img1_mask = np.zeros((height, width), dtype=np.int16)
    img2_mask = np.zeros((height, width), dtype=np.int16)
    
    # find locations of the non-black pixels in both images
    for i in range(height):
        for j in range(width):
            if np.sum(img1[i, j]) > detect_threshold:
                img1_mask[i, j] = 1
            if np.sum(img2[i, j]) > detect_threshold:
                img2_mask[i, j] = 1
                  
    overlap_mask = img1_mask * img2_mask
    blended_image = np.zeros_like(img1, dtype=np.float32)
    
    for i in range(height):
        # create a scalar mask for blending if needed
        if np.count_nonzero(overlap_mask[i]) > 0:
            left_most = width
            right_most = 0
            for j in range(width):
                if overlap_mask[i, j] == 1:
                    left_most = min(left_most, j)
                    right_most = max(right_most, j)
            blend_width = right_most - left_most + 1
            blend_mask = np.linspace(1, 0, blend_width)
            
        # fill in the values
        for j in range(width):
            if overlap_mask[i, j] == 1:
                # perform blending
                alpha = blend_mask[j - left_most]
                blended_image[i, j] = alpha * img1[i, j] + (1 - alpha) * img2[i, j]
            elif img1_mask[i, j] == 1:
                blended_image[i, j] = img1[i, j]
            else:
                blended_image[i, j] = img2[i, j]

    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

def patch_images(img1, img2, H):
    # get an affine transformation matrix
    left_down = np.hstack(([0], [0], [1]))
    left_up = np.hstack(([0], [img1.shape[0]-1], [1]))
    right_down = np.hstack(([img1.shape[1]-1], [0], [1]))
    right_up = np.hstack(([img1.shape[1]-1], [img1.shape[0]-1], [1]))
    
    warped_left_down = np.dot(H,left_down.T) / np.dot(H, left_down.T)[2]
    warped_left_up = np.dot(H, left_up.T) / np.dot(H, left_up.T)[2]
    warped_right_down =  np.dot(H , right_down.T) / np.dot(H, right_down.T)[2]
    warped_right_up = np.dot(H, right_up.T) / np.dot(H, right_up.T)[2]

    x1 = min(warped_left_up[0], warped_left_down[0], warped_right_down[0], warped_right_up[0], 0)
    x2 = max(warped_left_up[0], warped_left_down[0], warped_right_down[0], warped_right_up[0], img2.shape[1])
    y1 = min(warped_left_up[1], warped_left_down[1], warped_right_down[1], warped_right_up[1], 0)
    y2 = max(warped_left_up[1], warped_left_down[1], warped_right_down[1], warped_right_up[1], img2.shape[0])
    width = int(x2 - x1)
    height = int(y2 - y1)
    size = (width, height)

    A = np.float32([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]])
    warped1 = cv2.warpPerspective(src=img1, M=A@H, dsize=size, flags=cv2.INTER_NEAREST)
    warped2 = cv2.warpPerspective(src=img2, M=A, dsize=size, flags=cv2.INTER_NEAREST)
    
    return warped1, warped2

def preprocess(image, shift=10):
    flag = cv2.INTER_NEAREST
    # do affine transformation on 4 images
    # define the 4 points of the first image
    x, y = image.shape[1], image.shape[0]
    pts1 = np.float32([[0, 0], [x/2, 0], [0, y/2], [x/2, y/2]])
    dest1 = np.float32([[shift, shift], [x/2, 0], [shift, y/2], [x/2, y/2]])
    result1 = cv2.getPerspectiveTransform(pts1, dest1)
    img_temp = np.array(image)
    img1 = img_temp[0:y//2, 0:x//2]
    warped_image = cv2.warpPerspective(img1, result1, (x//2, y//2), flags=flag)

    pts2 = np.float32([[0, 0], [x-x/2, 0], [0, y/2], [x-x/2, y/2]])
    dest2 = np.float32([[0, 0], [x-x/2-shift, shift], [0, y/2], [x-x/2-shift, y/2]])
    result2 = cv2.getPerspectiveTransform(pts2, dest2)
    img2 = img_temp[0:y//2, x//2:x]
    warped_image2 = cv2.warpPerspective(img2, result2, (x - x//2, y//2), flags=flag)
    
    pts3 = np.float32([[0, 0], [x/2, 0], [0, y-y/2], [x/2, y-y/2]])
    dest3 = np.float32([[shift, 0], [x/2, 0], [shift, y-y/2-shift], [x/2, y-y/2]])
    result3 = cv2.getPerspectiveTransform(pts3, dest3)
    img3 = img_temp[y//2:y, 0:x//2]
    warped_image3 = cv2.warpPerspective(img3, result3, (x//2, y - y//2), flags=flag)
    
    pts4 = np.float32([[0, 0], [x-x/2, 0], [0, y-y/2], [x-x/2, y-y/2]])
    dest4 = np.float32([[0, 0], [x-x/2-shift, 0], [0, y-y/2], [x-x/2-shift, y-y/2-shift]])
    result4 = cv2.getPerspectiveTransform(pts4, dest4)
    img4 = img_temp[y//2:y, x//2:x]
    warped_image4 = cv2.warpPerspective(img4, result4, (x-x//2, y-y//2), flags=flag)
    
    # combine the 4 images
    if image.ndim == 2:
        result = np.zeros((y, x), np.uint8)
    else:
        result = image
    result[0:y//2, 0:x//2] = warped_image
    result[0:y//2, x//2:x] = warped_image2
    result[y//2:y, 0:x//2] = warped_image3
    result[y//2:y, x//2:x] = warped_image4
    return result

def preprocess2(img, shift=10):
    h_,w_ = img.shape[:2]
    K = np.array([[w_,0,w_/2],[0,h_,h_/2],[0,0,1]])
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3)
    Kinv = np.linalg.inv(K) 
    # normalized coords
    X = Kinv.dot(X.T).T
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    # project back to image-pixels plane
    B = K.dot(A.T).T 
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    return cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_NEAREST).clip(0, 255).astype(np.uint8)

def base_process(img_path1, img_path2, preprocessing=True):
    # settings, this is tunable!
    preprocess_shift = 50
    RANSAC_n_iter = 1000
    RANSAC_threshold = 5
    detect_threshold = 6
    
    # read and preprocess the images
    color_img1, gray_img1 = read_img(img_path1)
    color_img2, gray_img2 = read_img(img_path2)
    if preprocessing:
        color_img1 = preprocess(color_img1, shift=preprocess_shift)
        color_img2 = preprocess(color_img2, shift=preprocess_shift)
        gray_img1 = preprocess(gray_img1, shift=preprocess_shift)
        gray_img2 = preprocess(gray_img2, shift=preprocess_shift)
    
    print(color_img1.shape)
    # get the keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)
    
    # 2NN matching with Lowe's ratio test
    points1, points2 = get_matches(keypoints1, descriptors1, keypoints2, descriptors2, debugging=False)
    
    # Compute the homography matrix using RANSAC
    H = RANSAC_for_H(points1, points2, RANSAC_n_iter=RANSAC_n_iter, RANSAC_threshold=RANSAC_threshold, debugging=False)
    
    # transform the images
    warped1, warped2 = patch_images(color_img1, color_img2, H)
    
    # blend the images
    result = blend(warped1, warped2, detect_threshold=detect_threshold)
    return result

def Base():
    root_path = 'Photos/'
    result_path = 'Results/'
    mode = 'Base'
    result_path = os.path.join(result_path, mode)
    os.makedirs(result_path, exist_ok=True)
    
    img1_path = os.path.join(root_path, 'Base', 'Base1.jpg')
    img2_path = os.path.join(root_path, 'Base', 'Base2.jpg')
    result1_path = os.path.join(result_path, 'round1.jpg')
    result = base_process(img1_path, img2_path)
    cv2.imwrite(result1_path, result)
    print(f'{result1_path} is saved')
    
    img1_path = os.path.join(root_path, 'Base', 'Base2.jpg')
    img2_path = os.path.join(root_path, 'Base', 'Base3.jpg')
    result = base_process(img1_path, img2_path)
    result2_path = os.path.join(result_path, 'round2.jpg')
    cv2.imwrite(result2_path, result)
    print(f'{result2_path} is saved')
    
    result = base_process(result1_path, result2_path, preprocessing=False)
    final_result_path = os.path.join(result_path, 'result.jpg')
    cv2.imwrite(final_result_path, result)
    print(f'{final_result_path} is saved')

def calculate_pair(img1, img2):
    width, height = img1.shape[1], img1.shape[0]
    img1_mask = np.zeros((height, width), dtype=np.int16)
    img2_mask = np.zeros((height, width), dtype=np.int16)
    detect_threshold = 6
    for i in range(height):
        for j in range(width):
            if np.sum(img1[i, j]) > detect_threshold:
                img1_mask[i, j] = 1
            if np.sum(img2[i, j]) > detect_threshold:
                img2_mask[i, j] = 1
                  
    overlap_mask = img1_mask * img2_mask
    N_ij = np.count_nonzero(overlap_mask)
    if N_ij == 0:
        return None, None, 0
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    overlap_mask = overlap_mask.astype(np.float32)
    I_ij = np.sum(img1 * np.stack([overlap_mask, overlap_mask, overlap_mask], axis=2), axis=(0, 1)) / N_ij
    I_ji = np.sum(img2 * np.stack([overlap_mask, overlap_mask, overlap_mask], axis=2), axis=(0, 1)) / N_ij
    return I_ij, I_ji, N_ij

def task_gain_compensation(images, sigma_n=10, sigma_g=0.9):
    # extract overlap pairs
    n_images = len(images)
    
    coefficients = np.zeros((n_images, n_images, 3))
    results = np.zeros((n_images, 3))
    
    # fill in the matrix
    for i in range(n_images-1):
        for j in range(i+1, n_images):
            I_ij, I_ji, N_ij = calculate_pair(images[i], images[j])
            if N_ij == 0:
                continue
            # /1e6 for numerical stability
            coefficients[i][i] += N_ij * ( (2 * I_ij ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2) ) / 1e6
            coefficients[i][j] -= (2 / sigma_n ** 2) * N_ij * I_ij * I_ji / 1e6
            coefficients[j][i] -= (2 / sigma_n ** 2) * N_ij * I_ji * I_ij / 1e6
            coefficients[j][j] += N_ij * ( (2 * I_ji ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2) )  / 1e6
            results[i] += N_ij / sigma_g ** 2  / 1e6
            results[j] += N_ij / sigma_g ** 2  / 1e6
            
    gains = np.zeros_like(results)
    for channel in range(coefficients.shape[2]):
        coefs = coefficients[:, :, channel]
        res = results[:, channel]
        # solve with psuedo-inverse
        gains[:, channel] = np.linalg.pinv(coefs) @ res

    max_pixel_value = np.max([image for image in images])
    print(gains)
    # normalize
    if gains.max() * max_pixel_value > 255:
        gains = gains / (gains.max() * max_pixel_value) * 255

    return gains

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    for i in range(3):
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                image[j][k][i] = table[image[j][k][i]]
    return image

def Challange():
    preprocess_shift = 30
    RANSAC_n_iter = 1000
    RANSAC_threshold = 5
    detect_threshold = 1
    sigma_n = 10
    sigma_g = 0.9
    img_num = 6
    root_path = 'Photos/'
    result_path = 'Results/'
    mode = 'Challenge'
    flag = cv2.INTER_NEAREST
    result_path = os.path.join(result_path, mode)
    os.makedirs(result_path, exist_ok=True)
    A = np.float32([[1, 0, 617.6448], [0, 1, 22.620256], [0, 0, 1]])
    size = (2016, 1570)
    
    img_paths = [os.path.join(root_path, mode, mode+str(i+1)+'.jpg') for i in range(6)]
    
    sift = cv2.SIFT_create()
    
    warped_images = []

    last_result = cv2.imread(img_paths[3])
    last_result = np.clip(last_result, 5, 254).astype(np.uint8)
    last_result = preprocess2(last_result, preprocess_shift)
    last_result = cv2.cvtColor(last_result, cv2.COLOR_BGR2YUV)
    last_result[:, :, 0] = cv2.equalizeHist(last_result[:, :, 0])
    last_result = cv2.cvtColor(last_result, cv2.COLOR_YUV2BGR)
    last_result = cv2.warpPerspective(src=last_result, M=A, dsize=size, flags=flag)
    warped_images.append(last_result)

    for i in range(2, -1, -1):
        last_result = cv2.cvtColor(last_result, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(last_result, None)
        temp_keypoints = []
        temp_descriptors = []
        for (j, keypoint) in enumerate(keypoints):
            if last_result[int(keypoint.pt[1]), int(keypoint.pt[0])] != 0:
                temp_keypoints.append(keypoint)
                temp_descriptors.append(descriptors[j])
        keypoints = temp_keypoints
        descriptors = np.array(temp_descriptors)
        new_image = cv2.imread(img_paths[i])
        
        if i == 2:
            new_image = adjust_gamma(new_image, 0.4)
        else:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)
            new_image[:, :, 0] = cv2.equalizeHist(new_image[:, :, 0])
            new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
        new_image = np.clip(new_image, 5, 254).astype(np.uint8)
        
        if i == 0:
            new_image[:, new_image.shape[1]//2:new_image.shape[1]] = 0
        new_image = preprocess2(new_image, preprocess_shift)
        new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        new_keypoints, new_descriptors = sift.detectAndCompute(new_image_gray, None)
        
        points = get_matches(new_keypoints, new_descriptors, keypoints, descriptors, debugging=False)
        H = RANSAC_for_H(points[0], points[1], RANSAC_n_iter=RANSAC_n_iter, RANSAC_threshold=RANSAC_threshold, debugging=False)
        warped_images.insert(0, cv2.warpPerspective(src=new_image, M=H, dsize=size, flags=flag))
        last_result = last_result = blend(warped_images[0], last_result, detect_threshold=detect_threshold)
    
    last_result = warped_images[3]
    for i in range(4, img_num):
        last_result = cv2.cvtColor(last_result, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(last_result, None)
        new_image = cv2.imread(img_paths[i])
        if i == 4:
            new_image = adjust_gamma(new_image, 0.4)
        else:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)
            new_image[:, :, 0] = cv2.equalizeHist(new_image[:, :, 0])
            new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
        
        new_image = preprocess2(new_image, preprocess_shift)
        new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        new_image_gray = cv2.equalizeHist(new_image_gray)
        
        new_keypoints, new_descriptors = sift.detectAndCompute(new_image_gray, None)
        points = get_matches(new_keypoints, new_descriptors, keypoints, descriptors, debugging=False)
        H = RANSAC_for_H(points[0], points[1], RANSAC_n_iter=RANSAC_n_iter, RANSAC_threshold=RANSAC_threshold, debugging=False)
        warped_images.append(cv2.warpPerspective(src=new_image, M=H, dsize=size, flags=flag))
        last_result = blend(warped_images[3], warped_images[-1], detect_threshold=detect_threshold)
    
    # for i in range(img_num):
    #     cv2.imwrite(f'warped{i}.jpg', warped_images[i])
    
    print('finding gains')
    gains = task_gain_compensation(warped_images, sigma_n=sigma_n, sigma_g=sigma_g)
    print(gains)
    for i in range(img_num):
        warped_images[i] = (warped_images[i] * gains[i]).clip(0, 255).astype(np.uint8)
    
    print('blending')
    result = blend(warped_images[0], warped_images[1], detect_threshold=detect_threshold)
    for i in range(2, img_num):
        # cv2.imwrite(f'blended{i}.jpg', result)
        result = blend(result, warped_images[i], detect_threshold=detect_threshold)
    result = cv2.GaussianBlur(result, (1, 5), 0)
    
    result_path = os.path.join(result_path, 'result.jpg')
    cv2.imwrite(result_path, result)
    print(f'{result_path} is saved')
    


if __name__ == '__main__':
    Base()
    Challange()
