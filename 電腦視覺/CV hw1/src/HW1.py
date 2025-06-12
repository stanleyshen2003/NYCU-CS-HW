import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from utils import preprocess_gaussian_filter, mask_image_pixel, preprocess_low_pass, mask_normal, normal_low_pass
import argparse

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

def save_npy(N,used, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, N)
        np.save(f, used)

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

# read all the images and light sources from the object folder given the object name
def read_object(object_name):
    all_images = []
    light_sources = []
    for filename in sorted(os.listdir(os.path.join('data', object_name))):
        if filename.endswith('.bmp'):
            image = read_bmp(os.path.join('data', object_name, filename))
            #image = image.reshape(image_row * image_col)
            all_images.append(image)
        elif filename.endswith('.txt'):
            with open(os.path.join('data', object_name, filename)) as f:
                lines = f.readlines()
                for line in lines:
                    light_source = line.split('(')[1].split(')')[0].split(',')
                    sum = float(light_source[0])**2 + float(light_source[1])**2 + float(light_source[2])**2
                    sum = np.sqrt(sum)
                    light_sources.append([float(light_source[0]) / sum, float(light_source[1]) / sum, float(light_source[2]) / sum])
    return np.array(all_images, dtype=np.float32), np.array(light_sources)

# calculate the normal map given the images and light sources
def get_normal_map(images, light_sources):
    kdn = np.dot(np.dot(np.linalg.inv(np.dot(light_sources.T, light_sources)), light_sources.T), images)
    norms = np.linalg.norm(kdn, axis=0).reshape(image_row*image_col, 1)
    kdn = kdn.T
    normal_map = kdn/norms
    return normal_map

# mask the normal map
def get_mask_from_normal_map(normal_map):
    mask = np.zeros((image_row, image_col))
    normal_map = np.reshape(normal_map, (image_row, image_col, 3))

    for i in range(image_row):
        for j in range(image_col):
            if np.isnan(normal_map[i][j][0]):
                mask[i][j] = 0
            else:
                mask[i][j] = 1
    return mask


# calculate the depth map given the normal map and mask
def get_depth_map(normal_map, mask):
    normal_map = normal_map.reshape(image_row, image_col, 3)
    M = []
    V = []
    threshold = 5
    used = np.zeros((image_row * image_col))

    for i in range(image_row):
        for j in range(image_col-1):
            if mask[i][j] == 0:
                continue
            temp = np.zeros((image_row * image_col))
            temp[i * image_col + j] = -1
            temp[i * image_col + j + 1] = 1
            used[i * image_col + j] = 1
            used[i * image_col + j + 1] = 1
            M.append(temp)
            # clipping for extreme values
            V.append(min(max(- normal_map[i][j][0] / normal_map[i][j][2], -threshold), threshold))
            if mask[i][j+1] == 0:
                temp = np.zeros((image_row * image_col))
                temp[i * image_col + j+1] = 1
                used[i * image_col + j+1] = 1
                M.append(temp)
                V.append(0)
    for i in range(image_col):
        for j in range(image_row-1):
            if mask[j][i] == 0:
                continue
            temp = np.zeros((image_row * image_col))
            temp[j * image_col + i] = -1
            used[j * image_col + i] = 1
            temp[(j+1) * image_col + i] = 1
            used[(j+1) * image_col + i] = 1
            M.append(temp)
            V.append(min(max(normal_map[j][i][1] / normal_map[j][i][2], -threshold), threshold))
            if mask[j+1][i] == 0:
                temp = np.zeros((image_row * image_col))
                temp[(j+1) * image_col + i] = 1
                used[(j+1) * image_col + i] = 1
                M.append(temp)
                V.append(0)
                
    M = np.array(M, dtype=np.float32)
    V = np.array(V, dtype=np.float32).reshape(-1, 1)
    M = M.T
    M = [M[:][j] for j in range(image_row*image_col) if used[j]]
    M = np.array(M, dtype=np.float32)
    M = M.T
    z = np.dot(np.dot(np.linalg.inv(np.dot(M.T, M)), M.T), V)

    # fill in the values while rescale to (0, 40)
    count = 0
    temp = []
    maximum = np.max(z)
    minimum = np.min(z)
    for i in range(image_row*image_col):
        if used[i] == 1:
            temp.append((z[count][0]- (minimum))*40/(maximum-minimum))
            count += 1
        else:
            temp.append(0.0)
    return np.array(temp, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, default='bunny')
    args = parser.parse_args()
    
    object_name = args.object_name

    images, light_sources = read_object(object_name)
    # folder = 'no_normal_filter/'
    # os.makedirs(folder, exist_ok=True)
    # os.makedirs(folder + object_name, exist_ok=True)
    if object_name == 'noisy_venus':
        for image in images:
            image = preprocess_low_pass(image)
        images = mask_image_pixel(images)
    
    images = images.reshape(-1, image_row*image_col)
    normal_map = get_normal_map(images, light_sources)
    if object_name == 'noisy_venus':
        normal_map = normal_map.reshape((image_row, image_col, 3))
        normal_map = mask_normal(normal_map)
        #normal_map = normal_low_pass(normal_map)
    normal_visualization(normal_map)
    #plt.savefig(folder + object_name + '/normal_map.png')
    
    mask = get_mask_from_normal_map(normal_map)
    depth_map = get_depth_map(normal_map, mask)
    depth_map = preprocess_gaussian_filter(depth_map.reshape(image_row, image_col))
    depth_map = depth_map -20
    depth_visualization(depth_map)
    #plt.savefig(folder + object_name + '/depth_map.png')

    save_ply(depth_map, object_name+'.ply')
    show_ply(object_name+'.ply')
    
    # showing the windows of all visualization function
    plt.show()
    