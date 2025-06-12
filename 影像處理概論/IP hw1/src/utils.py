import numpy as np

def point_transform(img, x, y, degree):
    width, height = img.shape[:2]
    x = x-width/2
    y = y-height/2
    x1 = x * np.cos(degree / 180 * np.pi) - y * np.sin(degree / 180 * np.pi) + width / 2
    y1 = x * np.sin(degree / 180 * np.pi) + y * np.cos(degree / 180 * np.pi) + height / 2
    return x1, y1

def f_x_bicubic(x, p0, p1, p2, p3):
    return (-p0/2 + 3*p1/2 - 3*p2/2 + p3/2)* x**3 + (p0 - 5*p1/2 + 2*p2 - p3/2)* x**2 + (-p0/2 + p2/2)* x + p1

def get_point_value_bicubic(img, x, y):
    width, height = img.shape[:2]
    x0 = max(0, int(x-1))
    x1 = int(x)
    x2 = min(x1+1, width-1)
    x3 = min(x1+2, width-1)
    y0 = max(0, int(y-1))
    y1 = int(y)
    y2 = min(y1+1, height-1)
    y3 = min(y1+2, height-1)
    x = x-x1
    y = y-y1
    r0 = f_x_bicubic(x, img[x0, y0], img[x1, y0], img[x2, y0], img[x3, y0])
    r1 = f_x_bicubic(x, img[x0, y1], img[x1, y1], img[x2, y1], img[x3, y1])
    r2 = f_x_bicubic(x, img[x0, y2], img[x1, y2], img[x2, y2], img[x3, y2])
    r3 = f_x_bicubic(x, img[x0, y3], img[x1, y3], img[x2, y3], img[x3, y3])
    return np.clip(f_x_bicubic(y, r0, r1, r2, r3), [0,0,0], [255,255,255]).astype(np.uint8)

def get_point_value_bilinear(img, x, y):
    width, height = img.shape[:2]
    x1 = int(x)
    y1 = int(y)
    x2 = min(x1+1, width-1)
    y2 = min(y1+1, height-1)
    x = x-x1
    y = y-y1
    return (1-x)*(1-y)*img[x1, y1] + x*(1-y)*img[x2, y1] + (1-x)*y*img[x1, y2] + x*y*img[x2, y2]

def magnify(img, method):
    width, height = img.shape[:2]
    new_img = np.zeros((width*2, height*2, 3), np.uint8)
    img = img.astype(np.float32)
    for i in range(width*2):
        for j in range(height*2):
            x = i/2
            y = j/2
            if method == 'bilinear':
                new_img[i, j] = get_point_value_bilinear(img, x, y)
            elif method == 'bicubic':            
                new_img[i, j] = get_point_value_bicubic(img, x, y)
            else:
                new_img[i, j] = img[min(int(x + 0.5), width - 1), min(int(y + 0.5), height - 1)]
    return new_img

def rotate(img, degree, method):
    width, height = img.shape[:2]
    new_img = np.zeros((width, height, 3), np.uint8)
    img = img.astype(np.float32)
    for i in range(width):
        for j in range(height):
            x, y = point_transform(img, i, j, degree)
            if(x < 0 or x >= width or y < 0 or y >= height):
                continue
            if method == 'bilinear':
                new_img[i, j] = get_point_value_bilinear(img, x, y)
            elif method == 'bicubic':
                new_img[i, j] = get_point_value_bicubic(img, x, y)
            else:
                new_img[i, j] = img[min(int(x + 0.5), width - 1), min(int(y + 0.5), height - 1)]
    return new_img