import cv2
from utils import rotate


img = cv2.imread('building.jpg')

new_img = rotate(img, 30, method='nearest')
cv2.imwrite('./result/rotate/building_nearest_neighbor.jpg', new_img)

new_img = rotate(img, 30, method='bilinear')
cv2.imwrite('./result/rotate/building_bilinear.jpg', new_img)

new_img = rotate(img, 30, method='bicubic')
cv2.imwrite('./result/rotate/building_bicubic.jpg', new_img)