from utils import magnify
import cv2


img = cv2.imread('building.jpg')

new_img = magnify(img, method='nearest')
cv2.imwrite('./result/magnify/building_nearest_neighbor.jpg', new_img)

new_img = magnify(img, method='bilinear')
cv2.imwrite('./result/magnify/building_bilinear.jpg', new_img)

new_img = magnify(img, method='bicubic')
cv2.imwrite('./result/magnify/building_bicubic.jpg', new_img)

