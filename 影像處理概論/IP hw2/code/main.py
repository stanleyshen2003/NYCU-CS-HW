from utils import histogram_equalization, histogram_specification
import cv2

# Q1
folder = 'img/'
image = cv2.imread(folder + 'Q1.jpg', cv2.IMREAD_GRAYSCALE)
new_image = histogram_equalization(image)
cv2.imwrite(folder + 'Q1_output.jpg', new_image)

# Q2
image = cv2.imread(folder + 'Q2_source.jpg', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread(folder + 'Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)
new_image = histogram_specification(image, reference)
cv2.imwrite(folder + 'Q2_output.jpg', new_image)