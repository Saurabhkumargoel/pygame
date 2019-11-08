import cv2
import numpy as np

ak = cv2.imread('/home/divesh/Pictures/AK/AK.jpeg')
eyebrow = cv2.imread('/home/divesh/Pictures/AK/AK.jpeg')
# eyebrow = cv2.imread('../images/eyebrows/e10.png')

alpha = 0.5
# print(img2)
cv2.addWeighted(ak, alpha, ak, 1 - alpha, 0, 1)

vis = np.concatenate((ak, eyebrow), axis=1)
# cv2.imwrite('out.png', vis)
cv2.imshow('out.png', vis)
# cv2.imshow('img2.png', img2)
# cv2.imshow('img2.png', img2)
cv2.waitKey(0)