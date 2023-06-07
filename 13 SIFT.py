import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 得到特征点
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)  # 关键点
img = cv2.drawKeypoints(gray, kp, img)

plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('Gray Image'), plt.axis('off')
plt.subplot(122), plt.imshow(img, 'gray'), plt.title('Keypoints Image'), plt.axis('off')
plt.show()

# 计算特征
# kp为关键点keypoints
# des为描述子descriptors
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape) #(203,)
print(des.shape) #(203, 128)，128维向量

