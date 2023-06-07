# -*- coding:utf-8 -*-

from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2

# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius   # 领域像素点数

image = cv2.imread('lenna.jpg')  # 读取图像
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 按照RGB顺序展示原图
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)   # 灰度转换

# LBP处理
lbp = local_binary_pattern(image2, n_points, radius)

plt.subplot(131), plt.imshow(image1), plt.title('Original Image'), plt.axis('off')
plt.subplot(132), plt.imshow(image2, 'gray'), plt.title('Gray Image'), plt.axis('off')
plt.subplot(133), plt.imshow(lbp, 'gray'), plt.title('LBP Image'), plt.axis('off')
plt.show()


