import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

## 计算累计分布函数
#def C(rk):
#  # 读取图片灰度直方图
#  # bins为直方图直方柱的取值向量，hist为bins各取值区间上的频数取值
#  hist, bins = np.histogram(rk, 256, [0, 256])
#  # 计算累计分布函数
#  return hist.cumsum()

## 计算灰度均衡化映射
#def T(rk):
#  cdf = C(rk)
#  # 均衡化
#  cdf = (cdf - cdf.min()) * (255 - 0) / (cdf.max() - cdf.min()) + 0
#  #cdf = 255.0 * cdf / cdf[-1]
#  return cdf.astype('uint8')

## 读取图片
#img = cv.imread('lenna.jpg', 0)
## 将二维数字图像矩阵转变为一维向量
#rk = img.flatten()
## 原始图像灰度直方图
#plt.hist(rk, 256, [0, 255], color = 'r')
## 直方图均衡化
#imgDst = T(rk)[img]     
#plt.hist(imgDst.flatten(), 256, [0, 255], color = 'b')
#plt.legend(['Before Equalization','Equalization']) 
#plt.show()
## 展示前后对比图像
#plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Gray')
#plt.subplot(122), plt.imshow(imgDst, cmap='gray'), plt.title('Histogram Equalization')
#plt.show()


img = cv.imread('lenna.jpg', 0)
dst = cv.equalizeHist(img)
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Gray'), plt.axis('off')
plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Histogram Equalization'), plt.axis('off')
plt.show()



