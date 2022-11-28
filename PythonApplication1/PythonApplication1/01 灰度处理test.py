import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('lenna.png')
h,w,t = img.shape # 获取图像尺寸
print(h,w,t)
print(img[3,4])
gray = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
# 对原图像进行遍历，然后分别灰度化
for i in range(h):
    for j in range(w):
        gray[i,j] = max(img[i,j,0],img[i,j,1],img[i,j,2]) # 求3通道中最大值
print(gray[3,4])
gray = cv.cvtColor(gray,cv.COLOR_BGR2RGB)
plt.imshow(gray)
plt.title('Max_Gray')
#plt.axis('on')
plt.show()
