import cv2
import numpy as np
import matplotlib.pyplot as plt
 
## 读取图像
#img = cv2.imread('lenna.jpg', 0)
 
## 傅里叶变换
#dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
 
## 将频谱低频从左上角移动至中心位置
#dft_shift = np.fft.fftshift(dft)
 
## 频谱图像双通道复数转换为0-255区间
#result = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
 
## 显示图像
#plt.subplot(121), plt.imshow(img, cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(result, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()
 
# 读取图像
img = cv2.imread('lenna.jpg', 0)
 
# 傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
 
# 傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
 
# 显示图像
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.show()