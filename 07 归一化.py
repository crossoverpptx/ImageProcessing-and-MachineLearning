from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#img = cv.imread('lenna.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#h,w = gray.shape
#mn = np.min(gray)
#mx = np.max(gray)
#norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
#for i in range(h):
#    for j in range(w):
#        norm[i,j] = (gray[i,j] - mn) / (mx - mn)
#        #norm[i,j] = gray[i,j] / 255

#print('归一化前：')
#print(gray)
#print('归一化后：')
#print(norm)

#plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
#plt.axis('off')
#plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
#plt.axis('off')
#plt.show()


#img = cv.imread('lenna.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#h,w = gray.shape
#x_mean = np.mean(gray)
#vari = np.sqrt((np.sum((gray-x_mean)**2))/(h*w))
#norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
#for i in range(h):
#    for j in range(w):
#        norm[i,j] = (gray[i,j] - x_mean) / vari
#        #norm[i,j] = gray[i,j] / 127.5 - 1

#print('归一化前：')
#print(gray)
#print('归一化后：')
#print(norm)

#plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
#plt.axis('off')
#plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
#plt.axis('off')
#plt.show()


#img = cv.imread('lenna.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#h,w = gray.shape
#norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
#norm = np.log10(gray) / np.log10(gray.max())

#print('归一化前：')
#print(gray)
#print('归一化后：')
#print(norm)

#plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
#plt.axis('off')
#plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
#plt.axis('off')
#plt.show()



#img = cv.imread('lenna.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#h,w = gray.shape
#norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
#norm = np.arctan(gray) * (2 / np.pi)

#print('归一化前：')
#print(gray)
#print('归一化后：')
#print(norm)

#plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
#plt.axis('off')
#plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
#plt.axis('off')
#plt.show()

#img = cv.imread('lenna.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#h,w = gray.shape
#norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
#for i in range(h):
#    for j in range(w):
#        norm_x = 0.0 + gray[i,j]**2
#norm_x = np.sqrt(norm_x)
#norm = gray / norm_x

#print('归一化前：')
#print(gray)
#print('归一化后：')
#print(norm)

#plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
#plt.axis('off')
#plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
#plt.axis('off')
#plt.show()



img = cv.imread('lenna.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
h,w = gray.shape
norm = np.zeros((h,w),dtype=np.float32) # 自定义空白单通道图像，用于存放归一化图像
cv.normalize(gray, norm, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
# norm = np.uint8(norm*255.0)

print('归一化前：')
print(gray)
print('归一化后：')
print(norm)

plt.subplot(121), plt.imshow(gray, 'gray'), plt.title('gray')
plt.axis('off')
plt.subplot(122), plt.imshow(norm, 'gray'), plt.title('normalization')
plt.axis('off')
plt.show()


