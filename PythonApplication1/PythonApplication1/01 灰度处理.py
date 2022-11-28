import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#img = cv.imread('lenna.png')
#img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR转化为RGB格式
#plt.imshow(img1)
#plt.title('Src_img')
#plt.show()

def Max_Gray(srcImg_path):
    img = cv.imread(srcImg_path)
    h,w = img.shape[0:2] # 获取图像尺寸
    gray = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    # 对原图像进行遍历，然后分别灰度化
    for i in range(h):
        for j in range(w):
            gray[i,j] = max(img[i,j,0],img[i,j,1],img[i,j,2]) # 求3通道中最大值
    gray = cv.cvtColor(gray,cv.COLOR_BGR2RGB)
    plt.imshow(gray)
    plt.title('Max_Gray')
    #plt.axis('on')
    plt.show()

def Avrage_Gray(srcImg_path):
    img = cv.imread(srcImg_path)
    h,w = img.shape[0:2] # 获取图像尺寸
    gray = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    # 对原图像进行遍历，然后分别灰度化
    for i in range(h):
        for j in range(w):
            gray[i,j] = (int(img[i,j,0])+int(img[i,j,1])+int(img[i,j,2]))/3 # 求3通道像素的平均值作为灰度值
    gray = cv.cvtColor(gray,cv.COLOR_BGR2RGB)
    plt.imshow(gray)
    plt.title('Avrage_Gray')
    #plt.axis('on')
    plt.show()

def WeightedAvrage_Gray(srcImg_path):
    img = cv.imread(srcImg_path)
    h,w = img.shape[0:2] # 获取图像尺寸
    gray = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    # 对原图像进行遍历，然后分别灰度化
    for i in range(h):
        for j in range(w):
            gray[i,j] = 0.114*int(img[i,j,0])+0.578*int(img[i,j,1])+0.299*int(img[i,j,2])
    gray = cv.cvtColor(gray,cv.COLOR_BGR2RGB)
    plt.imshow(gray)
    plt.title('WeightedAvrage_Gray')
    #plt.axis('on')
    plt.show()

#Max_Gray('lenna.png')
#Avrage_Gray('lenna.png')
#WeightedAvrage_Gray('lenna.png')

def Show(srcImg_path):
    img = cv.imread(srcImg_path)
    h,w = img.shape[0:2] # 获取图像尺寸
    gray1 = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    gray2 = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    gray3 = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放灰度图
    # 对原图像进行遍历，然后分别灰度化
    for i in range(h):
        for j in range(w):
            gray1[i,j] = max(img[i,j,0],img[i,j,1],img[i,j,2]) # 求3通道中最大值
            gray2[i,j] = (int(img[i,j,0])+int(img[i,j,1])+int(img[i,j,2]))/3 # 求3通道像素的平均值作为灰度值
            gray3[i,j] = 0.114*int(img[i,j,0])+0.578*int(img[i,j,1])+0.299*int(img[i,j,2])
    gray1 = cv.cvtColor(gray1,cv.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(gray1)
    plt.title('Max_Gray')
    gray2 = cv.cvtColor(gray2,cv.COLOR_BGR2RGB)
    plt.subplot(132)
    plt.imshow(gray2)
    plt.title('Avrage_Gray')
    gray3 = cv.cvtColor(gray3,cv.COLOR_BGR2RGB)
    plt.subplot(133)
    plt.imshow(gray3)
    plt.title('WeightedAvrage_Gray')
    plt.show()

#Show('lenna.png')

img = cv.imread('lenna.png')
img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB) # BGR转化为RGB格式
plt.subplot(121)
plt.imshow(img1)
plt.title('Src_img')

# 灰度转换
image2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(122)
plt.imshow(image2, plt.cm.gray)
plt.title('Gray_img')
plt.show()


