import cv2
import numpy as np
import matplotlib.pyplot as plt

#####################################################################################
#img = cv2.imread("lenna.jpg")
#height, width = img.shape[:2]  # 获取图像的高度和宽度
#cv2.imshow('src', img)

## 缩放到原来的二分之一
#img_test1 = cv2.resize(img, (int(height / 2), int(width / 2)))
#cv2.imshow('resize1', img_test1)

## 最近邻插值法缩放，缩放到原来的四分之一
#img_test2 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
#cv2.imshow('resize2', img_test2)
#cv2.waitKey()
#cv2.destroyAllWindows()

######################################################################################
#import imutils
#img = cv2.imread("lenna.jpg")
#cv2.imshow('src', img)

## 旋转45度，可能局部丢失，缺失部分用黑色填充
#rot1 = imutils.rotate(img, angle=45)
#cv2.imshow("Rotated1", rot1)

## 旋转45度，保持原图完整，旋转完成图分辨率会改变
#rot2 = imutils.rotate_bound(img, angle=45)
#cv2.imshow("Rotated2", rot2)
#cv2.waitKey()
#cv2.destroyAllWindows()


#######################################################################################
#img = cv2.imread("lenna.jpg")  # 读取彩色图像(BGR)

#imgFlip1 = cv2.flip(img, 0)  # 垂直翻转
#imgFlip2 = cv2.flip(img, 1)  # 水平翻转
#imgFlip3 = cv2.flip(img, -1)  # 水平和垂直翻转

#plt.subplot(221), plt.axis('off'), plt.title("Original")
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 原始图像
#plt.subplot(222), plt.axis('off'), plt.title("Flipped Horizontally")
#plt.imshow(cv2.cvtColor(imgFlip2, cv2.COLOR_BGR2RGB))  # 水平翻转
#plt.subplot(223), plt.axis('off'), plt.title("Flipped Vertically")
#plt.imshow(cv2.cvtColor(imgFlip1, cv2.COLOR_BGR2RGB))  # 垂直翻转
#plt.subplot(224), plt.axis('off'), plt.title("Flipped Horizontally & Vertically")
#plt.imshow(cv2.cvtColor(imgFlip3, cv2.COLOR_BGR2RGB))  # 水平垂直翻转
#plt.show()


img = cv2.imread('lenna.jpg')
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 
# 图像下、上、右、左平移
M = np.float32([[1, 0, 0], [0, 1, 100]])
img1 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
M = np.float32([[1, 0, 0], [0, 1, -100]])
img2 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
M = np.float32([[1, 0, 100], [0, 1, 0]])
img3 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
M = np.float32([[1, 0, -100], [0, 1, 0]])
img4 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
# 显示图形
titles = [ 'Image-down', 'Image-up', 'Image-right', 'Image-left']  
images = [img1, img2, img3, img4]  
for i in range(4):  
   plt.subplot(2,2,i+1), plt.imshow(images[i]), plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()  


