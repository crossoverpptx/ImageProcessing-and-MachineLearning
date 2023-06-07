import cv2 as cv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def Easy_Binarization(srcImg_path):
    img = cv.imread(srcImg_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray[img_gray>127] = 255
    img_gray[img_gray<=127] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.title('Easy_Binarization')
    plt.show()

#def Easy_Binarization(srcImg_path):
#    img = cv.imread(srcImg_path)
#    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#    h,w = gray.shape # 获取图像尺寸
#    binarization = np.zeros((h,w),dtype=img.dtype) # 自定义空白单通道图像，用于存放二值化图像
#    for i in range(h):
#        for j in range(w):
#            if gray[i,j]>127:
#                binarization[i,j] = 255
#            else:
#                binarization[i,j] = 0
            
#    plt.imshow(binarization, cmap='gray')
#    plt.title('Easy_Binarization')
#    plt.show()

def Mean_Binarization(srcImg_path):
    img = cv.imread(srcImg_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = np.mean(img_gray)
    print(threshold)
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.title('Mean_Binarization')
    plt.show()

def Hist_Binarization(srcImg_path):
    img = cv.imread(srcImg_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = img_gray.flatten()
    plt.subplot(121)
    plt.hist(hist,256)

    cnt_hist = Counter(hist)
    print(cnt_hist)
    begin,end = cnt_hist.most_common(2)[0][0],cnt_hist.most_common(2)[1][0]
    if begin > end:
        begin, end = end, begin
    print(f'{begin}: {end}')

    cnt = np.iinfo(np.int16).max
    threshold = 0
    for i in range(begin,end+1):
        if cnt_hist[i]<cnt:
            cnt = cnt_hist[i]
            threshold = i
    print(f'{threshold}: {cnt}')
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.subplot(122)
    plt.imshow(img_gray, cmap='gray')
    plt.show()

def Otsu(srcImg_path):
    img = cv.imread(srcImg_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    threshold_t = 0
    max_g = 0
    
    for t in range(255):
        front = img[img < t]
        back = img[img >= t]
        front_p = len(front) / (h * w)
        back_p = len(back) / (h * w)
        front_mean = np.mean(front) if len(front) > 0 else 0.
        back_mean = np.mean(back) if len(back) > 0 else 0.
        
        g = front_p * back_p * ((front_mean - back_mean)**2)
        if g > max_g:
            max_g = g
            threshold_t = t
    print(f"threshold = {threshold_t}")

    img[img < threshold_t] = 0
    img[img >= threshold_t] = 255
    plt.imshow(img, cmap='gray')
    plt.title('Otsu')
    plt.show()

#Easy_Binarization('lenna.jpg')
#Mean_Binarization('lenna.jpg')
#Hist_Binarization('lenna.jpg')
#Otsu('lenna.jpg')

#img = cv.imread('lenna.jpg')
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
#ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)
#ret,thresh3 = cv.threshold(img_gray,127,255,cv.THRESH_TRUNC)
#ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO)
#ret,thresh5 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)

#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

#for i in range(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.axis('off')
#plt.show()

img = cv.imread('lenna.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
ret,th1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Simple Thresholding',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()

