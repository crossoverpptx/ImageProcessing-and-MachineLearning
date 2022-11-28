from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#img = cv.imread('lenna.jpg')

## 均值滤波
#img_blur = cv.blur(img, (3,3)) # (3,3)代表卷积核尺寸，随着尺寸变大，图像会越来越模糊
#img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

## 方框滤波
#img_boxFilter1 = cv.boxFilter(img, -1, (3,3), normalize=True) # 当 normalize=True 时，与均值滤波结果相同
#img_boxFilter1 = cv.cvtColor(img_boxFilter1, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

#img_boxFilter2 = cv.boxFilter(img, -1, (3,3), normalize=False)
#img_boxFilter2 = cv.cvtColor(img_boxFilter2, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

## 高斯滤波
#img_GaussianBlur= cv.GaussianBlur(img, (3,3), 0, 0) # 参数说明：(源图像，核大小，x方向的标准差，y方向的标准差)
#img_GaussianBlur = cv.cvtColor(img_GaussianBlur, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

## 中值滤波
#img_medianBlur = cv.medianBlur(img, 3)
#img_medianBlur = cv.cvtColor(img_medianBlur, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

## 双边滤波
## 参数说明：(源图像，核大小，sigmaColor，sigmaSpace)
#img_bilateralFilter=cv.bilateralFilter(img, 50, 100, 100)
#img_bilateralFilter = cv.cvtColor(img_bilateralFilter, cv.COLOR_BGR2RGB) # BGR转化为RGB格式

#titles = ['img_blur', 'img_boxFilter1', 'img_boxFilter2',
#          'img_GaussianBlur', 'img_medianBlur', 'img_bilateralFilter']
#images = [img_blur, img_boxFilter1, img_boxFilter2, img_GaussianBlur, img_medianBlur, img_bilateralFilter]

#for i in range(6):
#    plt.subplot(3,3,i+1), plt.imshow(images[i]), plt.title(titles[i])
#    plt.axis('off')
#plt.show()

# 低通滤波
def Low_Pass_Filter(srcImg_path):
    #img = cv.imread('srcImg_path', 0)
    img = np.array(Image.open(srcImg_path))
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2) # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv.idft(ishift)
    res = cv.magnitude(iimg[:,:,0], iimg[:,:,1])
    
    return res

# 高通滤波
def High_Pass_Filter(srcImg_path):
    #img = cv.imread(srcImg_path, 0)
    img = np.array(Image.open(srcImg_path))
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2) # 中心位置
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv.idft(ishift)
    res = cv.magnitude(iimg[:,:,0], iimg[:,:,1])

    return res

img_Low_Pass_Filter = Low_Pass_Filter('lenna.jpg')
plt.subplot(121), plt.imshow(img_Low_Pass_Filter, 'gray'), plt.title('img_Low_Pass_Filter')
plt.axis('off')

img_High_Pass_Filter = High_Pass_Filter('lenna.jpg')
plt.subplot(122), plt.imshow(img_High_Pass_Filter, 'gray'), plt.title('img_High_Pass_Filter')
plt.axis('off')

plt.show()


