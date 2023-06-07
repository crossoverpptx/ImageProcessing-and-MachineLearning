import cv2
import numpy as np
import matplotlib.pyplot as plt

def Roberts(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Roberts算子
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 显示图形
    titles = ["Original Image", "Roberts Image"]
    images = [img, Roberts]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def Prewitt(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 显示图形
    titles = ["Original Image", "Prewitt Image"]
    images = [img, Prewitt]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def Sobel_demo(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel算子
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    # 转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 显示图形
    titles = ["Original Image", "Sobel Image"]
    images = [img, Sobel]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def Laplacian_demo(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Laplacian算子
    Laplacian = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    # 转成uint8
    Laplacian = cv2.convertScaleAbs(Laplacian)
    # 显示图形
    titles = ["Original Image", "Laplacian Image"]
    images = [img, Laplacian]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def Canny_demo(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    img_GaussianBlur = cv2.GaussianBlur(gray, (3,3), 0)
    # Canny算子
    Canny = cv2.Canny(img_GaussianBlur, 0, 100)
    # 显示图形
    titles = ["Original Image", "Canny Image"]
    images = [img, Canny]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

#Roberts('lenna.jpg')
#Prewitt('lenna.jpg')
#Sobel_demo('lenna.jpg')
Laplacian_demo('lenna.jpg')
#Canny_demo('lenna.jpg')