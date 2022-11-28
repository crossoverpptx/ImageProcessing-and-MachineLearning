import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

src = cv.imread("demo-line.jpg")
img = src.copy()
 
# 二值化图像（Canny边缘检测）
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
dst_img = cv.Canny(gray_img, 50, 150)
 
# 霍夫线变换
lines = cv.HoughLines(dst_img, 0.5, np.pi / 180, 300)
 
# 将检测的线绘制在原图上（注意是极坐标）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    # 找两个点
    x0 = rho * a
    y0 = rho * b
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
 
# 显示图像
plt.subplot(311), plt.imshow(src, 'gray'), plt.title('src_img'), plt.axis('off')
plt.subplot(312), plt.imshow(dst_img, 'gray'), plt.title('canny_img'), plt.axis('off')
plt.subplot(313), plt.imshow(img, 'gray'), plt.title('HoughLines_img'), plt.axis('off')
plt.show()


