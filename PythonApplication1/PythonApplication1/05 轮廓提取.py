import cv2
import numpy as np
import matplotlib.pyplot as plt

def Otsu(srcImg_path):
    img = cv2.imread(srcImg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    
    return img

def Get_contour(bin_img):
    contour_img = np.zeros(shape=(bin_img.shape),dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(bin_img[i][j]==0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum ==  0:
                    contour_img[i][j] = 255

    return contour_img

#bin_img = Otsu('lenna.jpg')
#contour_img = Get_contour(bin_img)

#plt.subplot(121)
#plt.imshow(bin_img, cmap='gray')
#plt.title('Otsu')
#plt.axis('off')
#plt.subplot(122)
#plt.imshow(contour_img, cmap='gray')
#plt.title('contour')
#plt.axis('off')
#plt.show()


# 第一步：读入图像
img = cv2.imread('lenna.jpg')
# 第二步：对图像做灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 第三步：对图像做二值化处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 第四步：获得图像的轮廓值
contours, heriachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# 第五步：绘制图像轮廓
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
res = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

plt.imshow(res, cmap='gray')
plt.title('contour')
plt.axis('off')
plt.show()

