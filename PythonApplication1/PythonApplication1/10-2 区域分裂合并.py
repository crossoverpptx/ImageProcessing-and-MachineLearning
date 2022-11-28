import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#分裂
def Division_Judge(img, h0, w0, h, w) :
    area = img[h0 : h0 + h, w0 : w0 + w]
    mean = np.mean(area)
    std = np.std(area, ddof = 1)

    total_points = 0
    operated_points = 0

    for row in range(area.shape[0]) :
        for col in range(area.shape[1]) :
            if (area[row][col] - mean) < 2 * std :
                operated_points += 1
            total_points += 1

    if operated_points / total_points >= 0.95 :
        return True
    else :
        return False

def Merge(img, h0, w0, h, w) :
    # area = img[h0 : h0 + h, w0 : w0 + w]
    # _, thresh = cv.threshold(area, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    # img[h0 : h0 + h, w0 : w0 + w] = thresh
    for row in range(h0, h0 + h) :
        for col in range(w0, w0 + w) :
            if img[row, col] > 100 and img[row, col] < 200:
                img[row, col] = 0
            else :
                img[row, col] = 255

def Recursion(img, h0, w0, h, w) :
    #如果满足分裂条件继续分裂
    if not Division_Judge(img, h0, w0, h, w) and min(h, w) > 5 :
        #递归继续判断能否继续分裂
        #左上方块
        Division_Judge(img, h0, w0, int(h0 / 2), int(w0 / 2))
        #右上方块
        Division_Judge(img, h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
        #左下方块
        Division_Judge(img, h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
        #右下方块
        Division_Judge(img, h0 + int(h0 / 2), w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
    else :
        #合并
        Merge(img, h0, w0, h, w)

def Division_Merge_Segmented() :
    img = cv.imread('lenna.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist, bins = np.histogram(img_gray, bins = 256)
    print(f'五角星、椭圆、背景、五边形的像素值分别为：'
          f'{"、".join("%s" % pixel for pixel in np.unique(img_gray))}')

    segemented_img = img_gray.copy()
    Recursion(segemented_img, 0, 0, segemented_img.shape[0], segemented_img.shape[1])

    #plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off'), plt.title(f'$input\_image$')
    plt.subplot(132), plt.imshow(img_gray, cmap='gray', vmin = 0, vmax = 255)
    plt.axis('off'), plt.title(f'$gray\_image$')
    plt.subplot(133), plt.imshow(segemented_img, cmap='gray')
    plt.axis('off'), plt.title(f'$segmented\_image$')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Division_Merge_Segmented()