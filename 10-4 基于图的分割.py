
import cv2
import numpy as np
from matplotlib import pyplot as plt 


img = cv2.imread('lenna.jpg')
r = cv2.selectROI('input', img, False)  # 返回 (x_min, y_min, w, h)
print("input:", r)

# roi区域
roi = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# 原图mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))  # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1, 65), np.float64)  # bg模型的临时数组
fgdmodel = np.zeros((1, 65), np.float64)  # fg模型的临时数组

cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)

# 提取前景和可能的前景区域
mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
print(mask2.shape)

result = cv2.bitwise_and(img, img, mask=mask2)

plt.subplot(121), plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), plt.title('roi_img'), plt.axis('off')
plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Grabcut_image'), plt.axis('off')
plt.show()

