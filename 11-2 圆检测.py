import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('demo-circle.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,200,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(121), plt.imshow(img, 'gray'), plt.title('src_img'), plt.axis('off')
plt.subplot(122), plt.imshow(cimg, 'gray'), plt.title('HoughCircles_img'), plt.axis('off')
plt.show()

