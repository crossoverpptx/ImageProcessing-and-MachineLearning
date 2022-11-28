from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

## detector parameters
#block_size = 5
#sobel_size = 3
#k = 0.04

#image = cv2.imread('lenna.jpg')
#gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## uint8类型转化为float32类型  
#gray_img = np.float32(gray_img)

## detect the corners with appropriate values as input parameters
#corners_img = cv2.cornerHarris(gray_img, block_size, sobel_size, k)

## result is dilated for marking the corners, not necessary
#dst = cv2.dilate(corners_img, None)

## Threshold for an optimal value, marking the corners in Green? Red
#image[corners_img>0.01*corners_img.max()] = [0,0,255]

#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#plt.imshow(image), plt.title('Harris'), plt.axis('off')
#plt.show()


#maxCorners = 100
#qualityLevel = 0.01
#minDistance = 10

#image = cv2.imread('lenna.jpg')
#gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#corners = cv2.goodFeaturesToTrack(gray_img, maxCorners, qualityLevel, minDistance)

#corners = np.int0(corners)
#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(image,(x,y),2,(0,0,255),-1)
    
#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#plt.imshow(image), plt.title('Shi-Tomasi'), plt.axis('off')
#plt.show()


image = cv2.imread('lenna.jpg')
gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(gray_img,None)
img2 = cv2.drawKeypoints(image, kp, None, color=(0,0,255))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(gray_img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv2.drawKeypoints(image, kp, None, color=(0,0,255))

img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(img2), plt.title('fast_true'), plt.axis('off')
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
plt.subplot(122), plt.imshow(img3), plt.title('fast_false'), plt.axis('off')
plt.show()
