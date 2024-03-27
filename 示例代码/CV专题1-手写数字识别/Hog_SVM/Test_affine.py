import cv2
import numpy as np

img = cv2.imread('image\h.png')
h,w,ch= img.shape
#平移
M1 = np.float32([[1,0,100],[0,1,50]])
#缩放
M2 = np.float32([[2,0,0],[0,4,0]])
##旋转
#CV
M0 = cv2.getRotationMatrix2D((w/2,h/2),-30,1)
print(M0)
newimg = cv2.warpAffine(img,M0,(w,h))
#newimg = cv2.warpAffine(img,M,(int(w/2),int(h/2)))
print(newimg.shape)
#imags = np.hstack((img, newimg))
cv2.imshow('1',img)
cv2.imshow('2',newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()