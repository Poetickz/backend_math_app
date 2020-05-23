import numpy as np 
import cv2
import sys
from knn import Knn

knn = Knn()
knn.set_k(20)

im = cv2.imread('4.png')
im = cv2.resize(im,(28,28))
image_array = im.reshape((1,784))

# cv2.imshow( "Display window", im )
# cv2.waitKey(0)
for i in image_array:
  print(i, end=" ")
print(len(image_array))
print(knn.predict(image_array))


# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5), 0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# out = np.zeros(im.shape,np.uint8)

# # Finding Contours
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     if cv2.contourArea(cnt)>50:
#         [x,y,w,h] = cv2.boundingRect(cnt)
#         if  h>28:
#             cv2.imshow( "Display window", im )
#             waitKey(0);                                          
#             roi = thresh[y:y+h,x:x+w]
#             roismall = roismall.reshape((1,784))
#             print(knn.predict(roismall[0]))

