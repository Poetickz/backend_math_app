import numpy as np 
import cv2
import sys
from knn import Knn

knn = Knn()
knn.set_k(50)

im = cv2.imread('60.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# (2) threshold-inv and morph-open 
th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2)))
# (3) find and filter contours, then draw on src 
cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for cnt in cnts:
    x,y,w,h = bbox = cv2.boundingRect(cnt)
    if  h>28:
        cv2.rectangle(cnt, (x,y), (x+w, y+h), (255, 0, 255), 1, cv2.LINE_AA)
        roi = threshed[y:y+h+100,x:x+w+50]
        roismall = cv2.resize(roi,(28,28))
        cv2.imshow( "Display window", roismall )
        cv2.waitKey(0); 
        roismall = roismall.flatten()
        print(knn.predict(roismall))
        
