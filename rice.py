import cv2
import numpy as np

img = cv2.imread(r'img/test/rice_121.png', cv2.IMREAD_COLOR)

h, w, _ = img.shape

GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #图片灰度化处理

ret,binary = cv2.threshold(GrayImage,255,0,cv2.THRESH_BINARY) #图片二值化,灰度值大于40赋值255，反之0

threshold = h/100 * w/100   #设定阈值

#cv2.fingContours寻找图片轮廓信息
"""提取二值化后图片中的轮廓信息 ，返回值contours存储的即是图片中的轮廓信息，是一个向量，内每个元素保存
了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓，有多少轮廓，向量contours就有
多少元素"""
contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i]) #计算轮廓所占面积
    if area < threshold:                         #将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
        cv2.drawContours(img,[contours[i]],-1, (0,0,0), thickness=-1)     #原始图片背景BGR值(84,1,68)
        continue

cv2.imshow('Output',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r'img/test/rice_121_1', img) #保存图片