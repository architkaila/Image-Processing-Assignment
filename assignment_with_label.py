import cv2
import numpy as np

kern1 = np.ones((2,2), np.uint8)/4
kern2 = np.ones((2,6),np.uint8)/12

#Reading the given input image
frame = cv2.imread('StackedCups.jpg')
#Storing a copy of original image to work upon
frame2 = frame.copy()
cv2.rectangle(frame2,(22,22),(200,200),(0,0,255),2)

image = frame.copy()
#To specify which areas are background, foreground or probable background/foreground
mask = np.zeros(image.shape[:2],np.uint8)

backgroundModel = np.zeros((1,65),np.float64)
foregroundModel = np.zeros((1,65),np.float64)

#coordinates of the rectangle (area of image which contains the cups)
#in this case I took from 10% of image size to 90% of image size 
rect = (22,22,200,200)

#GrabCut is an algorithm to extract foreground from images
cv2.grabCut(image,mask,rect,backgroundModel,foregroundModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image2 = image*mask2[:,:,np.newaxis]

#Saving the image that contains the foreground
cv2.imwrite('foreground.jpg',image2)

image3 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image3 = cv2.medianBlur(image3,5)
image3 = cv2.adaptiveThreshold(image3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
image3 = cv2.erode(image3,kern1,iterations=1)

#Saving the Binary image obtained from the foreground image
cv2.imwrite('binary.jpg',image3)

image_inverted = cv2.bitwise_not(image3)
image_inverted = cv2.erode(image_inverted,kern2,iterations = 2)
image_inverted = cv2.dilate(image_inverted,kern2,iterations = 2)

#Saving the Inverted image obtained from Binary image
cv2.imwrite('inverted.jpg',image_inverted)

#findContours is used for edge detection of cups in the image
image_inverted2, contours, hierarchy = cv2.findContours(image_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#To Count the number of cups & mark red rectangular labels around them
count = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    if (aspect_ratio > 1) & (area > 100) & (perimeter > 100):
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        count += 1
cv2.imshow('Labelled', image)

print('Number of Cups: {}'.format(count))

foreground = cv2.imread('foreground.jpg', 1)
cv2.namedWindow('Foreground', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground',foreground)

binarized = cv2.imread('binary.jpg', 1)
cv2.namedWindow('Binarized', cv2.WINDOW_NORMAL)
cv2.imshow('Binarized',binarized)

inverted = cv2.imread('inverted.jpg', 1)
cv2.namedWindow('Inverted', cv2.WINDOW_NORMAL)
cv2.imshow('Inverted',inverted)

cv2.waitKey(0)
cv2.destroyAllWindows()