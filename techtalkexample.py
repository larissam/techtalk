# import sys
# import cv2 
# import numpy as np

# imagePath = sys.argv[1]

# faceCascade = cv2.CascadeClassifier('haarcascade.xml')
# eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# img = cv2.imread(imagePath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(
# 	gray, 
# 	scaleFactor = 1.1,
# 	minNeighbors = 5,
# 	minSize = (30, 30),
# 	flags = cv2.cv.CV_HAAR_SCALE_IMAGE
# 	)

# print "Found {0} faces!".format(len(faces))

# for (x,y,w,h) in faces:
# 	img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 	roi_gray = gray[y:y+h, x:x+w]
# 	roi_color = img[y:y+h, x:x+w]
# 	eyes = eyeCascade.detectMultiScale(roi_gray)
# 	for (ex,ey,ew,eh) in eyes:
# 		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow("Faces found", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2
import sys

from PIL import Image, ImageFilter



face_cascade = cv2.CascadeClassifier('haarcascade.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#it might not be a bad idea to just adjust the histogram first...

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces = face_cascade.detectMultiScale(
	gray, 
	scaleFactor = 1.3,
	minNeighbors = 5,
	minSize = (30, 30),
	flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

# image = Image.open('your_image.png')
# image = image.filter(ImageFilter.FIND_EDGES)
# image.save('new_name.png') 

for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        eye = roi_color[ey:ey+eh, ex:ex+ew]
        cv2.imshow('eye', eye)


        big_eye = cv2.resize(eye,None,fx=1.15, fy=1.15, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('big_eye', big_eye)

        # edgeMap = big_eye.edges() 
        # edgeMap = edgeMap.smooth(aperture=(3,3)).threshold(50) # this will widen the edges
        # finalImg = big_eye.blit(smoothImg,mask = edgeMap) #copy the smoothed image only where edgemap is white
        # finalImg.show()

        #this pastes the eye back on the face. currently a 5px offset. may have to change later
        #to make sure the pasting is smooth, use either image pyramids (smooth blending). decided against grabcut b/c of eyebrows
        roi_color[ey-5:ey+big_eye.shape[1]-5, ex-5:ex+big_eye.shape[0]-5] = big_eye

    smoothed_face = cv2.bilateralFilter(roi_color, 5, 75, 75) #smooth the face. this filter saves edges but blurs other stuff
    roi_color[0:smoothed_face.shape[1], 0:smoothed_face.shape[0]] = smoothed_face #completely replace the face


cv2.imshow('img',img)
cv2.imshow('blurred', smoothed_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def createGaussianLayers(image):
#     #stuff here
#     G = image.copy()
#     layers = [G]
#     for i in xrange(6):
#         G = cv2.pyrDown(G)
#         layers.append(G)
#     return layers

# def createLaplacianLayers(gaussianLayers):
#     #stuff here
#     laplacianLayers = [gaussianLayers[5]]
#     for i in xrange(5,0,-1):
#         GE = cv2.pyrUp(gaussianLayers[i])
#         L = cv2.subtract(gaussianLayers[i-1],GE)
#         laplacianLayers.append(L)
#     return laplacianLayers