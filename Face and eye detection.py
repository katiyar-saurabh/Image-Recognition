import cv2               #Importing libraries
import numpy as np

# Defining classifiers for what to be detected
face_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_eye.xml')

# Image acquisition and grayscale conversion
image = cv2.imread('obama.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Determining Face in the image
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if faces is ():
    print("no face found")

#  Enclosing face in a frame
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (127, 255, 0), 2)
    cv2.imshow('face', image)
    cv2.waitKey(0)
                                        # cropping the image and acquiring only face from image and grayscale conversion
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
                                                     # Determining eyes from the cropped image of face
    eyes = eye_classifier.detectMultiScale(roi_gray)


    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color , (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        cv2.imshow('eyes', image)
        cv2.waitKey(0)
cv2.destroyAllWindows()