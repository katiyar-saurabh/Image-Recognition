# Importing libraries
import cv2
import time
import numpy as np


# Defining classifiers for the object to be detected
car_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_car.xml')

cap = cv2.VideoCapture('Traffic2.mp4')  # Acquiring video


while cap.isOpened():
    time.sleep(0.05)
    ret, frame = cap.read()    # read video frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Converting frame to grayscale

    cars = car_classifier.detectMultiScale(gray, 1.1, 2)   # Detection of car if any

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (127, 255, 0), 2)  # Framing of object
        cv2.imshow("car", frame)

    if cv2.waitKey(1) == 13:    # to exit
        break
cap.release()
cv2.destroyAllWindows()