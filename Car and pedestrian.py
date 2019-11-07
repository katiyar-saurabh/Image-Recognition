# Importing libraries
import cv2
import time
import numpy as np

# Defining classifiers for the object to be detected
car_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('NewYorkCity.mp4')    # Acquisition of video in which cars and pedestrians to be detected


while cap.isOpened():
    time.sleep(0.05)
    ret, frame = cap.read()                          # read video frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Grayscale conversion

    cars = car_classifier.detectMultiScale(gray, 1.1, 2)     # Detection of cars if any present in frame
    body = body_classifier.detectMultiScale(gray, 1.1, 2)    # Detection of pedestrians if any present in frame
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (127, 255, 0), 2)

    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow("body", frame)

    if cv2.waitKey(1) == 13:     # to exit
        break

cap.release()
cv2.destroyAllWindows()