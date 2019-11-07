# Importing libraries
import cv2
import numpy as np

# Defining classifiers for the object to be detected
body_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('NewYorkCity.mp4')  # Acquisition of video in which pedestrians to be detected

while cap.isOpened():
    ret, frame = cap.read()                            # Reading frames from video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = body_classifier.detectMultiScale(gray, 1.1, 2)   # Detecting bodies of pedestrians

    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)       # Framing the detected object
        cv2.imshow("body", frame)

    if cv2.waitKey(1) == 13:      # to exit
        break
cap.release()
cv2.destroyAllWindows()