import cv2
import numpy as np             # Importing libraries

# Defining classifiers for thing to be detected
face_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:/Users/Acer/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_eye.xml')

#2 Defining a function  to detect Face and Eyes
def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3,5 )    # Detecting face in the passed grayscale image
    for(x, y, w, h ) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (127,255,0),2)   # Framing of face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray,1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)   # Framing of eyes
    return frame

cap = cv2.VideoCapture(0)    # Start the camera

while True:
    ret, frame = cap.read()                          # capture the frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # frame to grayscale
    output = detect(gray,frame)                      # calling the function
    cv2.imshow(' face and eyes', output)
    if cv2.waitKey(1) & 0xff == ord('q'):            # to exit
        break
cap.release()
cv2.destroyAllWindows()