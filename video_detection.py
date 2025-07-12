import cv2
import matplotlib.pyplot as plt

video_capture=cv2.VideoCapture(0)

classifier=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

def dectect_box(vid):
    grayimg=cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(grayimg,scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
    for(x,y,w,h) in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)
    return faces

while(True):
    result,video_frame=video_capture.read()
    if result is False:
        break
    faces=dectect_box(video_frame)
    cv2.imshow("Live",video_frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()   