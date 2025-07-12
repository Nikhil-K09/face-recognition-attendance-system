import cv2
import matplotlib.pyplot as plt

image_path='input/input2.jpeg'
img=cv2.imread(image_path)

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

classifier=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

face=classifier.detectMultiScale(grayimg,scaleFactor=1.1,minSize=(40,40),minNeighbors=5)

for(x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

video_capture=cv2.VideoCapture(0)

'''
# For image
plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(img_rgb)
plt.show()
'''
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