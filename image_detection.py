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

# For image
plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(img_rgb)
plt.show()