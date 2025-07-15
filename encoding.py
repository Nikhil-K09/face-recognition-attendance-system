import cv2
import os
import face_recognition
import numpy as np

path = 'faces'

images=[]
classNames=[]

for img in os.listdir(path):
    image=cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])
# print(classNames)

def findecoding(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        enc=face_recognition.face_encodings(img)[0]
        encodeList.append(enc)
        # print(enc)
    return encodeList

knownEncodings=findecoding(images)

#print("encoding done")


scale=0.25
box_multiplier=1/scale

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    Current_image = cv2.resize(img,(0,0),None,scale,scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(Current_image, model='hog')
    face_encodes = face_recognition.face_encodings(Current_image, face_locations)


    for encodeFace, faceLocation in zip(face_encodes, face_locations):
        matches=face_recognition.compare_faces(knownEncodings, encodeFace, tolerance=0.6)
        faceDis=face_recognition.face_distance(knownEncodings, encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown'

        y1,x2,y2,x1=faceLocation
        y1,x2,y2,x1=int(y1*box_multiplier),int(x2*box_multiplier),int(y2*box_multiplier),int(x1*box_multiplier)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-20),(x2,y2),(255,255,255),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),2)

    cv2.imshow("Capturing", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()