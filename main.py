import face_recognition
import cv2

image_path1='input/Kamal Haasan.jpg'
img1=cv2.imread(image_path1)
rgb_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
enc1=face_recognition.face_encodings(rgb_img1)[0]

image_path2='input/Darren Watkins.jpeg'
img2=cv2.imread(image_path2)
rgb_img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
enc2=face_recognition.face_encodings(rgb_img2)[0]

result=face_recognition.compare_faces([enc1],enc2)
