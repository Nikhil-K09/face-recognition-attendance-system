import cv2

name=input("Enter Name : ")
cap=cv2.VideoCapture(0)

while True:
    success,frame =cap.read()
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) == ord('c'):
        filename= 'faces/'+name+'.jpg'
        cv2.imwrite(filename,frame)
        print("Image saved-", filename)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


