import cv2
import face_recognition
import numpy as np
import sqlite3

db_path = 'database/face_encodings.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name, encoding FROM faces")
rows = cursor.fetchall()
conn.close()

known_encodings = []
known_names = []

for name, enc_bytes in rows:
    enc_np = np.frombuffer(enc_bytes, dtype=np.float64)
    known_encodings.append(enc_np)
    known_names.append(name)


scale = 0.25
box_multiplier = 1 / scale
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encodeFace)
        min_distance = np.min(distances) if distances.size > 0 else None

        if min_distance is not None and min_distance < 0.6:
            match_index = np.argmin(distances)
            name = known_names[match_index].upper()
        else:
            name = 'Unknown'

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier), int(x1 * box_multiplier)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Capturing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
