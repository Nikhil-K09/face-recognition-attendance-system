import cv2
import os
import face_recognition
import numpy as np
import sqlite3

os.makedirs('database', exist_ok=True)
db_path = 'database/face_encodings.db'
path = 'faces'

images = []
classNames = []

for img_name in os.listdir(path):
    img = cv2.imread(os.path.join(path, img_name))
    if img is None:
        continue
    images.append(img)
    classNames.append(os.path.splitext(img_name)[0])

def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:  
            encode_list.append(encodings[0])
    return encode_list


conn = sqlite3.connect(db_path)
cursor = conn.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        name TEXT PRIMARY KEY,
        encoding BLOB
    )
''')

known_encodings = find_encodings(images)

for name, encoding in zip(classNames, known_encodings):
    encoding_bytes = encoding.tobytes()
    cursor.execute("REPLACE INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_bytes))

conn.commit()
conn.close()
print("[INFO] Encodings stored in face_encodings.db")
