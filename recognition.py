import cv2
import face_recognition
import numpy as np
import sqlite3
import os
from datetime import datetime


encoding_db_path = 'database/face_encodings.db'
log_db_path = 'database/logs.db'
os.makedirs('database', exist_ok=True)


conn = sqlite3.connect(encoding_db_path)
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


log_conn = sqlite3.connect(log_db_path)
log_cursor = log_conn.cursor()

log_cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT
    )
''')
log_conn.commit()


def log_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M")

    
    log_cursor.execute("SELECT * FROM logs WHERE name = ? AND date = ?", (name, today))
    if log_cursor.fetchone() is None:
        log_cursor.execute("INSERT INTO logs (name, date, time) VALUES (?, ?, ?)", (name, today, time_now))
        log_conn.commit()
        print(f"[LOGGED] {name} at {time_now} on {today}")
    else:
        print(f"[SKIPPED] Already logged {name} for today")


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
            log_attendance(name)
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
log_conn.close()
cv2.destroyAllWindows()
