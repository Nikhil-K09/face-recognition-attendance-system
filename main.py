import tkinter as tk
from tkinter import messagebox
import subprocess
import webbrowser
import threading
import os

LOCALHOST_URL = "http://127.0.0.1:5000"

VENV_PYTHON_PATH = r"D:/Workspace/Develop/Projects/face-recognition/.venv/Scripts/python.exe"
def run_script(script_name):
    try:
        subprocess.Popen([VENV_PYTHON_PATH, script_name])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {script_name}\n{str(e)}")

def open_app():
    def launch_app():
        try:
            subprocess.Popen([VENV_PYTHON_PATH, 'app.py'])
            threading.Timer(2.0, lambda: webbrowser.open(LOCALHOST_URL)).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch web app\n{str(e)}")

    threading.Thread(target=launch_app, daemon=True).start()

root = tk.Tk()
root.title("Face Recognition Control Panel")
root.geometry("400x400")
root.configure(bg="#1e1e1e")

label = tk.Label(root, text="Face Recognition System", font=("Arial", 16, "bold"), fg="yellow", bg="#1e1e1e")
label.pack(pady=20)

def create_button(text, command):
    return tk.Button(root, text=text, width=25, height=2, bg="#444", fg="white", font=("Arial", 12), command=command)

create_button("➕ Add Face", lambda: run_script("add_face.py")).pack(pady=5)
create_button("⚙️ Encode Faces", lambda: run_script("encoding.py")).pack(pady=5)
create_button("🎥 Start Recognition", lambda: run_script("recognition.py")).pack(pady=5)
create_button("🌐 Launch Web App", open_app).pack(pady=5)

root.mainloop()
