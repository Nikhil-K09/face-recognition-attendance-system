from flask import Flask, render_template, request
import sqlite3
import os

app = Flask(__name__)
DB_PATH = os.path.join("database", "logs.db")

@app.route("/", methods=["GET", "POST"])
def index():
    logs = []
    selected_date = ""

    if request.method == "POST":
        selected_date = request.form["date"]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, time FROM logs WHERE date = ?", (selected_date,))
        logs = cursor.fetchall()
        conn.close()

    return render_template("index.html", logs=logs, selected_date=selected_date)

if __name__ == "__main__":
    app.run(debug=True)
