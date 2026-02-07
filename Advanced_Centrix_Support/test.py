from flask import Flask, request, jsonify
import sqlite3
app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/index", methods=["POST"])
def login():
    data = request.get_json()
    user_id = data.get("user_id")
    email = data.get("email")
    password = data.get("pass")
    if not email or not password:
        return jsonify({"Email and password required"}),
    if data in get_db_connection.conn:
        print("acsess")
    else:
        print("Denied")
    return jsonify({
        "message": "Login successful",
        "user_id": user_id,
        "users.db":data
    }),

if __name__ == "__main__":
    app.run(debug=True)