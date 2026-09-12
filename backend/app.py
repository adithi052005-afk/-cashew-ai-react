from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from dotenv import load_dotenv
import sqlite3
import secrets
import time
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

bcrypt = Bcrypt(app)

# ---------------- GMAIL CONFIGURATION ----------------

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)


# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect("cashew_ai.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------- HOME ----------------

@app.route("/")
def home():
    return "Cashew AI Backend is running!"


# ---------------- REGISTER ----------------

@app.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({
            "message": "Email and password are required"
        }), 400

    email = email.strip().lower()

    hashed_password = bcrypt.generate_password_hash(
        password
    ).decode("utf-8")

    try:

        conn = sqlite3.connect("cashew_ai.db")

        conn.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hashed_password)
        )

        conn.commit()
        conn.close()

        return jsonify({
            "message": "User registered successfully"
        }), 201

    except sqlite3.IntegrityError:

        return jsonify({
            "message": "Email already registered"
        }), 409


# ---------------- LOGIN ----------------

@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({
            "message": "Email and password are required"
        }), 400

    email = email.strip().lower()

    conn = sqlite3.connect("cashew_ai.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, email, password FROM users WHERE email = ?",
        (email,)
    )

    user = cursor.fetchone()

    conn.close()

    if user is None:

        return jsonify({
            "message": "Invalid email or password"
        }), 401

    user_id, user_email, hashed_password = user

    if bcrypt.check_password_hash(
        hashed_password,
        password
    ):

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user_email
            }
        }), 200

    return jsonify({
        "message": "Invalid email or password"
    }), 401


# ---------------- FORGOT PASSWORD ----------------

@app.route("/forgot-password", methods=["POST"])
def forgot_password():

    data = request.get_json()

    email = data.get("email")

    if not email:

        return jsonify({
            "message": "Email is required"
        }), 400

    email = email.strip().lower()

    conn = sqlite3.connect("cashew_ai.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,)
    )

    user = cursor.fetchone()

    if user is None:

        conn.close()

        return jsonify({
            "message": "No account found with this email."
        }), 404

    # Generate secure reset token
    token = secrets.token_urlsafe(32)

    # Token expires after 15 minutes
    expires_at = int(time.time()) + (15 * 60)

    # Delete old reset tokens for this email
    cursor.execute(
        "DELETE FROM password_resets WHERE email = ?",
        (email,)
    )

    # Store new token
    cursor.execute(
        """
        INSERT INTO password_resets
        (email, token, expires_at)
        VALUES (?, ?, ?)
        """,
        (email, token, expires_at)
    )

    conn.commit()
    conn.close()

    # ---------------- RESET LINK ----------------
    # React is running on port 5173
    reset_link = (
        "http://192.168.0.107:5173/reset-password?token="
        + token
    )

    try:

        print("=" * 60)
        print("Attempting to send reset email to:", email)

        msg = Message(
            subject="Cashew AI - Password Reset",
            sender=app.config["MAIL_USERNAME"],
            recipients=[email]
        )

        msg.body = f"""
Hello,

We received a request to reset your Cashew AI password.

Click the link below to reset your password:

{reset_link}

This link will expire in 15 minutes.

If you did not request a password reset, you can ignore this email.

Regards,
Cashew AI Team
"""

        mail.send(msg)

        print("Reset email sent successfully to:", email)
        print("Reset link:", reset_link)
        print("=" * 60)

        return jsonify({
            "message": "Password reset link has been sent to your email."
        }), 200

    except Exception as error:

        print("=" * 60)
        print("EMAIL ERROR:")
        print(repr(error))
        print("=" * 60)

        return jsonify({
            "message": "Unable to send reset email. Please try again later."
        }), 500


# ---------------- RESET PASSWORD ----------------

@app.route("/reset-password", methods=["POST"])
def reset_password():

    data = request.get_json()

    token = data.get("token")
    new_password = data.get("password")

    if not token or not new_password:

        return jsonify({
            "message": "Token and new password are required"
        }), 400

    conn = sqlite3.connect("cashew_ai.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT email, expires_at
        FROM password_resets
        WHERE token = ?
        """,
        (token,)
    )

    reset_data = cursor.fetchone()

    if reset_data is None:

        conn.close()

        return jsonify({
            "message": "Invalid or expired reset link."
        }), 400

    email, expires_at = reset_data

    # Check expiration
    if int(time.time()) > expires_at:

        cursor.execute(
            "DELETE FROM password_resets WHERE token = ?",
            (token,)
        )

        conn.commit()
        conn.close()

        return jsonify({
            "message": "Reset link has expired."
        }), 400

    # Hash new password
    hashed_password = bcrypt.generate_password_hash(
        new_password
    ).decode("utf-8")

    # Update password
    cursor.execute(
        """
        UPDATE users
        SET password = ?
        WHERE email = ?
        """,
        (hashed_password, email)
    )

    # Delete used token
    cursor.execute(
        "DELETE FROM password_resets WHERE token = ?",
        (token,)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "message": "Password reset successfully."
    }), 200


# ---------------- START SERVER ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )