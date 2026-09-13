from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from dotenv import load_dotenv
from google import genai

import sqlite3
import secrets
import time
import os

from datetime import datetime


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("=" * 60)
print("GEMINI KEY CHECK")
print("Gemini API key loaded:", bool(GEMINI_API_KEY))
print("=" * 60)

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )
else:
    gemini_client = None


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

CORS(app)

bcrypt = Bcrypt(app)


# =========================================================
# GMAIL CONFIGURATION
# =========================================================

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)


# =========================================================
# DATABASE
# =========================================================

def init_db():

    conn = sqlite3.connect("cashew_ai.db")
    cursor = conn.cursor()

    # -----------------------------------------------------
    # USERS TABLE
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL DEFAULT 'User',
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # -----------------------------------------------------
    # ADD NAME COLUMN TO OLD DATABASE IF NEEDED
    # -----------------------------------------------------

    cursor.execute("PRAGMA table_info(users)")

    columns = [
        column[1]
        for column in cursor.fetchall()
    ]

    if "name" not in columns:

        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN name TEXT NOT NULL DEFAULT 'User'
        """)

    # -----------------------------------------------------
    # PASSWORD RESETS
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        )
    """)

    # -----------------------------------------------------
    # PREDICTION HISTORY
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_name TEXT,
            prediction TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return "Cashew AI Backend is running!"


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:

        return jsonify({
            "message": "Name, email and password are required"
        }), 400

    name = name.strip()
    email = email.strip().lower()

    if not name:

        return jsonify({
            "message": "Name is required"
        }), 400

    hashed_password = bcrypt.generate_password_hash(
        password
    ).decode("utf-8")

    try:

        conn = sqlite3.connect("cashew_ai.db")

        conn.execute(
            """
            INSERT INTO users
            (name, email, password)
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                hashed_password
            )
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

    except Exception as error:

        print("REGISTER ERROR:", repr(error))

        return jsonify({
            "message": "Registration failed."
        }), 500


# =========================================================
# LOGIN
# =========================================================

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
        """
        SELECT id, name, email, password
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    conn.close()

    if user is None:

        return jsonify({
            "message": "Invalid email or password"
        }), 401

    user_id, user_name, user_email, hashed_password = user

    if bcrypt.check_password_hash(
        hashed_password,
        password
    ):

        return jsonify({
            "message": "Login successful",

            "user": {
                "id": user_id,
                "name": user_name,
                "email": user_email
            }

        }), 200

    return jsonify({
        "message": "Invalid email or password"
    }), 401


# =========================================================
# FORGOT PASSWORD
# =========================================================

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

    token = secrets.token_urlsafe(32)

    expires_at = int(time.time()) + (15 * 60)

    cursor.execute(
        "DELETE FROM password_resets WHERE email = ?",
        (email,)
    )

    cursor.execute(
        """
        INSERT INTO password_resets
        (email, token, expires_at)
        VALUES (?, ?, ?)
        """,
        (
            email,
            token,
            expires_at
        )
    )

    conn.commit()
    conn.close()

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

If you did not request this email, you can ignore it.

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


# =========================================================
# RESET PASSWORD
# =========================================================

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

    hashed_password = bcrypt.generate_password_hash(
        new_password
    ).decode("utf-8")

    cursor.execute(
        """
        UPDATE users
        SET password = ?
        WHERE email = ?
        """,
        (
            hashed_password,
            email
        )
    )

    cursor.execute(
        "DELETE FROM password_resets WHERE token = ?",
        (token,)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "message": "Password reset successfully."
    }), 200


# =========================================================
# AI PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "message": "No image uploaded"
        }), 400

    image = request.files["image"]

    if image.filename == "":

        return jsonify({
            "message": "No image selected"
        }), 400

    allowed_extensions = {
        "jpg",
        "jpeg",
        "png",
        "webp"
    }

    filename = image.filename.lower()

    extension = (
        filename.rsplit(".", 1)[-1]
        if "." in filename
        else ""
    )

    if extension not in allowed_extensions:

        return jsonify({
            "message": "Only JPG, JPEG, PNG, and WEBP images are allowed"
        }), 400

    # -----------------------------------------------------
    # TEMPORARY PREDICTION
    # -----------------------------------------------------

    prediction = "Leaf Miner"
    category = "Pest"
    confidence = 94.6
    status = "detected"

    # -----------------------------------------------------
    # GET USER ID
    # -----------------------------------------------------

    user_id = request.form.get("user_id")

    try:

        user_id = int(user_id) if user_id else None

    except ValueError:

        user_id = None

    # -----------------------------------------------------
    # SAVE PREDICTION HISTORY
    # -----------------------------------------------------

    conn = sqlite3.connect("cashew_ai.db")

    conn.execute(
        """
        INSERT INTO prediction_history
        (
            user_id,
            image_name,
            prediction,
            category,
            confidence,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            image.filename,
            prediction,
            category,
            confidence,
            status,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()
    conn.close()

    return jsonify({
        "prediction": prediction,
        "category": category,
        "confidence": confidence,
        "status": status
    }), 200


# =========================================================
# GET PREDICTION HISTORY
# =========================================================

@app.route("/history/<int:user_id>", methods=["GET"])
def get_history(user_id):

    conn = sqlite3.connect("cashew_ai.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            image_name,
            prediction,
            category,
            confidence,
            status,
            created_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    history = []

    for row in rows:

        history.append({
            "id": row[0],
            "image_name": row[1],
            "prediction": row[2],
            "category": row[3],
            "confidence": row[4],
            "status": row[5],
            "created_at": row[6]
        })

    return jsonify({
        "history": history
    }), 200


# =========================================================
# AI CHATBOT - GEMINI
# =========================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    print("=" * 60)
    print("CHAT ENDPOINT CALLED")
    print("=" * 60)

    try:

        data = request.get_json()

        print("CHAT REQUEST DATA:")
        print(data)

        if not data:

            return jsonify({
                "message": "No request data received."
            }), 400

        message = data.get(
            "message",
            ""
        ).strip()

        diagnosis = data.get(
            "diagnosis",
            "Unknown"
        )

        confidence = data.get(
            "confidence",
            "Unknown"
        )

        print("User message:", message)
        print("Diagnosis:", diagnosis)
        print("Confidence:", confidence)

        # -------------------------------------------------
        # CHECK MESSAGE
        # -------------------------------------------------

        if not message:

            return jsonify({
                "message": "Please enter a question."
            }), 400

        # -------------------------------------------------
        # CHECK GEMINI KEY
        # -------------------------------------------------

        if not GEMINI_API_KEY or gemini_client is None:

            print("=" * 60)
            print("GEMINI API KEY IS MISSING")
            print("=" * 60)

            return jsonify({
                "message": "Gemini API key is not configured."
            }), 500

        # -------------------------------------------------
        # CREATE SHORT CHATBOT PROMPT
        # -------------------------------------------------

        prompt = f"""
You are Cashew AI Assistant, an agricultural AI assistant
specialized in cashew pest and disease information.

Current AI diagnosis: {diagnosis}
AI confidence: {confidence}%

User question:
{message}

Answer the user's question directly and naturally.

IMPORTANT RESPONSE RULES:

1. Keep the answer SHORT and easy to read.
2. Usually answer in 3 to 6 short paragraphs or bullet points.
3. Do not give a long article unless the user specifically asks
   for detailed information.
4. Start with the direct answer to the user's question.
5. Use simple language suitable for cashew farmers and students.
6. Use Markdown headings and bullet points when useful.
7. Do not use escaped Markdown such as \\*\\* or \\*.
8. Do not repeat the diagnosis and confidence unnecessarily because
   they are already displayed separately in the chatbot.
9. Do not repeat the user's question.
10. Do not include unnecessary introductions such as
    "Hello, I am Cashew AI Assistant."
11. Give only information relevant to the user's question.

You may explain:
- What the pest or disease is
- Symptoms
- Causes
- Spread
- Effects on the cashew plant
- Prevention
- Cultural management
- Monitoring

Do not claim that the AI prediction is a confirmed laboratory diagnosis.

If the confidence is low, mention that the prediction may need confirmation.

Do not provide unsafe pesticide instructions, specific chemical doses,
concentrations, or application schedules.

For chemical control, advise the user to follow the approved product
label and consult qualified agricultural experts or local agricultural
authorities.

Keep the answer concise and practical.
"""

        # -------------------------------------------------
        # SEND REQUEST TO GEMINI
        # -------------------------------------------------

        print("=" * 60)
        print("SENDING REQUEST TO GEMINI...")
        print("=" * 60)

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        # -------------------------------------------------
        # GET RESPONSE
        # -------------------------------------------------

        reply = response.text

        print("=" * 60)
        print("GEMINI RESPONSE RECEIVED")
        print(reply)
        print("=" * 60)

        if not reply:

            print("GEMINI RETURNED EMPTY RESPONSE")

            return jsonify({
                "message": "The AI assistant returned an empty response."
            }), 500

        return jsonify({
            "reply": reply
        }), 200

    except Exception as error:

        print("=" * 60)
        print("GEMINI CHATBOT ERROR:")
        print("ERROR TYPE:", type(error).__name__)
        print("ERROR:", repr(error))
        print("=" * 60)

        return jsonify({
            "message": "Unable to connect to the AI assistant."
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CASHEW AI BACKEND STARTING")
    print("http://localhost:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )