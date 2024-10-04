from app import app, db
from flask import request, jsonify
from firebase_admin import auth as firebase_auth
from datetime import datetime, timedelta
from app.config import JWT_SECRET_KEY

import jwt
import bcrypt
import firebase_admin

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')
    age = data.get('age')
    gender = data.get('gender')
    location = data.get('location')
    interests = data.get('interests')
    education = data.get('education')

    try:
        user = firebase_auth.create_user(
            email=email,
            password=password
        )

        db.collection('user_data').document(user.uid).set({
            'email': email,
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'location': location,
            'interests': interests,
            'education': education
        })

        return jsonify({"message": "User signed up successfully"}), 201

    except Exception as e:
        return jsonify({"message": str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    try:
        user = firebase_auth.get_user_by_email(email)

        # Verify the password
        user_record = db.collection('user_data').document(user.uid).get()
        if not user_record.exists:
            return jsonify({"message": "Invalid credentials"}), 401

        user_data = user_record.to_dict()
        if not bcrypt.checkpw(password.encode('utf-8'), user_data['password'].encode('utf-8')):
            return jsonify({"message": "Invalid credentials"}), 401

        token = jwt.encode({'uid': user.uid, 'exp': datetime.utcnow() + timedelta(hours=24)}, JWT_SECRET_KEY)

        return jsonify({"message": "Login successful", "token": token}), 200

    except firebase_admin.auth.UserNotFoundError:
        return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/protected', methods=['GET'])
def protected():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"message": "Missing authorization header"}), 401

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        return jsonify({"message": f"Hello user {user_id}!"}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"message": "Invalid token"}), 401
