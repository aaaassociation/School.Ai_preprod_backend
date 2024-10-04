from flask import Flask
from flask_cors import CORS
from firebase_admin import credentials, firestore
import firebase_admin

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*")

cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

from app.api.auth import *
from app.api.teacher import *
from app.api.exam import *