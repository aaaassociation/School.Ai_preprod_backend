from app import app
from flask import jsonify

if __name__ == "__main__":
  app.run(threaded=True)