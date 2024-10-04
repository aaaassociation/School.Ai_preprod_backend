from dotenv import load_dotenv
import os

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOEY_API_KEY = os.getenv('GOOEY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
STABILITY_KEY = os.getenv('STABILITY_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')