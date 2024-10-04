import requests
import json
import time
import uuid
import os
import httpx

from concurrent.futures import ThreadPoolExecutor
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs, ApiError
from ratelimit import limits, sleep_and_retry
from app import db
from app.config import SERPAPI_API_KEY, ELEVENLABS_API_KEY, STABILITY_KEY

MAX_WORKERS = 10

http_client = httpx.Client(timeout=httpx.Timeout(3600.0, connect=60.0))
client = ElevenLabs(api_key=ELEVENLABS_API_KEY, httpx_client=http_client)

def send_request(endpoint, headers, payload):
    for _ in range(5):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response:
                try:
                    print("Response Content:", response.json())
                except Exception:
                    print("Response Content (non-JSON):", response.text)
            time.sleep(5)
    raise requests.exceptions.RequestException("Failed to connect after multiple attempts.")

def send_request_with_rate_limit(endpoint, headers, payload):
    while True:
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()

            limit = int(response.headers.get('x-ratelimit-limit', 100))
            remaining = int(response.headers.get('x-ratelimit-remaining', 100))
            reset_time = int(response.headers.get('x-ratelimit-reset', 1))
            adjust_thread_pool(remaining, reset_time)
            
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                reset_time = int(response.headers.get('Retry-After', 60))
                time.sleep(reset_time)
            else:
                print(f"HTTP error occurred: {e}. Retrying...")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2)

def adjust_thread_pool(remaining, reset_time):
    global MAX_WORKERS
    if remaining < 10:
        MAX_WORKERS = max(1, MAX_WORKERS // 2)
        time.sleep(reset_time + 1)
    else:
        MAX_WORKERS = min(20, MAX_WORKERS + 1)

def search_image(query):
    """
    Searches for an image using the SERP API with a given query.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            params = {
                "engine": "google_images",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": 1
            }
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code == 200:
                images = response.json().get('images_results', [])
                if images:
                    return images[0].get('original', None)
                else:
                    print("No images found in the response.")
            else:
                print(f"Error: Received response with status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Request Error on attempt {attempt + 1}: {e}")
        
        time.sleep(2)
    
    return None

def generate_voice(chapter_name: str, subchapter_name: str, text: str, voice_id: str) -> str:
    unique_id = uuid.uuid4()
    output_file_path = os.path.join('static', 'audio', f"{unique_id}.mp3")
    output_format="mp3_22050_32"
    output_file_path = os.path.abspath(output_file_path)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    db.collection('audio').add({
        'chapter_name': chapter_name,
        'subchapter_name': subchapter_name,
        'prompt': text,
        'uuid': str(unique_id)
    })

    return text_to_speech_file(text, voice_id, output_file_path)

def text_to_speech_file(text: str, voice_id: str, file_path: str) -> str:
    max_chars = 2500
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    with open(file_path, "wb") as f:
        for chunk in chunks:
            retry_attempts = 3
            for attempt in range(retry_attempts):
                try:
                    response = client.text_to_speech.convert(
                        voice_id=voice_id,
                        optimize_streaming_latency="0",
                        output_format="mp3_22050_32",
                        text=chunk,
                        model_id="eleven_turbo_v2",
                        voice_settings=VoiceSettings(
                            stability=0.0,
                            similarity_boost=1.0,
                            style=0.0,
                            use_speaker_boost=True,
                        ),
                    )
                    for data_chunk in response:
                        if data_chunk:
                            f.write(data_chunk)
                    break
                except ApiError as e:
                    print(f"API Error: {e.body}, Status Code: {e.status_code}")
                    raise
                except Exception as e:
                    print(f"Exception on attempt {attempt + 1} for chunk: {e}")
                    time.sleep(5)
                    if attempt == retry_attempts - 1:
                        raise
    return file_path

def send_generation_request(host, params):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response