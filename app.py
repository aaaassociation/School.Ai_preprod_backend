import os
import requests
import json
import time
import openai
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import bcrypt
import base64
from html import escape
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs, ApiError
import logging
from logging.handlers import RotatingFileHandler
import httpx
import uuid
from concurrent.futures import ThreadPoolExecutor
import jwt
from dotenv import load_dotenv
from pydantic import BaseModel, Field

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Firebase setup
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def create_jwt_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")

load_dotenv()
load_dotenv(dotenv_path='./.env')

# Environment Variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GOOEY_API_KEY = os.getenv('GOOEY_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
STABILITY_KEY = os.getenv('STABILITY_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

if not STABILITY_KEY:
    raise ValueError("API key is required. Please enter a valid API key.")

http_client = httpx.Client(timeout=httpx.Timeout(3600.0, connect=60.0))
client = ElevenLabs(api_key=ELEVENLABS_API_KEY, httpx_client=http_client)

# Set up logging
if not app.debug:
    handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)

# @app.errorhandler(500)
# def internal_error(error):
#     app.logger.error(f"Server Error: {error}, route: {request.url}")
#     return jsonify({"error": "Internal server error"}), 500

# @app.errorhandler(Exception)
# def unhandled_exception(e):
#     app.logger.error(f"Unhandled Exception: {e}, route: {request.url}")
#     return jsonify({"error": "Unhandled exception"}), 500

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
                    app.logger.error(f"API Error: {e.body}, Status Code: {e.status_code}")
                    raise
                except Exception as e:
                    app.logger.error(f"Exception on attempt {attempt + 1} for chunk: {e}")
                    time.sleep(5)
                    if attempt == retry_attempts - 1:
                        raise
    return file_path

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}, route: {request.url}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error(f"Unhandled Exception: {e}, route: {request.url}")
    return jsonify({"error": "Unhandled exception"}), 500

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

@app.route('/save-user-input', methods=['POST'])
def save_user_input():
    data = request.json
    user_id = data.get("user_id")
    input_data = data.get("input_data")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.collection('course_data').add({
        'user_id': user_id,
        'input_data': input_data,
        'date': date
    })

    return jsonify({"message": "User input saved successfully"}), 201

@app.route('/save-course-data', methods=['POST'])
def save_course_data():
    data = request.json
    user_id = data.get("user_id")
    course_outline = data.get("course_outline")
    course_content = data.get("course_content")
    input_data = data.get("input_data")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.collection('course_data').add({
        'user_id': user_id,
        'course_outline': json.dumps(course_outline),
        'course_content': json.dumps(course_content),
        'input_data': input_data,
        'date': date
    })

    return jsonify({"message": "Course data saved successfully"}), 201

@app.route('/get-course-data/<user_id>', methods=['GET'])
def get_course_data(user_id):
    courses_ref = db.collection('course_data').where('user_id', '==', user_id).stream()
    course_data = [
        {
            "input_data": course.get('input_data'),
            "input_date": course.get('date'),
            "course_outline": json.loads(course.get('course_outline')) if course.get('course_outline') else None,
            "course_content": json.loads(course.get('course_content')) if course.get('course_content') else None,
        }
        for course in courses_ref
    ]
    return jsonify(course_data), 200

@app.route('/get-specific-course-data', methods=['POST'])
def get_specific_course_data():
    data = request.json
    user_id = data.get("user_id")
    input_data = data.get("input_data")

    courses_ref = db.collection('course_data').where('user_id', '==', user_id).where('input_data', '==', input_data).stream()
    course_data = next(courses_ref, None)

    if course_data:
        return jsonify({
            "course_outline": json.loads(course_data.get('course_outline')),
            "course_content": json.loads(course_data.get('course_content'))
        }), 200
    return jsonify({"error": "Course data not found"}), 404

def send_request(endpoint, headers, payload):
    for _ in range(5):  # Retry up to 5 times
        try:
            response = requests.post(endpoint, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Request Error:", e)
            print("Request Payload:", json.dumps(payload, indent=2))
            print("Response Status Code:", response.status_code if response else "No response")
            if response:
                try:
                    print("Response Content:", response.json())
                except Exception:
                    print("Response Content (non-JSON):", response.text)
            time.sleep(5)
    raise requests.exceptions.RequestException("Failed to connect after multiple attempts.")

def search_image(query):
    """
    Searches for an image using the SERP API with a given query.
    """
    max_attempts = 3  # Retry up to 3 times
    for attempt in range(max_attempts):
        try:
            params = {
                "engine": "google_images",  # Correct engine name for image search
                "q": query,
                "api_key": serpapi_api_key,
                "num": 1  # Fetch only one image to minimize data and response time
            }
            response = requests.get("https://serpapi.com/search", params=params)
            
            # Check if the response status is 200 (OK)
            if response.status_code == 200:
                images = response.json().get('images_results', [])
                if images:
                    # Return the first image URL if available
                    return images[0].get('original', None)
                else:
                    print("No images found in the response.")
            else:
                print(f"Error: Received response with status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Request Error on attempt {attempt + 1}: {e}")
        
        # Wait before retrying
        time.sleep(2)
    
    # Return None if no image was found after max_attempts
    return None

def send_generation_request(host, params):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
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

@app.route('/generate-chapters', methods=['POST'])
def generate_chapters():
    data = request.json
    prompt = data['prompt']
    num_chapters = data.get('chapters', 6)  # Default to 6 if not provided
    num_subchapters = data.get('chapterDepth', 3)  # Default to 3 if not provided
    
    prompt_message = f"""
    Generate a comprehensive list of chapters and subchapters for a detailed course on {prompt}.
    The number of chapters is {num_chapters}, and each chapter has {num_subchapters} subchapters.
    Please adhere strictly to the following format:
    {{
        "Chapter 1: Chapter Title": ["Subchapter Title 1", "Subchapter Title 2", ..., "Subchapter Title {num_subchapters}"],
        "Chapter 2: Chapter Title": ["Subchapter Title 1", "Subchapter Title 2", ..., "Subchapter Title {num_subchapters}"],
        ...
    }}
    Do not include any explanations, introductions, or any other extraneous information. 
    Only provide the structured JSON format as specified. 
    Do not include any prefixes like 'Subchapter x.x' in the subchapter titles. 
    Make sure all subchapters are provided as a list of strings.
    """
    request_payload = [
        {"role": "system", "content": "You are an expert course designer."},
        {"role": "user", "content": prompt_message},
    ]
    payload = {
        "model": "gpt-4",
        "messages": request_payload,
        "max_tokens": 2000
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']
    print("Chapters Response:", gpt_response)  # Debug response
    
    try:
        json_response = json.loads(gpt_response)
        if isinstance(json_response, dict):
            for chapter, subchapters in json_response.items():
                if not isinstance(subchapters, list):
                    raise ValueError(f"Subchapters for {chapter} are not in a list format")
        else:
            raise ValueError("The response is not a dictionary")
        
        for chapter, subchapters in json_response.items():
            cleaned_subchapters = [subchapter.split(": ", 1)[-1] for subchapter in subchapters]
            json_response[chapter] = cleaned_subchapters

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to decode JSON response: {e}")
        return jsonify({"error": "Failed to decode JSON response from OpenAI"}), 500

    sorted_chapters = dict(sorted(json_response.items(), key=lambda item: int(item[0].split(' ')[1].strip(':'))))

    return jsonify(sorted_chapters)

@app.route('/generate-content', methods=['POST'])
def generate_content():
    data = request.json
    chapter_name = data['chapter_name']
    subchapter_name = data['subchapter_name']
    prompt = data['prompt']

    def fetch_content(chapter_name, subchapter_name):
        prompt_message = f"""
        Generate the content for a subchapter in a course. The chapter title is {chapter_name}. The title of the subchapter is {subchapter_name}. The course is about {prompt}.
        The content should be formatted with below delimiters, and no other text or explanations should be included.
        1. Concepts marked with <<Concept>>.
        2. Titles marked with <<Title>>.
        3. Subheadings marked with <<Subheading>>.
        4. Emphasis marked with <<Emphasis>>.
        5. Code sections marked with <<Code>>.
        If possible, suggest relevant images in the format [IMAGE: Image description or keyword].
        """

        request_payload = [
            {"role": "system", "content": "You are an expert content generator."},
            {"role": "user", "content": prompt_message},
        ]
        payload = {
            "model": "gpt-4",
            "messages": request_payload,
            "max_tokens": 4000
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        endpoint = "https://api.openai.com/v1/chat/completions"
        response = send_request(endpoint, headers, payload)
        gpt_response = response['choices'][0]['message']['content']
        return gpt_response

    with ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_content, chapter_name, subchapter_name)
        gpt_response = future.result()

    # Split content based on [IMAGE: ...] markers
    content_parts = gpt_response.split('[IMAGE:')
    final_content = content_parts[0]

    # Process each part that contains an image prompt
    for part in content_parts[1:]:
        try:
            image_prompt, rest_of_content = part.split(']', 1)
            image_url = search_image(image_prompt.strip())
            if image_url:
                # Include the image URL in the formatted content
                final_content += f'<<Image:URL>> {image_url}\n' + rest_of_content
            else:
                # If image search fails, include the prompt as is
                final_content += f'[IMAGE: {image_prompt.strip()}]' + rest_of_content
        except ValueError:
            # Handles case where split fails, appending the original part
            final_content += f'[IMAGE: {part.strip()}]'
    print(final_content)
    return jsonify(final_content)

@app.route('/dig-deeper', methods=['POST'])
def dig_deeper():
    data = request.json
    chapter_name = data['chapter_name']
    subchapter_name = data['subchapter_name']
    prompt = data['prompt']

    prompt_message = f"""
    Provide detailed content for the subchapter '{subchapter_name}' in the chapter '{chapter_name}' of the course on '{prompt}'. 
    Include detailed explanations, examples, case studies, step-by-step guides, and suggestions for images.
    Format the content in HTML and include image suggestions in the format [IMAGE: ...].
    """

    request_payload = [
        {"role": "system", "content": "You are an expert course content generator."},
        {"role": "user", "content": prompt_message},
    ]

    payload = {
        "model": "gpt-4",
        "messages": request_payload,
        "max_tokens": 4000
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']
    print("Detailed Content Response:", gpt_response)

    content_parts = gpt_response.split('[IMAGE:')
    final_content = content_parts[0]
    for part in content_parts[1:]:
        image_prompt, rest_of_content = part.split(']', 1)
        image_url = search_image(image_prompt.strip())
        if image_url:
            final_content += f'<img src="{image_url}" alt="{image_prompt.strip()}"/>' + rest_of_content
        else:
            final_content += f'[IMAGE: {image_prompt.strip()}]' + rest_of_content

    return jsonify(final_content)

@app.route('/generate-explanation', methods=['POST'])
def generate_explanation():
    data = request.json
    prompt = f"Explain the course content for the chapter '{data['chapter_name']}' and subchapter '{data['subchapter_name']}'. The course is about {data['prompt']}."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert educator."},
            {"role": "user", "content": prompt},
        ]
    )
    explanation_text = response['choices'][0]['message']['content']
    audio_content = generate_voice(explanation_text, data['voice_id'])
    
    return send_file(audio_content, as_attachment=True, mimetype='audio/mpeg')

@app.route('/ask-question', methods=['POST'])
def ask_question():
    data = request.json
    prompt = data['prompt']
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in answering questions related to the course content."},
            {"role": "user", "content": prompt},
        ]
    )
    answer_text = response['choices'][0]['message']['content']
    audio_content = generate_voice(answer_text, data['voice_id'])
    
    return send_file(audio_content, as_attachment=True, mimetype='audio/mpeg')

@app.route('/generate-final-exam', methods=['POST'])
def generate_final_exam():
    data = request.json
    prompt = data['prompt']
    
    prompt_message = f"""
    Generate a final exam for the course on '{prompt}'.
    Include three types of questions for each chapter:
    1. Selection problems (multiple-choice) - 3 questions
    2. Fill-in-the-blank problems - 3 questions
    3. Entry problems (short answer) - 3 questions

    Format the response as a JSON array with the following structure:
    [
        {{
            "type": "selection",
            "question": "question text",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "option1"
        }},
        {{
            "type": "fill-in-the-blank",
            "question": "question text with __blank__",
            "correct_answer": "answer"
        }},
        {{
            "type": "entry",
            "question": "question text",
            "correct_answer": "answer"
        }}
    ]
    """
    
    request_payload = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_message},
    ]
    
    payload = {
        "model": "gpt-4",
        "messages": request_payload,
        "max_tokens": 2000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    
    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']
    print("Final Exam Questions Response:", gpt_response)  # Debug response
    
    try:
        json_response = json.loads(gpt_response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        return jsonify({"error": "Failed to decode JSON response from OpenAI"}), 500

    return jsonify(json_response)

@app.route('/evaluate-final-exam', methods=['POST'])
def evaluate_final_exam():
    data = request.json
    questions = data['questions']
    user_answers = data['answers']

    correct_answers = {q['question']: q['correct_answer'] for q in questions}
    results = {q['question']: (user_answers[q['question']] == q['correct_answer']) for q in questions}
    score = sum(results.values())
    total_questions = len(questions)
    score_5_point = (score / total_questions) * 5

    explanations = {}
    for question in questions:
        explanation_prompt = f"Explain the correct answer for the following question:\nQuestion: {question['question']}\nCorrect Answer: {question['correct_answer']}"
        request_payload = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": explanation_prompt},
        ]
        payload = {
            "model": "gpt-4",
            "messages": request_payload,
            "max_tokens": 500
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        endpoint = "https://api.openai.com/v1/chat/completions"
        response = send_request(endpoint, headers, payload)
        explanation_response = response['choices'][0]['message']['content']
        explanations[question['question']] = explanation_response

    return jsonify({
        'results': results,
        'score': round(score_5_point, 1),
        'total': total_questions,
        'explanations': explanations
    })

@app.route('/generate-exam', methods=['POST'])
def generate_exam():
    data = request.json
    chapter_name = data['chapter_name']
    subchapter_name = data['subchapter_name']
    prompt = data['prompt']
    
    prompt_message = f"""
    Generate an exam for the subchapter '{subchapter_name}' in the chapter '{chapter_name}' of the course on '{prompt}'. 
    Include three types of questions:
    1. Selection problems (multiple-choice) - 1 questions
    2. Fill-in-the-blank problems - 1 questions
    3. Entry problems (short answer) - 1 questions

    Format the response as a JSON array with the following structure:
    [
        {{
            "type": "selection",
            "question": "question text",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "option1"
        }},
        {{
            "type": "fill-in-the-blank",
            "question": "question text with __blank__",
            "correct_answer": "answer"
        }},
        {{
            "type": "entry",
            "question": "question text",
            "correct_answer": "answer"
        }}
    ]
    """
    
    request_payload = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_message},
    ]
    
    payload = {
        "model": "gpt-4",
        "messages": request_payload,
        "max_tokens": 2000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    
    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']
    print("Exam Questions Response:", gpt_response)  # Debug response
    
    try:
        json_response = json.loads(gpt_response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        return jsonify({"error": "Failed to decode JSON response from OpenAI"}), 500

    return jsonify(json_response)

@app.route('/evaluate-exam', methods=['POST'])
def evaluate_exam():
    data = request.json
    questions = data['questions']
    user_answers = data['answers']

    correct_answers = {q['question']: q['correct_answer'] for q in questions}
    results = {q['question']: (user_answers[q['question']] == q['correct_answer']) for q in questions}
    score = sum(results.values())
    total_questions = len(questions)
    score_5_point = (score / total_questions) * 5

    explanations = {}
    for question in questions:
        explanation_prompt = f"Explain the correct answer for the following question:\nQuestion: {question['question']}\nCorrect Answer: {question['correct_answer']}"
        request_payload = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": explanation_prompt},
        ]
        payload = {
            "model": "gpt-4",
            "messages": request_payload,
            "max_tokens": 500
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        endpoint = "https://api.openai.com/v1/chat/completions"
        response = send_request(endpoint, headers, payload)
        explanation_response = response['choices'][0]['message']['content']
        explanations[question['question']] = explanation_response

    return jsonify({
        'results': results,
        'score': round(score_5_point, 1),
        'total': total_questions,
        'explanations': explanations
    })

@app.route('/generate-teacher', methods=['POST'])
def generate_teacher():
    data = request.json
    prompt = data['prompt']
    negative_prompt = ""
    aspect_ratio = "3:2"
    seed = 0
    output_format = "png"

    host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format
    }

    try:
        response = send_generation_request(host, params)
        output_image = response.content
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        if finish_reason == 'CONTENT_FILTERED':
            return jsonify({"error": "Generation failed NSFW classifier"}), 400

        generated = f"generated_{seed}.{output_format}"
        with open(generated, "wb") as f:
            f.write(output_image)
        return send_file(generated, mimetype=f'image/{output_format}')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_voice(text: str, voice_id: str) -> str:
    unique_id = uuid.uuid4()
    output_file_path = os.path.join('static', 'audio', f"generated_{unique_id}.mp3")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    return text_to_speech_file(text, voice_id, output_file_path)

@app.route('/generate-avatar', methods=['POST'])
def generate_avatar():
    if request.method == 'POST':
        try:
            data = request.get_json()
            image_file_name = data.get('imageFileName')
            image_file_data = data.get('imageFileData')
            audio_file_name = data.get('audioFileName')
            audio_file_data = data.get('audioFileData')
            
            if not image_file_name or not image_file_data or not audio_file_name or not audio_file_data:
                return jsonify({'error': 'Missing required file data or model'}), 400
            
            image_buffer = base64.b64decode(image_file_data)
            audio_buffer = base64.b64decode(audio_file_data)
            
            sadtalker_settings = {
                "still": False,
                "ref_pose": None,
                "input_yaw": None,
                "input_roll": None,
                "pose_style": 0,
                "preprocess": 'full',
                "input_pitch": None,
                "ref_eyeblink": None,
                "expression_scale": 1
            }
            
            payload = {
                "face_padding_top": 0,
                "face_padding_bottom": 18,
                "face_padding_left": 0,
                "face_padding_right": 0,
                "sadtalker_settings": sadtalker_settings,
                "selected_model": "SadTalker"
            }
            
            files = {
                'json': (None, json.dumps(payload), 'application/json'),
                'input_face': (image_file_name, BytesIO(image_buffer), 'image/jpeg'),
                'input_audio': (audio_file_name, BytesIO(audio_buffer), 'audio/wav')
            }
            
            headers = {
                "Authorization": f"Bearer {GOOEY_API_KEY}"
            }
            
            response = requests.post("https://api.gooey.ai/v3/Lipsync/async/form/", files=files, headers=headers)
            
            if response.status_code != 200:
                return jsonify({'error': response.text}), response.status_code
            
            status_url = response.headers.get('Location')
            if not status_url:
                raise Exception("Missing status URL")
            
            result = None
            while True:
                status_response = requests.get(status_url, headers=headers)
                if status_response.status_code != 200:
                    raise Exception(f"{status_response.status_code}")
                
                result = status_response.json()
                if result.get('status') in ["completed", "failed"]:
                    break
                else:
                    time.sleep(3)
            
            print("Final Result:", result)  # Debug log for final result
            return jsonify(result), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    else:
        return jsonify({'error': f"Method {request.method} Not Allowed"}), 405


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)