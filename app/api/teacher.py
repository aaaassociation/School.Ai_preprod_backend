from app import app, db
from app.config import OPENAI_API_KEY, GOOEY_API_KEY
from app.utils import send_request, send_request_with_rate_limit, send_generation_request, search_image, generate_voice
from flask import request, jsonify, send_file
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO

import os
import json
import time
import base64
import requests
import openai

socketio = SocketIO(app, cors_allowed_origins="*")

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

@app.route('/generate-chapters', methods=['POST'])
def generate_chapters():
    data = request.json
    prompt = data['prompt']
    num_chapters = data.get('chapters', 6)
    num_subchapters = data.get('chapterDepth', 3)
    voice = data.get('voice')
    
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
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']
    
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
    chapters = data['chapters']
    prompt = data['prompt']
    voice_id = data['voice_id']
    voice = data.get('voice')

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
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        endpoint = "https://api.openai.com/v1/chat/completions"

        while True:
            response = send_request_with_rate_limit(endpoint, headers, payload)

            if response and 'choices' in response and len(response['choices']) > 0:
                gpt_response = response['choices'][0]['message']['content']
                return gpt_response
            else:
                print("Invalid response, retrying...")
                time.sleep(2)

    def notify_user(subchapter_content, chapter_name, subchapter_name):
        content_parts = subchapter_content.split('[IMAGE:')
        final_content = content_parts[0]
        
        for part in content_parts[1:]:
            try:
                image_prompt, rest_of_content = part.split(']', 1)
                image_url = search_image(image_prompt.strip())
                if image_url:
                    final_content += f'<<Image:URL>> {image_url}\n' + rest_of_content
                else:
                    final_content += f'[IMAGE: {image_prompt.strip()}]' + rest_of_content
            except ValueError:
                final_content += f'[IMAGE: {part.strip()}]'
          
                
        socketio.emit('subchapter_created', {
            'chapter': chapter_name,
            'subchapter': subchapter_name,
            'content': final_content
        })
        print("SubChapter Created", chapter_name, subchapter_name)

    def process_chapters():
        with ThreadPoolExecutor() as executor:
            first_subchapter_flag = False
            
            for chapter_name, subchapters in chapters.items():
                for subchapter_index, subchapter_name in enumerate(subchapters):
                    # Fetch content for the subchapter
                    future_fetch = executor.submit(fetch_content, chapter_name, subchapter_name)                    
                    subchapter_content = future_fetch.result()
                    
                    if subchapter_content:
                        if not first_subchapter_flag and subchapter_index == 0:
                            first_subchapter_flag = True
                            socketio.emit('first_subchapter_created', {
                                'chapter': chapter_name,
                                'subchapter': subchapter_name,
                                'content': subchapter_content
                            })
                            print("First SubChapter", chapter_name, subchapter_name)
                        else:
                            notify_user(subchapter_content, chapter_name, subchapter_name)
                        
                        if (voice):
                            executor.submit(generate_voice, chapter_name, subchapter_name, subchapter_content, prompt, voice_id)
                            
            socketio.emit('content_generation_complete', {'message': 'All content has been generated.'})
            print("Finished")
            
    socketio.start_background_task(process_chapters)
    return jsonify({"message": "Content generation started."})

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
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    endpoint = "https://api.openai.com/v1/chat/completions"
    response = send_request(endpoint, headers, payload)
    gpt_response = response['choices'][0]['message']['content']

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

@app.route('/fetch-explanation', methods=['POST'])
def fetch_explanation():
    data = request.json
    chapter_name = data['chapter_name']
    subchapter_name = data['subchapter_name']
    prompt = data['prompt']

    max_wait_time = 60
    poll_interval = 10
    elapsed_time = 0

    while elapsed_time < max_wait_time:
        audio_record = db.collection('audio')\
            .where('chapter_name', '==', chapter_name)\
            .where('subchapter_name', '==', subchapter_name)\
            .where('prompt', '==', prompt)\
            .stream()

        audio_record_list = [doc.to_dict() for doc in audio_record]

        if audio_record_list:
            name = audio_record_list[0]['name']
            file_path = os.path.join('static', 'audio', f"{name}.mp3")
            file_path = os.path.abspath(file_path)

            if os.path.exists(file_path):
                return send_file(file_path, mimetype='audio/mpeg')
            else:
                print(f"File {file_path} not found, waiting...")
        else:
            print("Audio info not found in Firebase, waiting...")

        time.sleep(poll_interval)
        elapsed_time += poll_interval

    return jsonify({"error": "Audio file is not yet available after waiting."}), 408

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