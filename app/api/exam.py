from app import app
from flask import request, jsonify
from app.config import OPENAI_API_KEY
from app.utils import send_request

import json
import openai

openai.api_key = OPENAI_API_KEY

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
