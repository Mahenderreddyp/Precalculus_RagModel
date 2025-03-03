from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import logging
import sys
from model import *
from config import *
import os
import json
import datetime

app = Flask(__name__)
CORS(app)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create templates folder if it doesn't exist
templates_folder = 'templates'
if not os.path.exists(templates_folder):
    os.makedirs(templates_folder)

# Move the HTML file to the templates folder
template_path = os.path.join(templates_folder, 'precalculus_template.html')


@app.route('/', methods=['GET'])
def index():
    """Serve HTML interface for browser interaction"""
    return render_template('precalculus_template.html')

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Precalculus AI Assistant API",
        "version": "1.0",
        "endpoints": {
            "/": "GET - Browser interface",
            "/api/info": "GET - API information",
            "/api/question": "POST - Ask a question about precalculus"
        },
        "usage": "Send a POST request to /api/question with JSON body containing 'question', 'user_id', 'mode', and 'topic' fields"
    }), 200
    
@app.route('/api/feedback', methods=['POST'])
def post_feedback():
    """Handle feedback from both JSON and form submissions"""
    # Check if JSON data
    if request.is_json:
        data = request.get_json(silent=True)
    else:
        # Handle form data
        data = request.form
    
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    
    # Process feedback
    is_helpful = data.get('helpful') == 'true' or data.get('helpful') == True
    user_id = data.get('user_id', 'unknown')
    topic = data.get('topic', 'unknown')
    mode = data.get('mode', 'unknown')
    
    feedback_type = "positive" if is_helpful else "negative"
    logging.info(f"Received {feedback_type} feedback from user '{user_id}' on topic '{topic}' in mode '{mode}'")
    
    # For form submissions, return a simple HTML message
    if request.is_json:
        return jsonify({"status": "success"}), 200
    else:
        return """
        <html>
        <body>
            <p>Feedback received, thank you!</p>
        </body>
        </html>
        """, 200
        

@app.route('/api/question', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    
    if not json:
        return jsonify({"error": "Invalid JSON request"}), 400
    
    if 'question' not in json or 'user_id' not in json:
        return jsonify({"error": "Missing required fields: 'question' and 'user_id' are required"}), 400
    
    question = json['question']
    user_id = json['user_id']
    mode = json.get('mode', 'review')  # Default to review mode
    topic = json.get('topic', '')      # Get the selected topic
    
    # Format question based on mode and topic
    formatted_question = f"Mode: {mode} | Topic: {topic} | Question: {question}"
    
    logging.info("post question `%s` for user `%s`", formatted_question, user_id)

    try:
        resp = chat(formatted_question, user_id)
        data = {'answer': resp}
        return jsonify(data), 200
    except Exception as e:
        logging.error("Error processing question: %s", str(e))
        return jsonify({"error": "Failed to process question"}), 500

if __name__ == '__main__':
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)