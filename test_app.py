# test_app.py - A minimal working example to test Flask setup
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import logging
import sys

app = Flask(__name__)
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple HTML template 
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Precalculus Test</title>
    <style>
        body { font-family: Arial; padding: 20px; }
    </style>
</head>
<body>
    <h1>Precalculus Test App</h1>
    <p>This is a test page to verify the server is working correctly.</p>
    <button id="test-btn">Test API</button>
    <div id="result"></div>
    
    <script>
        document.getElementById('test-btn').addEventListener('click', async function() {
            try {
                const response = await fetch('/api/info', {
                    method: 'GET',
                });
                
                const data = await response.json();
                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').textContent = 'Error: ' + error.message;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Serve HTML interface for browser interaction"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Test API",
        "version": "1.0",
        "status": "Working"
    }), 200

if __name__ == '__main__':
    # For testing, we'll use a hard-coded port
    PORT = 7654
    print(f"Starting test server on http://localhost:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=True)