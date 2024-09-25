from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To enable cross-origin requests

# Route for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_query = request.json.get('query')
    return jsonify({
        "question": user_query,
        "answer": "This is a placeholder answer."
    })

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
