from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To enable cross-origin requests

# Load pre-trained transformer model for Question-Answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load the college data from the file dynamically
with open("college_data.txt", "r") as file:
    context = file.read()

# Route for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_query = request.json.get('query')
    
    # Use transformer to answer the user's question based on context
    qa_input = {
        'question': user_query,
        'context': context  # Using the file content as context
    }
    
    # Get answer from the QA model
    answer = qa_pipeline(qa_input)
    
    return jsonify({
        "question": user_query,
        "answer": answer['answer']
    })

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
