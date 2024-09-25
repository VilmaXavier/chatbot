import tensorflow as tf

# Configure GPU settings if applicable
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Set to 4096MB or appropriate value
            )
    except RuntimeError as e:
        print(e)

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

    # Check and truncate answer if too long
    max_length = 1000  # Set a maximum length for the answer
    answer_text = answer['answer'][:max_length]  # Truncate to first 1000 characters if necessary

    return jsonify({
        "question": user_query,
        "answer": answer_text
    })

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
