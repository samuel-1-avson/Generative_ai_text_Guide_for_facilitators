# Import necessary libraries
import PyPDF2
import openai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Initialize the Flask application
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'api-key'

# Global variable to store the guide text
guide_text = ''

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to search for relevant sections in the guide text based on user query
def search_relevant_section(query, guide_text):
    keywords = query.lower().split()
    sentences = guide_text.split('.')
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return ' '.join(relevant_sentences[:5])

# Function to generate a response from the chatbot using OpenAI API
def chatbot_response(user_query, relevant_guide_text):
    prompt = f"User: {user_query}\nGuide Content: {relevant_guide_text}\nAnswer based on the guide content:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Give precise and helpful answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Define the route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    global guide_text 
    
    # Get the uploaded file from the request
    file = request.files.get('file')
    
    if file is None or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Save the file to the server
    upload_folder = 'uploads/'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    
    # Extract text from the PDF file and update global guide_text
    guide_text = extract_text_from_pdf(file_path)
    
    return jsonify({'message': 'File uploaded and processed successfully.'})

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index2.html')

# Define the route for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
    global guide_text  # To access the global guide_text variable
    
    # Check if the PDF has been uploaded and processed
    if not guide_text:
        return jsonify({'response': "No guide uploaded. Please upload a PDF first."}), 400
    
    user_query = request.form['query']
    relevant_section = search_relevant_section(user_query, guide_text)
    if relevant_section:
        response = chatbot_response(user_query, relevant_section)
    else:
        response = "I couldn't find relevant information in the guide."
    return jsonify({'response': response})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
