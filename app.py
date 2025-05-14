# Import necessary libraries



# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import PyPDF2
import os

# Initialize the Flask application
app = Flask(__name__)

# Set your OpenAI API key securely (RECOMMENDED: use environment variables)
client = OpenAI(api_key='api-key')  # Replace with your actual key

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to search for relevant sections in the guide text based on user query
def search_relevant_section(query, guide_text):
    keywords = query.lower().split()
    sentences = guide_text.split('.')
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return '. '.join(relevant_sentences[:5])

# Function to generate a response from the chatbot using OpenAI API
def chatbot_response(user_query, relevant_guide_text):
    prompt = f"User: {user_query}\nGuide Content: {relevant_guide_text}\nAnswer based on the guide content:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Give precise and helpful answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Load the PDF once when the server starts
pdf_path = r'C:\Users\HP\OneDrive\Desktop\Applications\Projects\Geneartive_ai\[Facilitator Guide] Generative AI.pdf'
guide_text = extract_text_from_pdf(pdf_path)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
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
