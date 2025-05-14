import PyPDF2 
import openai  
from flask import Flask, render_template, request, jsonify  

# Initialize the Flask application
app = Flask(__name__)

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = 'api-key'  # Replace with your actual key

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  
        text = ''  
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]  
            text += page.extract_text()  
    return text  

# Function to generate content based on PDF text and user-provided prompt
def generate_content_from_pdf(guide_text, prompt):
    # Create a prompt for generating new content based on the PDF
    content_generation_prompt = f"Here is some information from the guide: {guide_text[:1000]}. Now, based on this information, create a detailed content or article on the topic '{prompt}'."

    # Use OpenAI's API to generate new content based on the extracted text and user prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a content creator that generates detailed articles and content based on provided documents."},
            {"role": "user", "content": content_generation_prompt}
        ],
        max_tokens=500 
    )
    return response['choices'][0]['message']['content'].strip()

# Function to generate content based on a direct user prompt (without the PDF)
def generate_content_from_prompt(user_prompt):
    # Create a prompt for generating new content based on the user's input
    content_generation_prompt = f"Create a detailed article, explanation, or content based on this prompt: '{user_prompt}'."

    # Use OpenAI's API to generate new content based on the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert content creator who generates detailed articles and explanations based on user requests."},
            {"role": "user", "content": content_generation_prompt}
        ],
        max_tokens=500  
    )
    return response['choices'][0]['message']['content'].strip()

# Load the PDF once when the server starts
pdf_path = r'C:\Users\HP\OneDrive\Desktop\Applications\Projects\Geneartive_ai\[Facilitator Guide] Generative AI.pdf'  # Replace with your PDF file path
guide_text = extract_text_from_pdf(pdf_path)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index3.html')  

# Route to generate content based on the PDF
@app.route('/create-content-from-pdf', methods=['POST'])
def create_content_from_pdf():
    user_prompt = request.form['prompt']
    generated_content = generate_content_from_pdf(guide_text, user_prompt)  
    return jsonify({'response': generated_content}) 

# Route to generate content based on a direct prompt
@app.route('/create-content-from-prompt', methods=['POST'])
def create_content_from_prompt():
    user_prompt = request.form['prompt']  
    generated_content = generate_content_from_prompt(user_prompt)  
    return jsonify({'response': generated_content})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)  
import PyPDF2 
import openai  
from flask import Flask, render_template, request, jsonify  

# Initialize the Flask application
app = Flask(__name__)

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = 'api-key'  # Replace with your actual key

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  
            text = ''  
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]  
                text += page.extract_text()  
        return text  
    except Exception as e:
        return str(e)

# Function to generate content based on PDF text and user-provided prompt
def generate_content_from_pdf(guide_text, prompt):
    try:
        # Create a prompt for generating new content based on the PDF
        content_generation_prompt = f"Here is some information from the guide: {guide_text[:1000]}. Now, based on this information, create a detailed content or article on the topic '{prompt}'."

        # Use OpenAI's API to generate new content based on the extracted text and user prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content creator that generates detailed articles and content based on provided documents."},
                {"role": "user", "content": content_generation_prompt}
            ],
            max_tokens=500 
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)

# Function to generate content based on a direct user prompt (without the PDF)
def generate_content_from_prompt(user_prompt):
    try:
        # Create a prompt for generating new content based on the user's input
        content_generation_prompt = f"Create a detailed article, explanation, or content based on this prompt: '{user_prompt}'."

        # Use OpenAI's API to generate new content based on the prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert content creator who generates detailed articles and explanations based on user requests."},
                {"role": "user", "content": content_generation_prompt}
            ],
            max_tokens=500  
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)

# Load the PDF once when the server starts
pdf_path = r'C:\Users\HP\OneDrive\Desktop\Applications\Projects\Geneartive_ai\[Facilitator Guide] Generative AI.pdf'  # Replace with your PDF file path
try:
    guide_text = extract_text_from_pdf(pdf_path)
except Exception as e:
    guide_text = str(e)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index3.html')  

# Route to generate content based on the PDF
@app.route('/create-content-from-pdf', methods=['POST'])
def create_content_from_pdf():
    user_prompt = request.form['prompt']
    generated_content = generate_content_from_pdf(guide_text, user_prompt)  
    return jsonify({'response': generated_content}) 

# Route to generate content based on a direct prompt
@app.route('/create-content-from-prompt', methods=['POST'])
def create_content_from_prompt():
    user_prompt = request.form['prompt']  
    generated_content = generate_content_from_prompt(user_prompt)  
    return jsonify({'response': generated_content})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)import PyPDF2 
import openai  
from flask import Flask, render_template, request, jsonify  

# Initialize the Flask application
app = Flask(__name__)

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = 'api-key'  # Replace with your actual key

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  
            text = ''  
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]  
                text += page.extract_text()  
        return text  
    except Exception as e:
        return str(e)

# Function to generate content based on PDF text and user-provided prompt
def generate_content_from_pdf(guide_text, prompt):
    try:
        # Create a prompt for generating new content based on the PDF
        content_generation_prompt = f"Here is some information from the guide: {guide_text[:1000]}. Now, based on this information, create a detailed content or article on the topic '{prompt}'."

        # Use OpenAI's API to generate new content based on the extracted text and user prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content creator that generates detailed articles and content based on provided documents."},
                {"role": "user", "content": content_generation_prompt}
            ],
            max_tokens=500 
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)

# Function to generate content based on a direct user prompt (without the PDF)
def generate_content_from_prompt(user_prompt):
    try:
        # Create a prompt for generating new content based on the user's input
        content_generation_prompt = f"Create a detailed article, explanation, or content based on this prompt: '{user_prompt}'."

        # Use OpenAI's API to generate new content based on the prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert content creator who generates detailed articles and explanations based on user requests."},
                {"role": "user", "content": content_generation_prompt}
            ],
            max_tokens=500  
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)

# Load the PDF once when the server starts
pdf_path = r'C:\Users\HP\OneDrive\Desktop\Applications\Projects\Geneartive_ai\[Facilitator Guide] Generative AI.pdf'  # Replace with your PDF file path
try:
    guide_text = extract_text_from_pdf(pdf_path)
except Exception as e:
    guide_text = str(e)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index3.html')  

# Route to generate content based on the PDF
@app.route('/create-content-from-pdf', methods=['POST'])
def create_content_from_pdf():
    try:
        user_prompt = request.form['prompt']
        generated_content = generate_content_from_pdf(guide_text, user_prompt)  
        return jsonify({'response': generated_content}) 
    except Exception as e:
        return jsonify({'response': str(e)})

# Route to generate content based on a direct prompt
@app.route('/create-content-from-prompt', methods=['POST'])
def create_content_from_prompt():
    try:
        user_prompt = request.form['prompt']  
        generated_content = generate_content_from_prompt(user_prompt)  
        return jsonify({'response': generated_content})
    except Exception as e:
        return jsonify({'response': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)