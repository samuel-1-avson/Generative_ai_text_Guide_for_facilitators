##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
import openai
import os
from flask import Flask, request, jsonify, render_template

# Set your OpenAI API key (use environment variable for security)
openai.api_key = os.getenv("api-key")  # Replace with your actual environment variable

app = Flask(__name__)

# Route for homepage (chat interface)
@app.route('/')
def home():
    return render_template('chatbot.html')  

# Route to handle chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']  

    # Differentiate between commands based on input
    if 'lesson plan' in user_input.lower():
        grade_level = request.form.get('grade_level', 'high school')  # Default grade level if not provided
        topic = request.form.get('topic', 'robotics')
        
        # Generate a lesson plan using OpenAI
        prompt = f"Create a detailed lesson plan for {grade_level} students on {topic}. Include objectives, materials, activities, and assessments."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        answer = response.choices[0].text.strip()

    elif 'troubleshoot' in user_input.lower():
        issue_description = user_input.replace('troubleshoot', '').strip()
        
        # Provide troubleshooting steps
        prompt = f"Provide troubleshooting steps for this robotics issue: {issue_description}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        answer = response.choices[0].text.strip()

    else:
        # Generic question-answering
        prompt = f"Answer this question for a robotics class: {user_input}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        answer = response.choices[0].text.strip()

    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
