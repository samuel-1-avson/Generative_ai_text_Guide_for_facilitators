import openai
import os

# Load the API key from an environment variable
openai.api_key =os.getenv('api-key')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

def generate_image(prompt):
    """
    Generates an image based on the given prompt using OpenAI's updated API.

    Args:
        prompt (str): The description of the image to generate.

    Returns:
        str: The URL of the generated image.
    """
    try:
        response = openai.Image.create_edit(
            prompt=prompt,
            n=1,  # Number of images to generate
            size="512x512"  # Image resolution
        )
        return response['data'][0]['url']
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"Failed to generate image: {e}")

def main():
    """
    Main function to interact with the user and generate images based on their input.
    """
    print("Generative Image AI: Hi! Describe the image you'd like to generate.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Generative Image AI: Goodbye! Have a great day!")
            break

        try:
            image_url = generate_image(user_input)
            print(f"Generative Image AI: Here is your generated image: {image_url}")
        except RuntimeError as e:
            print(f"Generative Image AI: {e}")

if __name__ == "__main__":
    main()