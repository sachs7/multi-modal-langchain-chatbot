from langchain.agents import tool

from services.dalle3_image_generator import image_generator


@tool
def generate_image_based_on_user_prompt(
    prompt: str,
):
    """Generates an image based on the prompt."""
    result = image_generator(prompt)
    return result
