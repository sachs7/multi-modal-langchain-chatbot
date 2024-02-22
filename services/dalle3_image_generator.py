from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from loguru import logger

# Reference: [GitHub](https://github.com/nirbar1985/country-compass-ai/blob/main/agents/tools/countries_image_generator.py)

def image_generator(query: str):
    """Call this to get an image of a country"""
    system_prompt = f"""
    You are a creative abstract painter, generate a subtle but captivating abstract painting/image
    of the user {query}. Resize the image to 800x800 pixels.
    """
    logger.info(f"Generating image of {query} using DALL-E 3")

    res = DallEAPIWrapper(model="dall-e-3").run(system_prompt)

    answer_to_agent = (
        f"Use this format- Here is an image of {query}: [{query} Image]"
        f"url= {res}"
    )
    return answer_to_agent
