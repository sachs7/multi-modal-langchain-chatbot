from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from loguru import logger

# Reference: [GitHub](https://github.com/nirbar1985/country-compass-ai/blob/main/agents/tools/countries_image_generator.py)

def countries_image_generator(country: str):
    """Call this to get an image of a country"""
    system_prompt = f"""
    You generate image of a country representing the most typical country's characteristics, 
    incorporating its flag. the country is {country}.
    Resize the image to show 800x800 pixels.
    """
    logger.info(f"Generating image of {country} using DALL-E 3")

    res = DallEAPIWrapper(model="dall-e-3").run(system_prompt)

    answer_to_agent = (
        f"Use this format- Here is an image of {country}: [{country} Image]"
        f"url= {res}"
    )
    return answer_to_agent
