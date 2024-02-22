import requests
from typing import Dict
from loguru import logger

# import json


def make_api_request(
    url: str = "https://paperswithcode.com/api/v1/papers/",
    params: Dict[str, int | str] = {"q": "llm", "page": 1, "items_per_page": 2},
    retries: int = 3,
):
    """
    Makes an API request to the Papers with Code Site

    Parameters:
    - url: The URL of the API endpoint
    - params: The query parameters to send with the request
        example: {"page": 1, "items_per_page": 2, "q": "llm"}
    - retries: The number of times to retry the request if it fails

    Returns:
    - A list of dictionaries containing the details of the papers
    """
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # total_papers = len(data['results'])
        papers = data["results"]

        final_result = []
        logger.info(f"Request URL: {response.url}")

        for paper in papers:
            title = paper["title"]
            authors = paper["authors"]
            abstract = paper["abstract"]
            pdf_url = paper["url_pdf"]
            final_result.append(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                }
            )

        return final_result

    except requests.exceptions.RequestException as e:
        if retries > 0:
            # Retry the request
            return make_api_request(url, params, retries - 1)
        else:
            logger.error("Error making API request:", e)
            return None


# url = "https://paperswithcode.com/api/v1/papers/"
# params = {"page": 1, "items_per_page": 2, "q": "llm"}
# result = make_api_request(url, params=params)
# json_result = json.dumps(result)
# print(json_result)
