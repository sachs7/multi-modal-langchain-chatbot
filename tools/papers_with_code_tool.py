import json
from typing import Dict
from langchain.agents import tool

from services.papers_with_code import make_api_request


@tool
def make_api_request_to_papers_with_code(
    url: str = "https://paperswithcode.com/api/v1/papers/",
    params: Dict[str, int | str] = {"q": "llm", "page": 1, "items_per_page": 2},
    retries=3,
):
    """Makes an API request to the Papers with Code Site"""
    result = make_api_request(url, params=params)
    json_response = json.dumps(result)
    return json_response
