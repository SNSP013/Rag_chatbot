import os
import requests
from dotenv import load_dotenv
load_dotenv() 


SERPER_API_KEY = os.getenv("SERPER_API_KEY")  
SERPER_URL = "https://google.serper.dev/search"

def search_web(query: str, num_results: int = 3):
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query
    }

    response = requests.post(SERPER_URL, headers=headers, json=payload)

    results = response.json()

    web_results = []
    for item in results.get("organic", [])[:num_results]:
        web_results.append({
            "source": "web",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })
    return web_results
