from ollama import Client
from dotenv import load_dotenv
import os

load_dotenv() 
client = Client(host='http://localhost:11434')

def generate_answer(query, local_snippets, web_snippets):
    system_prompt = """
You are a medical first-aid assistant. Provide ≤250-word responses based on trusted info.
Include:
- Most likely condition
- First-aid steps
- Key medication(s)
- Citations from sources

⚠️ Always start with this:
"This information is for educational purposes only and is not a substitute for professional medical advice."
"""

    input_context = "\n".join(["Local: " + s for s in local_snippets] + ["Web: " + s for s in web_snippets])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nRelevant Info:\n{input_context}"}
    ]

    response = client.chat(model='llama3', messages=messages)
    return response['message']['content']

