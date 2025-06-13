from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("corpus.index")

with open("corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

def infer_condition(query: str) -> str:
    q = query.lower()
    if "glucometer" in q and ("55" in q or "low" in q or "shaky" in q):
        return "Hypoglycemia"
    elif "sugar crashed" in q or "unconscious" in q and "diabetic" in q:
        return "Severe Hypoglycemia"
    elif "gestational" in q and ("130" in q or "high sugar" in q):
        return "Gestational Diabetes"
    elif "chest pain" in q and "left arm" in q:
        return "Myocardial Infarction"
    elif "angina" in q:
        return "Angina"
    elif "short of breath" in q and "heart failure" in q:
        return "Chronic Heart Failure"
    elif "creatinine" in q and ("barely urinated" in q or "sun" in q):
        return "Acute Kidney Injury (AKI)"
    elif "potassium" in q and "6.1" in q:
        return "Hyperkalemia"
    elif "ibuprofen" in q and "flanks hurt" in q:
        return "Drug-induced AKI"
    elif "thirsty" in q and ("hi" in q or "very high glucose" in q):
        return "Type 2 Diabetes - Hyperglycemia"
    else:
        return "General Medical Emergency"


def retrieve_answer(query, top_k=3):
    # Step 1: Local Semantic Search
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    local_results = []
    for idx in indices[0]:
        local_results.append(corpus[idx])

    # Step 2: Web Search via Serper.dev
    web_data = search_web(query)
    web_snippet, web_title, web_link = None, None, None

    if 'answerBox' in web_data:
        web_snippet = web_data['answerBox'].get('snippet')
        web_title = web_data['answerBox'].get('title')
        web_link = web_data['answerBox'].get('link')
    elif 'organic' in web_data and len(web_data['organic']) > 0:
        top_result = web_data['organic'][0]
        web_snippet = top_result.get('snippet')
        web_title = top_result.get('title')
        web_link = top_result.get('link')

    web_result = {
        "title": web_title,
        "snippet": web_snippet,
        "link": web_link
    } if web_snippet else None

    return {
        "local": local_results,
        "web": web_result
    }

from src.query_engine import infer_condition, retrieve_answer
from src.web_search import search_web

def generate_response(user_query: str, num_sources: int = 2) -> str:
    disclaimer = (
        "⚠️ This information is for educational purposes only and is not a substitute for professional medical advice.\n"
    )

    condition = infer_condition(user_query)

    local_result = retrieve_answer(user_query)
    web_result = search_web(f"first aid for {condition}")

    bullet_points = fuse_local_web(local_result["local"], local_result["web"])

    response_lines = [
        disclaimer,
        f"\n🩺 **Likely Condition:** {condition}",
        "\n🆘 **Recommended First-Aid Steps:**",
    ]
    response_lines.extend(bullet_points)

    if web_result:
        response_lines.append("\n📚 **Source(s):**")
        for item in web_result:
            response_lines.append(f"- {item['title']} ({item['link']})")

    return "\n".join(response_lines[:20])


def fuse_local_web(local_results, web_result):
    fused_lines = set()

    for line in local_results:
        line = line.strip("-• \n\t.").capitalize()
        if line and len(line.split()) > 3:
            fused_lines.add(f"- {line}")

    if web_result and web_result.get("snippet"):
        web_line = web_result["snippet"].strip().capitalize()
        if web_line and len(web_line.split()) > 4:
            fused_lines.add(f"- {web_line} ({web_result['title']})")

    return list(fused_lines)[:6]
