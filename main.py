from src.retriever import semantic_search, serper_search
from src.responder import generate_answer
from dotenv import load_dotenv
load_dotenv()

def chatbot_pipeline(user_input):
    local = semantic_search(user_input)
    web = serper_search(user_input)
    answer = generate_answer(user_input, local, web)
    return answer

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]: break
        print("\n" + chatbot_pipeline(q) + "\n")
