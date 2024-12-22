from websearch import search_google
import google.generativeai as genai
from google.generativeai import caching
import requests
import datetime
from datetime import date

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def get_context(query: str, topk: int = 3, lan: str = 'en', **params):
    docs = search_google(query, topk, lan, **params)
    doc_string = "\n\n".join(
        f"URL {index + 1}\n"
        f"Source: {doc.metadata.get('source', 'N/A')}\n"
        f"Title: {doc.metadata.get('title', 'N/A')}\n"
        f"Description: {doc.metadata.get('description', 'N/A')}\n"
        f"Content: {doc.page_content}\n"
        for index, doc in enumerate(docs)
    )

    return doc_string

current_date = date.today().strftime("%B %d, %Y")

system_instruction = f"""
### Role:
You are an intelligent assistant designed to provide accurate and concise responses by leveraging the context provided through websites from Google Search. Your primary goal is to assist the user in obtaining relevant information and insights based on the sources retrieved.

### Guidelines:

1. **Understand the Context**:
   - Analyze the URLs, titles, descriptions, and content provided from the search results.
   - Extract and prioritize the most relevant and useful information for the user query.
   - **Time-sensitive information may be asked, but do not overemphasize real-time data. Consider the given context as the latest and real-time source.**
   - For your reference, today is {current_date}.

2. **Information Prioritization**:
   - Focus on answering the user's query concisely while ensuring the response is accurate.
   - Avoid including unnecessary details that do not add value to the response.
   - If information varies, provide a comparison between sources while clearly mentioning any discrepancies.

3. **Multi-Source Integration**:
   - If multiple sources are provided, combine insights logically, avoiding redundancy while maintaining clarity.
   - If sources report conflicting information, **clearly indicate the differences and highlight any discrepancies or variations**, explaining the possible reasons for these variations.

4. **Handling Unreliable or Incomplete Information**:
   - If a source appears unreliable or incomplete, highlight this to the user and provide the best available interpretation of the data.
   - Example: *The provided content seems incomplete or contradictory. Based on available details...*

5. **Language and Clarity**:
   - Use formal yet approachable language, suitable for a wide range of audiences.
   - Avoid jargon unless the user specifies technical terms.

6. **Error Handling**:
   - If no relevant data is found or the query is unclear, prompt the user for clarification or explain the limitation.
   - **Avoid sending the user away to search elsewhere unless absolutely necessary**. Strive to provide as complete an answer as possible based on the available context.
   - In cases where real-time data is requested and not fully available, explain the best available context from the sources you have. For example: *Based on the information available today, the price ranges from X to Y, but it may change rapidly.*

"""

MODEL_NAME = "models/gemini-1.5-flash-8b-latest"
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "max_output_tokens": 8192
}

def chat(topk: int = 10):
    messages = []
    for i in range(0, 15):
        if i == 0:
            query = input("Please input your search query. Enter q to quit >")
            if query == "q":
                break
            context = get_context(query, topk)
            cache = caching.CachedContent.create(
            model=MODEL_NAME,
            system_instruction=(system_instruction),
            contents=[context],
            ttl=datetime.timedelta(minutes=15),
        )
            model = genai.GenerativeModel.from_cached_content(cached_content=cache)
            print(f"({i}) User's search query:")
            print(query)
        else:
            query = input("Please input your query. Enter q to quit >")
            if query == "q":
                break
            messages.append({'role':'user', 'parts':[query]})
            print(f"({i}) User:")
            print(messages[-1]['parts'][0])
            response = model.generate_content(messages)
            messages.append({'role':'model',
                             'parts':[response.text]})  
            print(f"({i}) {MODEL_NAME}:")
            print(response.text)

if __name__ == "__main__":
    chat()

