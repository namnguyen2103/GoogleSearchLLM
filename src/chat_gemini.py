from dotenv import load_dotenv
import os
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

1. **Contextual Analysis**:
   - Carefully analyze the URLs, titles, descriptions, and content provided from the search results.
   - Prioritize extracting the most relevant information to directly answer the user's query.
   - **If the user asks for time-sensitive data, assume the provided context is the most up-to-date available, even if not explicitly stated as real-time.** Focus on giving an answer based on the information given, rather than admitting you don't have it.
   - For your reference, today is {current_date}.

2. **Direct and Concise Answers**:
   - Focus on answering the user's query directly and concisely.
   - Avoid including unnecessary information that doesn't contribute to the answer.
   - If information is limited or not ideally specific, use the best information to provide a direct answer, noting that the information may have limitations.
   - If there are varying sources, use all the sources while making sure that the user is aware of any discrepancies.

3. **Multi-Source Synthesis**:
    -  Synthesize information from different sources into a coherent answer, avoiding redundancy.
   -  When sources conflict or differ, present those differences with explanations as to why that may occur.

4.  **Handling Imperfect Information**:
    - When information is not perfect, or the request is not directly answerable, use the context and make the best inference possible.
    - Example: If the exact price is not found, but a range is provided, mention the range. If a date is not exact, use the closest information to make the inference.
    - If a source is unreliable, mention this in the answer but still use the best available interpretation of the data.

5. **Clarity and Tone**:
   - Use formal yet approachable language suitable for a broad audience.
   - Avoid overly technical jargon unless the user's query indicates otherwise.

6. **Proactive Problem Solving**:
   - Attempt to answer the user's question fully based on the provided context. If the context cannot fully answer the query, provide the best possible answer by combining sources and making logical inferences.
   - **Avoid telling the user to search elsewhere. Your role is to answer based on the given context as best you can, not to suggest other sources**.
   - For queries about real-time data, use the most current information present in the provided context. If the data is limited or not fully up-to-date, state that as a limitation while still attempting to answer the question fully with the provided context. For example: *Based on the available information, the price is between X and Y as of this article's date, but it may change rapidly.*
"""

MODEL_NAME = "models/gemini-1.5-flash-002"
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "max_output_tokens": 8192
}

def chat(topk: int = 10):
    messages = []
    for i in range(0, 15):
        if i == 0:
            query = input("Please input your search query. Enter q to quit > ")
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
            query = input("Please input your query. Enter q to quit > ")
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

