from dotenv import load_dotenv
import os
import httpx
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import bs4
import requests
from langchain.docstore.document import Document
import re
from typing import List

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_KEY = os.getenv("SEARCH_KEY")

# Helper function for preprocessing documents
def preprocess_documents(documents: List[Document]) -> List[Document]:
    processed_docs = []
    for doc in documents:
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', doc.page_content).strip()

        # Remove any remaining script or style tags if they were not already removed
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)

        # Remove any leftover special characters and whitespace
        text = re.sub(r'[^\w\s.,?!()/;-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Preserve the metadata (source, title, etc.)
        processed_doc = Document(page_content=text, metadata=doc.metadata)
        processed_docs.append(processed_doc)

    return processed_docs

# Main function to search, extract URLs, and load documents
def search_google(query: str, topk: int = 3, lan: str = 'en', **params):
    # Perform Google search
    base_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': SEARCH_KEY,
        'q': query,
        'hl': lan,
        **params
    }
    response = httpx.get(base_url, params=params)
    response.raise_for_status()
    items = response.json().get('items', [])
    urls = [item['link'] for item in items]

    # Validate the URLs:
    unique_urls=set()
    search_urls=[]
    
    for url in urls:
        if url.split("/")[2] not in unique_urls:
            try:
                response = requests.get(url, timeout=1) 
                status_code = response.status_code
                
            except requests.exceptions.RequestException as e:              
                status_code = None
                
            if status_code == 200:
                unique_urls.add(url.split("/")[2])
                print(f"Currently searching the website: {url}")
                search_urls.append(url) 
                
            if len(unique_urls) >= topk:
                break
    
    if search_urls:
        # Load documents from the URLs
        loader = WebBaseLoader(
            web_paths=search_urls,  # Limit to top 3 URLs
            bs_kwargs=dict(parse_only=bs4.SoupStrainer()),
            requests_kwargs={"timeout": 600}
        )
        docs = loader.load()
        return preprocess_documents(docs)

        # # Create a formatted string of document data
        # doc_string = "\n\n".join(
        #     f"URL {index + 1}\n"
        #     f"Source: {doc.metadata.get('source', 'N/A')}\n"
        #     f"Title: {doc.metadata.get('title', 'N/A')}\n"
        #     f"Description: {doc.metadata.get('description', 'N/A')}\n"
        #     f"Content: {doc.page_content}\n"
        #     for index, doc in enumerate(docs)
        # )

        # return doc_string

if __name__ == "__main__":
    search_google('bitcoin price today', 10)
