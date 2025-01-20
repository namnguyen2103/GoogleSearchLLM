# GoogleSearchLLM

## Introduction

GoogleSearchLLM is a Google search chatbot that enables users to search Google for any information and chat about related information. 

The chatbot consists of two main components:
1. **Google Search**: Utilizes the Google search API to query and LangChain's `WebBaseLoader` to retrieve the contents of the top websites.
2. **Large Language Model**: Currently, two approaches are implemented:  
   - **RAG (Retrieve-Augmented Generation)**:  Uses VinAI's `PhoGPT` model, which is resource-intensive and has been run using Kaggle's hardware.  
   - **CAG (Cache-Augmented Generation)**: Leverages `Gemini 1.5`'s long-context window and caching capabilities. Instead of being chunked and retrieved later, the crawled websites from WebBaseLoader are directly integrated into the cached context, combined with a custom system instruction.

## Usage
### 1. Clone the repo and install the required dependencies
```bash
git clone https://github.com/namnguyen2103/GoogleSearchLLM.git
cd GoogleSearchLLM
pip install -r requirements.txt
```

### 2. Set up environment variables
Make sure you have a `.env` file in the root folder containing the necessary keys (instructions to create the Google search API key can be found [here](https://www.youtube.com/watch?v=4YhxXRPKI0c)
```
GOOGLE_API_KEY=
SEARCH_KEY=
GEMINI_API_KEY=
```

### 3. Run the application
``` bash
python src/app.py
```
or if you want to chat using Vietnamese:
``` bash
python src/vn_chat.py
```

### 4. Interact with the chatbot
Once the application launches, open the Gradio interface in your web browser (the default URL is usually http://127.0.0.1:7860.) 

Start by entering a search query to initialize the context, and then you can continue chatting about the topic you searched for or asking follow-up questions.

## Future direction
### 1. Google Search
- [ ] Use LLMs to enhance the quality of search queries.
- [ ] Add function-calling features to dynamically switch between search and chat for each query, rather than limiting searches to the initial message.

### 2. Website crawling
- [ ] Explore methods to process raw HTML files from various websites, especially for sites with Cloudflare protection that cannot be crawled using WebBaseLoader.
- [ ] Integrate a vector database to store queries and crawled website data.

### 3. RAG 
- [ ] Fine-tune a custom LLM model instead of using the pretrained `PhoGPT` model.

### 4. CAG 
- [ ] Explore the integration of other media types, such as images or videos, for enhanced conversational ability.
- [ ] Conduct prompt engineering for additional languages beyond English and Vietnamese.
