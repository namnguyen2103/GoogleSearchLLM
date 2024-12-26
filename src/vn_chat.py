import os
import gradio as gr
from websearch import search_google
import google.generativeai as genai
from google.generativeai import caching
import datetime
from datetime import date
from dotenv import load_dotenv
from io import StringIO
import contextlib

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Helper function
def get_context(query: str, topk: int = 10, lan: str = 'vi', **params):
    output_capture = StringIO()
    with contextlib.redirect_stdout(output_capture): 
        docs = search_google(query, topk, lan, **params)
    printed_urls = output_capture.getvalue().strip() 
    doc_string = "\n\n".join(
        f"URL {index + 1}\n"
        f"Source: {doc.metadata.get('source', 'N/A')}\n"
        f"Title: {doc.metadata.get('title', 'N/A')}\n"
        f"Description: {doc.metadata.get('description', 'N/A')}\n"
        f"Content: {doc.page_content}\n"
        for index, doc in enumerate(docs)
    )

    return printed_urls, doc_string

current_date = date.today().strftime("ngày %d tháng %m năm %Y")

system_instruction = f"""
### Vai Trò:
Bạn là một trợ lý thông minh được thiết kế để cung cấp câu trả lời chính xác và ngắn gọn bằng cách tận dụng nội dung mới nhất từ các trang web được lấy thời gian thực thông qua Google. Mục tiêu chính của bạn là hỗ trợ người dùng có được thông tin và hiểu biết liên quan dựa trên các nguồn đã truy xuất. Hãy trả lời hoàn toàn bằng tiếng Việt.

### Hướng Dẫn:

1. **Hiểu Ngữ Cảnh**:
   - Phân tích các URL, tiêu đề, mô tả và nội dung được cung cấp từ kết quả tìm kiếm.
   - Trích xuất và ưu tiên thông tin quan trọng và hữu ích nhất cho câu hỏi của người dùng.
   - **Thông tin nhạy cảm về thời gian có thể được hỏi, hãy coi ngữ cảnh đã cung cấp là nguồn mới nhất và thời gian thực.**
   - Để tham khảo, hôm nay là {current_date}.

2. **Ưu Tiên Thông Tin**:
   - Tập trung trả lời câu hỏi của người dùng một cách ngắn gọn trong khi đảm bảo câu trả lời chính xác.
   - Tránh đưa vào những chi tiết không cần thiết hoặc không mang lại giá trị cho câu trả lời.
   - Nếu thông tin khác nhau, hãy cung cấp sự so sánh giữa các nguồn và rõ ràng chỉ ra bất kỳ sự khác biệt nào.

3. **Tích Hợp Nhiều Nguồn**:
   - Nếu nhiều nguồn được cung cấp, kết hợp thông tin một cách logic, tránh lặp lại và đảm bảo sự rõ ràng.
   - Nếu các nguồn cung cấp thông tin mâu thuẫn, **hãy chỉ rõ sự khác biệt và nhấn mạnh bất kỳ sự mâu thuẫn hoặc biến thể nào**, giải thích các lý do có thể gây ra những khác biệt này.

4. **Xử Lý Khi Không Có Thông Tin Cụ Thể**:
   - Nếu không có dữ liệu trực tiếp trả lời câu hỏi, **hãy đưa ra dự đoán hợp lý hoặc gợi ý thông tin hữu ích dựa trên ngữ cảnh**.
   - Ví dụ: *Không có thông tin chi tiết về thời tiết Hà Nội hôm nay, nhưng dựa trên xu hướng mùa, có thể dự đoán trời se lạnh với khả năng mưa nhẹ.*
   - Tuyệt đối tránh câu trả lời như "Tôi xin lỗi, không có thông tin" trừ khi không thể đưa ra bất kỳ suy đoán hoặc gợi ý nào dựa trên ngữ cảnh.

5. **Ngôn Ngữ và Sự Rõ Ràng**:
   - Sử dụng ngôn ngữ trang trọng nhưng dễ tiếp cận, phù hợp với nhiều đối tượng.
   - Tránh sử dụng thuật ngữ chuyên môn trừ khi người dùng yêu cầu các thuật ngữ kỹ thuật.

6. **Xử Lý Lỗi**:
   - Nếu không tìm thấy dữ liệu liên quan hoặc câu hỏi không rõ ràng, hãy yêu cầu người dùng làm rõ hoặc giải thích giới hạn của hệ thống.
   - **Tránh yêu cầu người dùng tìm kiếm ở nơi khác trừ khi thực sự cần thiết**. Cố gắng cung cấp câu trả lời đầy đủ nhất có thể dựa trên ngữ cảnh hiện có.
   - Trong trường hợp yêu cầu dữ liệu thời gian thực mà không có sẵn đầy đủ, hãy giải thích ngữ cảnh tốt nhất hiện có từ các nguồn bạn có. Ví dụ: *Dựa trên thông tin có sẵn hôm nay, giá dao động từ X đến Y, nhưng có thể thay đổi nhanh chóng.*
"""

MODEL_NAME = "models/gemini-1.5-flash-002"
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 8192
}

def search_and_cache(query):
    printed_urls, context = get_context(query)
    cache = caching.CachedContent.create(
        model=MODEL_NAME,
        system_instruction=(system_instruction),
        contents=[context],
        ttl=datetime.timedelta(minutes=15),
    )
    model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    return model, printed_urls

class ChatApp:
    def __init__(self):
        self.model = None
        self.gemini_history = []

    def process_query(self, message, history):
        if self.model is None:  # First query initializes the model
            self.model, urls = search_and_cache(message)   
            response_content = f"{urls}\n\nNgữ cảnh đã được khởi tạo. Bạn có thể tiếp tục trò chuyện."
        else:  # Subsequent queries
            self.gemini_history.append({"role": "user", "parts": [message]})
            response = self.model.generate_content(self.gemini_history)
            response_content = response.text
            self.gemini_history.append({"role": "model", "parts": [response_content]})
        return response_content

app = ChatApp()

demo = gr.ChatInterface(
    fn=app.process_query,
    type="messages",
    title="Google Search Chatbot",
    description="Start with a Google search query to initialize context, then continue chatting with the assistant.",
)

if __name__ == "__main__":
    demo.launch()