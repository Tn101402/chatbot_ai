from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    context: str
    user_message: str
    history: list[dict] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = f"""
Bạn là nhân viên bán hàng SIÊU CHI TIẾT, thuyết phục và chuyên chốt đơn.
Thông tin sản phẩm đầy đủ:
{request.context}

Chiến lược:
- Trả lời hấp dẫn, xử lý objection, tạo urgency, social proof.
- CTA mạnh ở cuối nội dung chính.
- BẮT BUỘC kết thúc reply bằng đúng 1 dòng định dạng:
[Quick: Gợi ý câu hỏi 1 | Gợi ý câu hỏi 2 | Gợi ý câu hỏi 3 | Gợi ý câu hỏi 4]
(Gợi ý phải relevant với conversation, ngắn gọn, giúp khách hỏi nhanh để chốt đơn. Ví dụ: Có size M không? | Giá khuyến mãi? | Chốt đơn màu đỏ | Gửi thêm hình)
- Trả lời tiếng Việt tự nhiên, thêm emoji.
""".strip()

    limited_history = request.history[-8:]

    messages = [
        {"role": "system", "content": system_prompt},
        *limited_history,
        {"role": "user", "content": request.user_message}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.9,
        max_tokens=450
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
