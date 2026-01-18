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
    # Rút gọn context đầu vào (đã xử lý ở frontend)
    system_prompt = f"""
Bạn là nhân viên tư vấn bán hàng nhiệt tình, chuyên nghiệp.
Thông tin sản phẩm chính (từ trang hiện tại):
{request.context}

Yêu cầu:
- Trả lời CHI TIẾT nhưng NGẮN GỌN (tối đa 300 từ), dùng từ khóa tự nhiên từ sản phẩm.
- Nhấn mạnh lợi ích, ưu điểm, giá trị.
- Luôn kết thúc bằng CTA mạnh: hỏi thêm thông tin khách hoặc gợi đặt hàng.
- Nếu có link liên quan, gợi ý tự nhiên: "Anh/chị xem thêm sản phẩm tương tự tại đây nhé".
- Trả lời 100% tiếng Việt, thân thiện, thêm emoji phù hợp.
""".strip()

    # Giới hạn history chỉ 6 tin nhắn gần nhất → tiết kiệm token
    limited_history = request.history[-6:]

    messages = [
        {"role": "system", "content": system_prompt},
        *limited_history,
        {"role": "user", "content": request.user_message}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8,
        max_tokens=400  # Giới hạn output token
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
