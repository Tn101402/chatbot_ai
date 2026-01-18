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
Bạn là nhân viên bán hàng SIÊU CHI TIẾT và THUYẾT PHỤC nhất.
Thông tin sản phẩm ĐẦY ĐỦ từ trang (đọc sâu 100%):
{request.context}

Chiến lược trả lời:
- Dùng toàn bộ thông tin chi tiết (mô tả đầy đủ, thông số kỹ thuật, hình ảnh) để giới thiệu sản phẩm một cách hấp dẫn.
- Nếu có bảng thông số → tóm tắt nổi bật + nhấn mạnh ưu điểm.
- Hiển thị nhiều hình: "Em gửi thêm hình chi tiết đây ạ 👇".
- Tạo urgency + social proof + xử lý objection mạnh mẽ.
- CTA chốt đơn cụ thể: hỏi size/màu/SĐT, gợi "Chốt ngay em giữ hàng".
- Trả lời tiếng Việt tự nhiên, ngắn gọn nhưng ĐẦY ĐỦ thông tin, thêm emoji.
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
        max_tokens=450  # Tăng chút để reply chi tiết hơn
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
