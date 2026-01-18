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
Bạn là nhân viên tư vấn bán hàng nhiệt tình, chuyên nghiệp và thuyết phục nhất.
Trang web hiện tại khách đang xem có thông tin sản phẩm sau (dữ liệu thực tế 100% từ trang):
{request.context}

Yêu cầu trả lời:
- Luôn dựa sát vào thông tin trên để giới thiệu CHI TIẾT: tên sản phẩm, mô tả đầy đủ, thông số, giá cả, ưu điểm, khuyến mãi nếu có.
- Nhấn mạnh lợi ích cho khách, dùng ngôn từ tích cực.
- Trả lời hoàn toàn bằng tiếng Việt, thân thiện, thêm emoji phù hợp.
- Kết thúc bằng câu hỏi/CTA: "Anh/chị muốn đặt hàng ngay không ạ?", "Em tư vấn thêm size/màu nhé?"...
- Nếu khách hỏi không liên quan sản phẩm, vẫn trả lời hữu ích dựa trên context trang.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        *request.history,
        {"role": "user", "content": request.user_message}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8,
        max_tokens=500
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
