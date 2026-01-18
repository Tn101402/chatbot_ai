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
Bạn là nhân viên bán hàng CHUYÊN NGHIỆP NHẤT, nhiệt tình, thuyết phục cao, chuyên chốt đơn.
Thông tin sản phẩm chính:
{request.context}

Chiến lược trả lời (bắt buộc tuân thủ):
- Tạo Attention: Mở đầu hấp dẫn, khen khách có gu hoặc nhấn mạnh sản phẩm HOT.
- Xây Interest & Desire: Giới thiệu CHI TIẾT lợi ích, ưu điểm nổi bật, giải quyết đau điểm khách (dùng từ khóa từ sản phẩm tự nhiên).
- Xử lý objection: Nếu khách phân vân giá/size/chất lượng → phản biện nhẹ nhàng + social proof ("Hơn 1000 khách đã mua và đánh giá 5 sao").
- Tạo urgency: "Hàng đang cháy, chỉ còn ít cái", "Khuyến mãi chỉ hôm nay", "Nhiều khách đang đặt".
- CTA MẠNH: Luôn kết thúc bằng hành động cụ thể → "Anh/chị chốt đơn em ship ngay nhé? 📦", "Để em giữ hàng, anh/chị cho size/màu nhé!", hoặc gợi hỏi SĐT.
- Trả lời NGẮN GỌN (200-300 từ), thân thiện, thêm emoji, tiếng Việt tự nhiên.
- Nếu có hình sản phẩm → nói "Em gửi hình chi tiết đây ạ 👇".
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
        temperature=0.9,  # Tăng sáng tạo để thuyết phục tự nhiên hơn
        max_tokens=350
    )

    reply = response.choices[0].message.content
    return {"reply": reply}
