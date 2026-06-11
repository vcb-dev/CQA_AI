from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerIntentMessage(BaseModel):
    sender: str
    text: str


class CustomerIntentRequest(BaseModel):
    messages: List[CustomerIntentMessage]
    customer_name: Optional[str] = None


class CustomerIntentResponse(BaseModel):
    summary: str = Field(description="Tóm tắt 1-2 câu khách đang cần gì")
    intent_label: str = Field(description="Nhãn ngắn: hỏi giá, size, đổi trả, khiếu nại, chốt đơn, …")
    topics: List[str] = Field(default_factory=list, description="Chủ đề / sản phẩm khách quan tâm")
    product_mentions: List[str] = Field(
        default_factory=list,
        description="Tên/mô tả sản phẩm khách nhắc trong hội thoại (để tra catalog)",
    )
    urgency: str = Field(default="normal", description="low | normal | high")
    suggested_focus: str = Field(
        default="",
        description="Gợi ý ngắn cho NV nên tập trung trả lời gì",
    )
