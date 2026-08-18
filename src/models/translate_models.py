from typing import List, Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Nội dung cần dịch")
    source_lang: Optional[str] = Field(
        default="auto",
        description="Mã ngôn ngữ nguồn (vi, th, en, zh, ...) hoặc auto",
    )
    target_lang: str = Field(
        default="vi",
        description="Mã ngôn ngữ đích (vi, th, en, zh, ...)",
    )
    direction: Optional[str] = Field(
        default=None,
        description="inbound (khách) | outbound (shop) — ảnh hưởng giọng dịch",
    )
    context_messages: Optional[List[str]] = Field(
        default=None,
        description="Vài tin gần đây trong cùng hội thoại để dịch đúng ngữ cảnh",
    )


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    detected_lang: str
    source_lang: str
    target_lang: str
    same_language: bool = False


class TranslateBatchItem(BaseModel):
    id: str = Field(..., description="ID tin nhắn (BE)")
    text: str = Field(..., min_length=1)
    direction: Optional[str] = Field(default=None, description="inbound | outbound")


class TranslateBatchRequest(BaseModel):
    items: List[TranslateBatchItem] = Field(..., min_length=1, max_length=24)
    target_lang: str = Field(default="vi")
    context_messages: Optional[List[str]] = Field(default=None)


class TranslateBatchResultItem(BaseModel):
    id: str
    original_text: str
    translated_text: str
    detected_lang: str
    same_language: bool = False


class TranslateBatchResponse(BaseModel):
    items: List[TranslateBatchResultItem]
    target_lang: str


class DetectLangRequest(BaseModel):
    texts: List[str] = Field(
        default_factory=list,
        description="Toàn bộ / nhiều tin hội thoại để phát hiện ngôn ngữ khách",
    )


class DetectLangResponse(BaseModel):
    lang: str
    lang_label: str
    confidence: str = "medium"
