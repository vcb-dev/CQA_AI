import json
import os
import traceback
from typing import List

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.customer_intent_models import CustomerIntentRequest, CustomerIntentResponse


class CustomerIntentService:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1,
        )

    def _format_lines(self, messages: List[dict]) -> str:
        msgs = messages
        if len(msgs) > 100:
            msgs = msgs[:15] + msgs[-80:]
        lines = []
        for m in msgs:
            sender = (m.get("sender") or "?").strip()
            text = (m.get("text") or "").strip()
            if not text or text in ("[Ảnh]", "[Sticker]", "[attachment]", "[Video]"):
                text = "[Gửi ảnh/media]"
            lines.append(f"{sender}: {text}")
        return "\n".join(lines)

    async def analyze(self, req: CustomerIntentRequest) -> CustomerIntentResponse:
        transcript = self._format_lines([m.model_dump() for m in req.messages])
        if not transcript.strip():
            return CustomerIntentResponse(
                summary="Chưa có nội dung tin nhắn để phân tích.",
                intent_label="Chưa rõ",
                topics=[],
                product_mentions=[],
                urgency="low",
                suggested_focus="Chờ khách nhắn thêm.",
            )

        customer = (req.customer_name or "Khách").strip()
        schemas = [
            ResponseSchema(name="summary", description="Tóm tắt 1-2 câu bằng tiếng Việt"),
            ResponseSchema(
                name="intent_label",
                description="Nhãn ngắn: hỏi giá, hỏi size, đổi trả, khiếu nại, chốt đơn, tư vấn SP, hỏi ship, khác",
            ),
            ResponseSchema(
                name="topics",
                description='JSON array string các chủ đề/SP khách quan tâm, VD: ["áo thun", "size M"]',
            ),
            ResponseSchema(
                name="product_mentions",
                description=(
                    'JSON array string — tên/mô tả SP cụ thể khách nhắc trong TOÀN BỘ hội thoại '
                    '(vd: ["nhẫn tàng hình bạc", "vòng kim hoàn"]). Không bịa tên không xuất hiện.'
                ),
            ),
            ResponseSchema(name="urgency", description="low, normal, hoặc high"),
            ResponseSchema(
                name="suggested_focus",
                description="1 câu gợi ý NV nên trả lời tập trung vào đâu",
            ),
            ResponseSchema(
                name="suggested_reply",
                description="Dự thảo tin nhắn phản hồi chi tiết, lịch sự, xưng hô phù hợp với ngữ cảnh (dùng từ 'shop' và 'bạn'/'anh'/'chị') để gửi trực tiếp cho khách.",
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(schemas)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là trợ lý CSKH thời gian thực. Đọc TOÀN BỘ hội thoại Facebook Messenger "
                    "(shop thời trang/trang sức) để xác định khách đang cần gì, "
                    "SP quan tâm, mức độ gấp, gợi ý hướng xử lý của NV (suggested_focus) "
                    "và đặc biệt dự thảo tin nhắn trả lời trực tiếp cho khách hàng (suggested_reply). "
                    "suggested_reply cần viết cực kỳ lịch sự, thân thiện, trả lời đúng trọng tâm câu hỏi cuối của khách, "
                    "xưng hô 'shop' và gọi khách là 'bạn/dạ anh/dạ chị' thích hợp. "
                    "Chỉ trả JSON đúng format. Tiếng Việt tự nhiên.",
                ),
                (
                    "human",
                    "Khách: {customer_name}\n\nToàn bộ hội thoại:\n{transcript}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.llm | parser

        try:
            parsed = await chain.ainvoke(
                {
                    "customer_name": customer,
                    "transcript": transcript,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            topics_raw = parsed.get("topics") or "[]"
            if isinstance(topics_raw, str):
                try:
                    topics = json.loads(topics_raw)
                except json.JSONDecodeError:
                    topics = [t.strip() for t in topics_raw.split(",") if t.strip()]
            elif isinstance(topics_raw, list):
                topics = [str(t).strip() for t in topics_raw if str(t).strip()]
            else:
                topics = []

            mentions_raw = parsed.get("product_mentions") or "[]"
            if isinstance(mentions_raw, str):
                try:
                    product_mentions = json.loads(mentions_raw)
                except json.JSONDecodeError:
                    product_mentions = [t.strip() for t in mentions_raw.split(",") if t.strip()]
            elif isinstance(mentions_raw, list):
                product_mentions = [str(t).strip() for t in mentions_raw if str(t).strip()]
            else:
                product_mentions = []

            urgency = str(parsed.get("urgency") or "normal").lower()
            if urgency not in ("low", "normal", "high"):
                urgency = "normal"

            return CustomerIntentResponse(
                summary=str(parsed.get("summary") or "").strip() or "Khách vừa nhắn tin.",
                intent_label=str(parsed.get("intent_label") or "Chưa rõ").strip(),
                topics=topics[:8],
                product_mentions=product_mentions[:10],
                urgency=urgency,
                suggested_focus=str(parsed.get("suggested_focus") or "").strip(),
                suggested_reply=str(parsed.get("suggested_reply") or "").strip(),
            )
        except Exception:
            traceback.print_exc()
            return CustomerIntentResponse(
                summary="Không phân tích được tin nhắn lúc này.",
                intent_label="Chưa rõ",
                topics=[],
                product_mentions=[],
                urgency="normal",
                suggested_focus="Xem tin nhắn mới và phản hồi trực tiếp.",
                suggested_reply="",
            )
