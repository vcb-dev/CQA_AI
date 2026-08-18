import logging
import re
from typing import List, Optional

from src.models.assistant_models import (
    AssistantChatRequest,
    AssistantChatResponse,
)
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.services.knowledge_base import KnowledgeBase
from src.services.rag_store import RAGStore

logger = logging.getLogger("cqa_ai.assistant")

_OUT_OF_SCOPE_MARKERS = (
    "chưa có kỹ năng",
    "chua co ky nang",
    "không thuộc chuyên môn",
    "ngoài chuyên môn",
)

# Cả dòng/câu chứa từ này bị xóa khỏi câu trả lời — không chỉ xóa đúng chữ.
_LEAK_LINE_RE = re.compile(
    r"(sop|checklist|quy\s*tắc|bước\s*\d+|bước vàng|bước bắt buộc|"
    r"đúng chuẩn|đúng sop|theo quy tắc|tài liệu nội bộ|lý do theo|"
    r"nhận xét nhanh|5 bước|khai thác nhu cầu \(\*\*)",
    re.IGNORECASE,
)


def _is_out_of_scope_reply(text: str) -> bool:
    lower = (text or "").lower()
    return any(m in lower for m in _OUT_OF_SCOPE_MARKERS)


def _scrub_visible_reply(text: str) -> str:
    """Xóa dòng giảng nội bộ. Giữ khối <<OPTION>>."""
    if not text:
        return text
    chunks = re.split(r"(<<\s*OPTION\s*>>)", text, flags=re.IGNORECASE)
    cleaned: List[str] = []
    in_option = False
    for chunk in chunks:
        if re.fullmatch(r"<<\s*OPTION\s*>>", chunk, flags=re.IGNORECASE):
            cleaned.append("<<OPTION>>")
            in_option = True
            continue
        if in_option:
            cleaned.append(chunk)
            continue
        lines = []
        for line in chunk.splitlines():
            if _LEAK_LINE_RE.search(line):
                continue
            if re.match(r"^[\s]*[✅⚠️✔✘]", line):
                continue
            lines.append(line)
        cleaned.append("\n".join(lines))
    out = "".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class AssistantService:
    """Agent: nhận câu hỏi NV → trả lời. Quy tắc chỉ học im lặng, không đọc ra."""

    def __init__(
        self,
        knowledge: Optional[KnowledgeBase] = None,
        rag: Optional[RAGStore] = None,
    ):
        self._llm = None
        self._knowledge = knowledge or KnowledgeBase()
        self._rag = rag or RAGStore()
        self._setup_llm()

    def _setup_llm(self):
        import os

        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("ASSISTANT_LLM_MODEL", "deepseek-chat").strip()

        if deepseek_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=model,
                    api_key=deepseek_key,
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    temperature=0.15,
                    max_tokens=900,
                )
                logger.info("Sử dụng DeepSeek LLM: %s", model)
            except Exception as e:
                logger.error("Lỗi khởi tạo DeepSeek: %s", e)
        elif openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=model,
                    api_key=openai_key,
                    temperature=0.15,
                    max_tokens=900,
                )
                logger.info("Sử dụng OpenAI LLM: %s", model)
            except Exception as e:
                logger.error("Lỗi khởi tạo OpenAI: %s", e)
        else:
            logger.warning("Không tìm thấy API key — assistant sẽ trả về lỗi khi chat")

    def load_knowledge(self) -> None:
        try:
            n = self._knowledge.load()
            logger.info("Assistant knowledge loaded: %s chunks", n)
        except Exception:
            logger.exception("Không load được knowledge base")
        try:
            self._rag.load()
        except Exception:
            logger.exception("Không load được FAISS index")

    async def chat(self, request: AssistantChatRequest) -> AssistantChatResponse:
        if self._llm is None:
            return AssistantChatResponse(
                reply="Dịch vụ AI chưa được cấu hình. Vui lòng liên hệ quản trị viên.",
                blocked=True,
                block_reason="no_llm_configured",
                scope="in_scope",
            )

        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            messages = [SystemMessage(content=self._build_system_prompt(request))]
            for item in request.history:
                if item.role == "user":
                    messages.append(HumanMessage(content=item.content))
                else:
                    messages.append(AIMessage(content=_scrub_visible_reply(item.content)))
            messages.append(HumanMessage(content=request.message))

            response = await self._llm.ainvoke(messages)
            reply_text = response.content if hasattr(response, "content") else str(response)
            if isinstance(reply_text, list):
                reply_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in reply_text
                )

            reply_text = _scrub_visible_reply(str(reply_text).strip())
            scope = "off_topic" if _is_out_of_scope_reply(reply_text) else "in_scope"
            return AssistantChatResponse(
                reply=reply_text,
                blocked=False,
                sources=[],
                scope=scope,
            )
        except Exception:
            logger.exception("Lỗi khi gọi LLM")
            return AssistantChatResponse(
                reply="Trợ lý tạm thời không phản hồi được. Vui lòng thử lại sau.",
                blocked=True,
                block_reason="llm_error",
                scope="in_scope",
            )

    def _build_system_prompt(self, request: AssistantChatRequest) -> str:
        user = request.user_context
        parts = [SYSTEM_PROMPT]

        parts.append(
            "Nhân viên:\n"
            f"- Tên: {user.full_name or 'Không rõ'}\n"
            f"- Vai trò: {user.app_role}\n"
        )

        ctx = request.conversation_context
        if ctx:
            labels = ", ".join(ctx.labels) if ctx.labels else "không có"
            parts.append(
                "Hội thoại đang mở:\n"
                f"- Khách: {ctx.customer_name or 'Không rõ'}\n"
                f"- Kênh: {ctx.platform or 'không rõ'} / {ctx.page_name or '—'}\n"
                f"- Nhãn: {labels}\n"
            )
            if ctx.recent_messages:
                lines = []
                for msg in ctx.recent_messages[-16:]:
                    who = "Khách" if msg.sender.lower() in {"customer", "khách", "user"} else "NV"
                    if msg.sender.lower() in {"staff", "agent", "nhân viên"}:
                        who = "NV"
                    lines.append(f"{who}: {msg.text.strip()[:500]}")
                parts.append("Tin gần đây:\n" + "\n".join(lines))

        parts.append(
            "Trả lời đúng câu hỏi. Không nhận xét quy trình. "
            "Nếu soạn tin: chỉ 2 khối <<OPTION>>."
        )
        return "\n\n".join(parts)
