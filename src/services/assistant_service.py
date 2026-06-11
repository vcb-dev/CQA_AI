import os
import logging
from typing import Optional

from src.models.assistant_models import (
    AssistantChatRequest,
    AssistantChatResponse,
    SourceItem,
)

logger = logging.getLogger("cqa_ai.assistant")


class AssistantService:
    """
    Dịch vụ trợ lý AI chính cho CQA CRM.
    Xử lý chat, trả lời câu hỏi dựa trên ngữ cảnh người dùng.
    """

    def __init__(self):
        self._llm = None
        self._setup_llm()

    def _setup_llm(self):
        """Khởi tạo LLM dựa trên API key được cấu hình."""
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
                    temperature=0.3,
                    max_tokens=2048,
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
                    temperature=0.3,
                    max_tokens=2048,
                )
                logger.info("Sử dụng OpenAI LLM: %s", model)
            except Exception as e:
                logger.error("Lỗi khởi tạo OpenAI: %s", e)
        else:
            logger.warning("Không tìm thấy API key — assistant sẽ trả về lỗi khi chat")

    async def chat(self, request: AssistantChatRequest) -> AssistantChatResponse:
        """
        Xử lý một lượt chat từ người dùng.
        """
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

            # Thêm lịch sử hội thoại
            for item in request.history:
                if item.role == "user":
                    messages.append(HumanMessage(content=item.content))
                else:
                    messages.append(AIMessage(content=item.content))

            # Thêm tin nhắn hiện tại
            messages.append(HumanMessage(content=request.message))

            response = await self._llm.ainvoke(messages)
            reply_text = response.content if hasattr(response, "content") else str(response)

            return AssistantChatResponse(
                reply=reply_text,
                blocked=False,
                sources=[],
                scope="in_scope",
            )

        except Exception as e:
            logger.exception("Lỗi khi gọi LLM")
            from src.utils.user_facing_error import to_user_facing_error
            return AssistantChatResponse(
                reply=to_user_facing_error(str(e)),
                blocked=True,
                block_reason="llm_error",
                scope="in_scope",
            )

    def _build_system_prompt(self, request: AssistantChatRequest) -> str:
        """Tạo system prompt dựa trên ngữ cảnh người dùng."""
        user = request.user_context
        return (
            "Bạn là trợ lý AI thông minh của hệ thống CQA CRM. "
            "Hãy trả lời ngắn gọn, chính xác và hữu ích bằng tiếng Việt.\n\n"
            f"Thông tin người dùng:\n"
            f"- Họ tên: {user.full_name or 'Không rõ'}\n"
            f"- Email: {user.email or 'Không rõ'}\n"
            f"- Vai trò: {user.app_role}\n"
            f"- Phòng ban: {user.department or 'Không rõ'}\n"
        )
