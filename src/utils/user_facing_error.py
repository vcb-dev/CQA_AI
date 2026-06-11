import re
from typing import Optional

TECHNICAL_PATTERNS = [
    re.compile(r"prisma", re.I),
    re.compile(r"invocation", re.I),
    re.compile(r"can't reach database", re.I),
    re.compile(r"AuthenticationError", re.I),
    re.compile(r"Invalid API key", re.I),
    re.compile(r"Incorrect API key", re.I),
    re.compile(r"RateLimitError", re.I),
    re.compile(r"APITimeoutError", re.I),
    re.compile(r"ConnectTimeout", re.I),
    re.compile(r"ReadTimeout", re.I),
    re.compile(r"Connection error", re.I),
    re.compile(r"Traceback", re.I),
    re.compile(r"File \"/", re.I),
]


def is_technical_error(message: str) -> bool:
    msg = (message or "").strip()
    if not msg:
        return False
    return any(p.search(msg) for p in TECHNICAL_PATTERNS)


def to_user_facing_error(raw: Optional[str]) -> str:
    msg = (raw or "").strip()
    if not msg:
        return "Đã có lỗi xảy ra. Vui lòng thử lại sau."

    lower = msg.lower()

    if "invalid api key" in lower or "incorrect api key" in lower or "authentication" in lower:
        return "Dịch vụ AI chưa được cấu hình đúng. Liên hệ quản trị hệ thống."
    if "rate limit" in lower or "429" in msg:
        return "Dịch vụ AI đang quá tải. Vui lòng thử lại sau vài phút."
    if "timeout" in lower or "timed out" in lower:
        return "Dịch vụ AI phản hồi quá chậm. Vui lòng thử lại."
    if "connection" in lower or "connect" in lower:
        return "Không kết nối được dịch vụ AI. Vui lòng thử lại sau."

    if is_technical_error(msg) or len(msg) > 220:
        return "Trợ lý tạm thời không phản hồi được. Vui lòng thử lại sau vài phút."

    return msg
