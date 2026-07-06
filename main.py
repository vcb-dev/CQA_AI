import os
from typing import Optional, List
from contextlib import asynccontextmanager

import logging
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.services.audit_service import AuditService
from src.services.assistant_service import AssistantService
from src.services.customer_intent_service import CustomerIntentService
from src.services.deepseek_balance_service import DeepSeekBalanceService
from src.models.customer_intent_models import CustomerIntentRequest, CustomerIntentResponse
from src.models.assistant_models import AssistantChatRequest, AssistantChatResponse
from src.utils.user_facing_error import to_user_facing_error

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))

DEFAULT_ASSISTANT_LLM_MODEL = "deepseek-chat"
logger = logging.getLogger("cqa_ai")


def get_assistant_model() -> str:
    return (os.getenv("ASSISTANT_LLM_MODEL") or DEFAULT_ASSISTANT_LLM_MODEL).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        logger.warning("DEEPSEEK_API_KEY / OPENAI_API_KEY chưa cấu hình — LLM sẽ lỗi")
    else:
        logger.info("LLM API key configured; model=%s", get_assistant_model())
    yield


app = FastAPI(title="CQA CRM - AI Service", lifespan=lifespan)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Services ─────────────────────────────────────────────────────────────────
audit_service = AuditService()
assistant_service = AssistantService()
customer_intent_service = CustomerIntentService()
deepseek_balance_service = DeepSeekBalanceService()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_llm_mode() -> str:
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "fallback"


def assert_assistant_secret(x_assistant_secret: Optional[str]) -> None:
    expected = os.getenv("ASSISTANT_INTERNAL_SECRET", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="ASSISTANT_INTERNAL_SECRET chưa cấu hình trên AI service",
        )
    if not x_assistant_secret or x_assistant_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid assistant secret")


# ─── Exception Handlers ───────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("422 %s %s — %s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, str) and len(detail) > 220:
        detail = to_user_facing_error(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("500 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": to_user_facing_error(str(exc))},
    )


# ─── Startup (Handled by lifespan) ───────────────────────────────────────────


class ChatMessage(BaseModel):
    sender: str
    text: str
    timestamp: Optional[str] = None

class AuditRequest(BaseModel):
    transcript: List[ChatMessage]
    no_reply: Optional[bool] = False
    agent_name: Optional[str] = None
    customer_name: Optional[str] = None

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "CQA CRM AI Service is running",
        "endpoints": [
            "/audit",
            "/cskh/customer-intent",
            "/deepseek/balance",
            "/assistant/chat",
            "/assistant/health",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    llm_mode = get_llm_mode()
    return {
        "ok": True,
        "service": "CQA_AI",
        "llm": llm_mode,
        "model": get_assistant_model() if llm_mode != "fallback" else None,
    }


@app.get("/assistant/health")
async def assistant_health():
    return {"status": "ok", "service": "cqa-assistant"}


@app.post("/assistant/chat", response_model=AssistantChatResponse)
async def assistant_chat(
    request: AssistantChatRequest,
    x_assistant_secret: Optional[str] = Header(default=None),
):
    assert_assistant_secret(x_assistant_secret)
    return await assistant_service.chat(request)


@app.get("/deepseek/balance")
async def deepseek_balance():
    return await deepseek_balance_service.get_balance()


@app.post("/cskh/customer-intent", response_model=CustomerIntentResponse)
async def cskh_customer_intent(request: CustomerIntentRequest):
    return await customer_intent_service.analyze(request)


@app.post("/audit")
async def audit_chat(request: AuditRequest):
    transcript_list = [m.model_dump() for m in request.transcript]
    result = await audit_service.audit_transcript(
        transcript_list,
        no_reply=request.no_reply or False,
        agent_name=request.agent_name,
        customer_name=request.customer_name,
    )
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
