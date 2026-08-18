from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatHistoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class UserContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str = Field(..., min_length=1)
    email: Optional[str] = None
    full_name: Optional[str] = None
    app_role: str = "USER"
    department: Optional[str] = None
    permission_ids: List[str] = Field(default_factory=list)

    @field_validator("app_role", mode="before")
    @classmethod
    def default_app_role(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "USER"
        return str(v)


class ConversationMessageItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sender: str = Field(..., min_length=1, max_length=40)
    text: str = Field(..., min_length=1, max_length=4000)


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_id: Optional[str] = None
    customer_name: Optional[str] = None
    platform: Optional[str] = None
    page_name: Optional[str] = None
    from_ad: Optional[bool] = None
    labels: List[str] = Field(default_factory=list)
    recent_messages: List[ConversationMessageItem] = Field(default_factory=list, max_length=24)

    @field_validator("recent_messages", mode="before")
    @classmethod
    def trim_recent(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return v[-24:]


class AssistantChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatHistoryItem] = Field(default_factory=list, max_length=20)
    user_context: UserContext
    session_facts: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: Optional[ConversationContext] = None

    @field_validator("history", mode="before")
    @classmethod
    def trim_history(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return v[-20:]


class SourceItem(BaseModel):
    title: str
    snippet: str


class AssistantChatResponse(BaseModel):
    reply: str
    blocked: bool = False
    block_reason: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
    scope: Literal["in_scope", "off_topic", "low_confidence"] = "in_scope"
