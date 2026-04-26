"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class StartConsultationRequest(BaseModel):
    """Request to start a new consultation."""
    user_name: Optional[str] = Field(None, description="Optional patient name")


class StartConsultationResponse(BaseModel):
    """Response after creating a consultation session."""
    thread_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)


class SendMessageRequest(BaseModel):
    """Request to send a message in a consultation."""
    content: str = Field(..., min_length=1, description="Message content")


class SSEEvent(BaseModel):
    """SSE event structure."""
    event: str = Field(..., description="Event type: thinking, message, complete")
    data: dict = Field(..., description="Event data")


class MessageResponse(BaseModel):
    """Message response in consultation."""
    content: str
    is_complete: bool = False


class CaseResponse(BaseModel):
    """Response containing the generated case."""
    case: str = Field(..., description="Generated case in markdown format")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str


class AuthRequest(BaseModel):
    """Request to register or log in."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class UserResponse(BaseModel):
    """Public user profile."""
    id: str
    username: str
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    """Auth response with bearer token."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
