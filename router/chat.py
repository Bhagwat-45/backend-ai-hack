"""
Chat endpoint - Python/FastAPI equivalent of the old .NET ChatController,
minus the context-blowup bug. Builds a bounded, summarized prompt
(services/chat_service.py) instead of dumping every raw event and every
past message into the request.
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.db_models import EventLog, Conversation, Message
from services.chat_service import build_messages, call_azure_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    model_name: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    event_log = db.query(EventLog).filter(
        EventLog.id == conversation.event_log_id
    ).first()
    if not event_log:
        raise HTTPException(status_code=404, detail="Linked event log not found.")

    params = json.loads(event_log.parameters_json)
    history = conversation.messages  # already ordered oldest -> newest

    messages = build_messages(params, history, request.message)

    try:
        ai_response = call_azure_chat(messages, model_name=request.model_name)
    except RuntimeError as e:
        # Clean, expected failure (missing/bad credentials, Azure-side
        # error) instead of an unhandled 500 with a stack trace.
        raise HTTPException(status_code=502, detail=str(e))

    db.add(Message(conversation_id=conversation.id, sender_type="user", content=request.message))
    db.add(Message(conversation_id=conversation.id, sender_type="assistant", content=ai_response))
    db.commit()

    return ChatResponse(response=ai_response)