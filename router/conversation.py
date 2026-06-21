"""
Conversation endpoints - create a conversation tied to an uploaded event
log, and fetch its message history. Python/FastAPI equivalent of the old
.NET ConversationController.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.db_models import EventLog, Conversation

router = APIRouter(prefix="/api/conversation", tags=["conversation"])


class CreateConversationRequest(BaseModel):
    event_log_id: int
    title: Optional[str] = None


class MessageOut(BaseModel):
    sender_type: str
    content: str
    created_at: str


class ConversationOut(BaseModel):
    id: str
    event_log_id: int
    title: str
    created_at: str


@router.post("", response_model=ConversationOut)
def create_conversation(request: CreateConversationRequest, db: Session = Depends(get_db)):
    event_log = db.query(EventLog).filter(EventLog.id == request.event_log_id).first()
    if not event_log:
        raise HTTPException(status_code=404, detail="event_log_id not found. Upload a log first.")

    conversation = Conversation(
        event_log_id=request.event_log_id,
        title=request.title or f"Risk Analysis - {event_log.filename}",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationOut(
        id=conversation.id,
        event_log_id=conversation.event_log_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
    )


@router.get("/{conversation_id}", response_model=List[MessageOut])
def get_history(conversation_id: str, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return [
        MessageOut(
            sender_type=m.sender_type,
            content=m.content,
            created_at=m.created_at.isoformat(),
        )
        for m in conversation.messages
    ]