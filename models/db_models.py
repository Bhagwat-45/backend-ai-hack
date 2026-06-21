import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=_now, nullable=False)

    # Output of extract_simulation_parameters(), as a JSON string.
    # This is what /simulate/run and /simulate/whatif now read instead of
    # current_params.json, and what the chat feature summarizes into a
    # prompt instead of dumping raw event rows.
    parameters_json = Column(Text, nullable=False)

    conversations = relationship(
        "Conversation", back_populates="event_log", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=_uuid)
    event_log_id = Column(Integer, ForeignKey("event_logs.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)

    event_log = relationship("EventLog", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender_type = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")