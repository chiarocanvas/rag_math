import secrets
import json
import copy
from typing import Optional, List, Any, Dict, Union
from datetime import datetime, timezone

from sqlalchemy import create_engine, Integer, String, Text, MetaData, func, Column, Table, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped


metadata = MetaData()


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str]
    user_id: Mapped[Optional[int]]
    user_name: Mapped[Optional[str]]
    reply_user_id: Mapped[Optional[int]]
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    conv_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[Optional[int]]
    message_id: Mapped[Optional[int]]
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rag_promt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)



class Subject(Base):
    __tablename__ = "subjects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    subject: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class Conversation(Base):
    __tablename__ = "current_conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    conv_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    timestamp: Mapped[int]


class Like(Base):
    __tablename__ = "likes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    feedback: Mapped[str]
    is_correct: Mapped[int]


# Define a new table for temporary data
class TempData(Base):
    __tablename__ = "temp_data"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class Database:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def get_current_ts() -> int:
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())




    def create_conv_id(self, user_id: int) -> str:
        conv_id = secrets.token_hex(nbytes=16)
        with self.Session() as session:
            new_conv = Conversation(user_id=user_id, conv_id=conv_id, timestamp=self.get_current_ts())
            session.add(new_conv)
            session.commit()
        return conv_id

    def get_user_id_by_conv_id(self, conv_id: str) -> int:
        with self.Session() as session:
            conv = (
                session.query(Conversation)
                .filter(Conversation.conv_id == conv_id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )
            assert conv
            return conv.user_id

    def get_current_conv_id(self, user_id: int) -> str:
        with self.Session() as session:
            conv = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )
            return conv.conv_id if conv else self.create_conv_id(user_id)

    def fetch_conversation(self, conv_id: str) -> List[Any]:
        with self.Session() as session:
            messages = session.query(Message).filter(Message.conv_id == conv_id).order_by(Message.timestamp).all()
            if not messages:
                return []
            clean_messages = []
            for m in messages:
                message = {
                    "role": m.role,
                    "content": self._parse_content(m.content),
                    "system_prompt": m.system_prompt,
                    "rag_promt":m.rag_promt,
                    "timestamp": m.timestamp,
                    "user_id": m.user_id,
                    "user_name": m.user_name,
                }
                clean_messages.append(message)
            return clean_messages

    def get_user_id(self, user_name: str) -> int:
        with self.Session() as session:
            user_id = session.query(Message.user_id).filter(Message.user_name == user_name).distinct().first()
            assert user_id, f"User ID not found for {user_name}"
            return int(user_id[0])

    def save_user_message(
        self,
        content: Union[None, str, List[Dict[str, Any]]],
        conv_id: str,
        user_id: int,
        user_name: Optional[str] = None,
    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="user",
                content=self._serialize_content(content),
                conv_id=conv_id,
                user_id=user_id,
                user_name=user_name,
                timestamp=self.get_current_ts(),
            )
            session.add(new_message)
            session.commit()

    def save_assistant_message(
        self,
        content: Union[str, List[Dict[str, Any]]],
        conv_id: str,
        message_id: int,
        reply_user_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
        rag_promt: Optional[str] = None,

    ) -> None:
        with self.Session() as session:
            new_message = Message(
                role="assistant",
                content=self._serialize_content(content),
                conv_id=conv_id,
                timestamp=self.get_current_ts(),
                message_id=message_id,
                rag_promt = rag_promt,
                system_prompt=system_prompt,
                reply_user_id=reply_user_id,
            )
            session.add(new_message)
            session.commit()


    def save_feedback(self, feedback: str, user_id: int, message_id: int) -> None:
        with self.Session() as session:
            new_feedback = Like(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=1,
            )
            session.add(new_feedback)
            session.commit()


    def get_all_conv_ids(self, min_timestamp: Optional[int] = None) -> List[str]:
        with self.Session() as session:
            if min_timestamp is None:
                conversations = session.query(Conversation).all()
            else:
                conversations = session.query(Conversation).filter(Conversation.timestamp >= min_timestamp).all()
            return [conv.conv_id for conv in conversations]

    def _serialize_content(self, content: Union[None, str, List[Dict[str, Any]]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content: Any) -> Any:
        try:
            if content is None:
                return None
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, list):
                return content
            for m in parsed_content:
                if not isinstance(m, dict):
                    return content
            return parsed_content
        except json.JSONDecodeError:
            return content

    def set_current_subject(self, user_id: int, subject_name: str) -> None:
        with self.Session() as session:
            subject = session.query(Subject).filter(Subject.user_id == user_id).first()
            if subject:
                subject.subject = subject_name
                session.commit()
            else:
                new_subject = Subject(user_id=user_id, subject=subject_name)
                session.add(new_subject)
                session.commit()

    def get_current_subject(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.Session() as session:
            subject = session.query(Subject).filter(Subject.user_id == user_id).first()
            return {
                "subject": subject.subject,
            } if subject else None

    def set_temp_data(self, chat_id: int, key: str, value: str) -> None:
        with self.Session() as session:
            temp_data = session.query(TempData).filter(TempData.chat_id == chat_id, TempData.key == key).first()
            if temp_data:
                temp_data.value = value
            else:
                new_temp_data = TempData(chat_id=chat_id, key=key, value=value)
                session.add(new_temp_data)
            session.commit()

    def get_temp_data(self, chat_id: int, key: str) -> Optional[str]:
        with self.Session() as session:
            temp_data = session.query(TempData).filter(TempData.chat_id == chat_id, TempData.key == key).first()
            return temp_data.value if temp_data else None


