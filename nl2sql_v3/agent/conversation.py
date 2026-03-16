import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    role: str
    content: str
    sql: Optional[str] = None
    tables: Optional[List[str]] = None
    explanation: Optional[str] = None


@dataclass
class Conversation:
    id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id in self._conversations:
            return conversation_id
        
        self._conversations[conversation_id] = Conversation(id=conversation_id)
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id

    def get_or_create_conversation(self, conversation_id: str) -> str:
        return self.create_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sql: Optional[str] = None,
        tables: Optional[List[str]] = None,
        explanation: Optional[str] = None,
    ) -> bool:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        message = ConversationMessage(
            role=role,
            content=content,
            sql=sql,
            tables=tables,
            explanation=explanation,
        )
        conv.messages.append(message)
        return True

    def get_history(self, conversation_id: str) -> List[ConversationMessage]:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return []
        return conv.messages

    def get_last_sql(self, conversation_id: str) -> Optional[str]:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return None
        
        for msg in reversed(conv.messages):
            if msg.role == "assistant" and msg.sql:
                return msg.sql
        return None

    def get_last_tables(self, conversation_id: str) -> Optional[List[str]]:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return None
        
        for msg in reversed(conv.messages):
            if msg.role == "assistant" and msg.tables:
                return msg.tables
        return None

    def clear_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        return False

    def list_conversations(self) -> List[str]:
        return list(self._conversations.keys())


conversation_manager = ConversationManager()
