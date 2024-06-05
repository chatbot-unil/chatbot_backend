from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import uuid

class SessionManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, initial_message="Bonjour ! Comment puis-je vous aider ?"):
        if not hasattr(self, "initialized"):
            self.store = {}
            self.init_message = initial_message
            self.initialized = True

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_new_session(self):
        session_id = str(uuid.uuid4())
        self.store[session_id] = ChatMessageHistory()
        self.store[session_id].messages.append(AIMessage(content=self.init_message))
        return session_id

    def test_is_session_id(self, session_id):
        return session_id in self.store

    def add_user_message(self, session_id, message):
        if not self.test_is_session_id(session_id):
            return False
        self.store[session_id].add_user_message(message)
        return True
    
    def add_ai_message(self, session_id, message):
        if not self.test_is_session_id(session_id):
            return False
        self.store[session_id].add_ai_message(message)
        return True

    def serialize_message(self, message):
        if isinstance(message, AIMessage):
            return {
                'type': 'bot',
                'content': message.content
            }
        elif isinstance(message, HumanMessage):
            return {
                'type': 'user',
                'content': message.content
            }
        else:
            raise TypeError(f"Type de message inconnu: {type(message)}")

    def get_session_messages(self, session_id):
        if not self.test_is_session_id(session_id):
            return {"error": "Session not found."}
        messages = self.store[session_id].messages
        return [self.serialize_message(msg) for msg in messages]

    def delete_session(self, session_id):
        if self.test_is_session_id(session_id):
            del self.store[session_id]
            return True
        return False

# Exemple d'utilisation:
# session_manager = SessionManager(initial_message="Welcome!")
# session_id = session_manager.create_new_session()
# messages = session_manager.get_session_messages(session_id)
