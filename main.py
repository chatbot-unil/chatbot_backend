from contextlib import asynccontextmanager
import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.middleware.cors import CORSMiddleware
import socketio

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    chroma_host = os.getenv("CHROMADB_HOST")
    chroma_port = int(os.getenv("CHROMADB_PORT"))

    collection_name = os.getenv("CHROMADB_COLLECTION_NAME")

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    global db, retriever, llm, prompt, document_chain, retrieval_chain, client, collection, system_prompt, contextualize_q_system_prompt, contextualize_q_prompt, history_aware_retriever, store, conversational_rag_chain

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection(collection_name)

    db = Chroma(collection_name=collection.name, client=client, embedding_function=embedding_function)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name="gpt-4o")

    history_context = (
        "Tu possèdes un historique de conversation avec l'utilisateur. "
        "\n\n"
    )
    history_context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_context),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = create_history_aware_retriever(
        llm, retriever, history_context_prompt
    )

    system_prompt = (
        "Tu es un assistant data science pour l'Université de Lausanne. "
        "Tu es chargé de répondre à des questions sur les statistiques de l'Université. "
        "Tu as accès à une base de données contenant des informations sur les statistiques de l'Université. "
        "Si tu ne trouves pas la réponse dans les données tu dois le dire. "
        "essaie de répondre le plus précisément possible et de manière informative. "
        "evite toute représentation autre que du texte pur (pas latex, pas de markdown, pas de code). "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(history_retriever, document_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app_asgi = socketio.ASGIApp(sio, app)

BotInitMessage = os.getenv("BOT_INIT_MESSAGE")

class Query(BaseModel):
    question: str
    session_id: str

class InitSession(BaseModel):
    session_id: str
    initial_message: str

class SessionHistory(BaseModel):
    session_id: str
    chat_history: dict

def create_new_session():
    session_id = str(uuid.uuid4())
    store[session_id] = ChatMessageHistory()
    return session_id

def add_initial_message(session_id, initial_message):
    if session_id not in store:
        session_id = create_new_session()
    store[session_id].messages.append(AIMessage(initial_message))

def test_is_session_id(session_id):
    return session_id in store

def serialize_message(message):
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
        raise TypeError(f"Unsupported message type: {type(message)}")

def get_session_history(session_id):
    if not test_is_session_id(session_id):
        return {"error": "Session not found."}
    messages = store[session_id].messages
    return [serialize_message(msg) for msg in messages]

async def query_bot(query):
    result = conversational_rag_chain.invoke(
        {"input": query['question']},
        config={
            "configurable": {"session_id": query['session_id']},
        },
    )
    return result['answer']

@sio.event
async def query(sid, query):
    if not test_is_session_id(query['session_id']):
        await sio.emit('error', {'message': 'Session not found.'}, room=sid)
        return
    result = await query_bot(query)
    await sio.emit('response', result['answer'], room=sid)

@sio.event
async def connect(sid, environ):
    print(f"connect {sid}")

@sio.event
async def init(sid):
    session_id = create_new_session()
    add_initial_message(session_id, BotInitMessage)
    await sio.emit('session_init', {'session_id': session_id, 'initial_message': BotInitMessage}, room=sid)

@sio.event
async def close_session(sid, session_id):
    if test_is_session_id(session_id):
        store.pop(session_id)
        # TODO: save session history in database
        await sio.emit('session_closed', {'session_id': session_id}, room=sid)

@sio.event
async def restore_session(sid, data):
    session_id = data.get('session_id')
    if session_id in store:
        messages = get_session_history(session_id)
        await sio.emit('session_restored', {'session_id': session_id, 'chat_history': messages}, room=sid)
    else:
        session_id = str(uuid.uuid4())
        add_initial_message(session_id, BotInitMessage)
        await sio.emit('session_init', {'session_id': session_id, 'initial_message': BotInitMessage}, room=sid)

@app.get("/get_session_history/{session_id}")
async def get_history(session_id: str):
    return get_session_history(session_id)

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
