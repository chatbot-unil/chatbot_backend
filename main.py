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

    chroma_host = str(os.getenv("CHROMADB_HOST"))
    chroma_port = int(os.getenv("CHROMADB_PORT"))
    
    collection_name = str(os.getenv("CHROMADB_COLLECTION_NAME"))
    
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    global db, retriever, llm, prompt, document_chain, retrieval_chain, client, collection, system_prompt, contextualize_q_system_prompt, contextualize_q_prompt, history_aware_retriever, store, conversational_rag_chain

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection(collection_name)

    db = Chroma(collection_name=collection.name, client=client, embedding_function=embedding_function)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
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

def add_initial_message(session_id, initial_message):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    store[session_id].messages.append(AIMessage(initial_message))

def test_is_session_id(session_id):
    return session_id in store

# @app.post("/query")
# async def query_data(query: Query):
#     if not test_is_session_id(query.session_id):
#         return {"error": "Session not found."}
#     result = conversational_rag_chain.invoke(
#         {"input": query.question},
#         config={
#             "configurable": {"session_id": query.session_id},
#         },
#     )
#     return {
#         "response": result['answer'],
#     }

@sio.event
async def query(sid, query):
    if not test_is_session_id(query['session_id']):
        return {"error": "Session not found."}
    result = conversational_rag_chain.invoke(
        {"input": query['question']},
        config={
            "configurable": {"session_id": query['session_id']},
        },
    )
    await sio.emit('response', result['answer'], room=sid)  

# @app.get("/init_session")
# async def init_session():
#     session_id = str(uuid.uuid4())
#     add_initial_message(session_id, BotInitMessage)
#     return {
#         "session_id": session_id,
#         "initial_message": BotInitMessage,
#     }

@sio.event
async def init(sid):
    session_id = str(uuid.uuid4())
    add_initial_message(session_id, BotInitMessage)
    await sio.emit('init', {'session_id': session_id, 'initial_message': BotInitMessage}, room=sid)

@sio.event
async def close_session(sid, session_id):
    if test_is_session_id(session_id):
        store.pop(session_id)
        # TODO: save session history in database
        await sio.emit('session_closed', {'session_id': session_id}, room=sid)

@app.get("/get_session_history/{session_id}")
async def get_session_history(session_id: str):
    if not test_is_session_id(session_id):
        return {"error": "Session not found."}
    else:
        return {
            "session_id": session_id,
            "chat_history": store[session_id].messages,
        }

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
