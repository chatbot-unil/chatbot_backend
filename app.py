from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socketio
from pydantic import BaseModel
from config import Config
from chatbot.agent import Agent
from chatbot.tools import Tools
from chatbot.retrieval import Retriever
from database import Database

@asynccontextmanager
async def lifespan(app: FastAPI):
    tools = Tools()
    global agent
    db = Database(
        dbname=Config.POSTGRES_DB,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD,
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT
    )
    retrivals = db.get_all_collections()
    for retrival in retrivals:
        retriver = Retriever(
            chroma_host=retrival['host'],
            chroma_port=retrival['port'],
            collection_name=retrival['collection'],
            description=retrival['description'],
            search_kwargs={"k": 30},
            search_type="similarity"
        )
        tools.add_retriever(retriver)
    agent = Agent(system_prompt=Config.SYSTEM_PROMPT, init_message=Config.BOT_INIT_MESSAGE, tools=tools, stream=Config.USE_STREAM)
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

class Query(BaseModel):
    question: str
    session_id: str

@sio.event
async def query(sid, query):
    if not agent.test_is_session_id(query['session_id']):
        await sio.emit('error', {'message': 'Session not found.'}, room=sid)
        return
    if Config.USE_STREAM:
        await sio.emit('response_start', True, room=sid)
        async for result in agent.query_stream(query['question'], query['session_id']):
            await sio.emit('response', result, room=sid)
        await sio.emit('response_end', True, room=sid)
    else:
        await sio.emit('response_start', True, room=sid)
        result = agent.query_invoke(query['question'], query['session_id'])
        await sio.emit('response', result['output'], room=sid)
        await sio.emit('response_end', True, room=sid)

@sio.event
async def connect(sid, environ):
    print(f"connect {sid}")

@sio.event
async def init(sid):
    session_id = agent.create_new_session()
    await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.system_prompt}, room=sid)

@sio.event
async def close_session(sid, session_id):
    agent.delete_session(session_id)
    await sio.emit('session_closed', {'session_id': session_id}, room=sid)

@sio.event
async def restore_session(sid, data):
    session_id = data.get('session_id')
    if agent.test_is_session_id(session_id):
        messages = agent.get_session_messages(session_id)
        await sio.emit('session_restored', {'session_id': session_id, 'chat_history': messages}, room=sid)
    else:
        session_id = agent.create_new_session()
        await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.system_prompt}, room=sid)

@app.get("/get_session_history/{session_id}")
async def get_history(session_id: str):
    return agent.get_session_messages(session_id)

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
