from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socketio
from pydantic import BaseModel
from config import Config
from chatbot.agent import Agent
from chatbot.tools import Tools
from chatbot.retrieval import Retriever
from chatbot.database import Database
from chatbot.session import SessionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, agent, session_manager
    tools = Tools()
    db = Database(
        dbname=Config.POSTGRES_DB,
        user=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD,
        host=Config.POSTGRES_HOST,
        port=Config.POSTGRES_PORT
    )
    retrivals = await db.get_all_collections()
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
    
    tools.print_retrievers()
    is_created = await db.create_chat_history_table()
    print(f"Chat history table created: {is_created}")
    await db.print_chat_history_shema()
    session_manager = SessionManager(initial_message=Config.BOT_INIT_MESSAGE)
    agent = Agent(system_prompt=Config.SYSTEM_PROMPT, init_message=Config.BOT_INIT_MESSAGE, tools=tools, stream=Config.USE_STREAM, session_get_func=session_manager.get_session_history)
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
    if not session_manager.test_is_session_id(query['session_id']):
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
async def disconnect(sid):
    session_id = session_manager.sid_to_session.get(sid)
    if session_id:
        if session_manager.test_is_session_id(session_id):
            messages = session_manager.get_session_history(session_id).messages
            if not await db.test_if_chat_history_exists(session_id):
                await db.init_chat_history(session_id)
            await db.insert_chat_messages(session_id, messages)
        session_manager.remove_sid_mapping(sid)
    print(f"disconnect {sid}")

@sio.event
async def init(sid):
    session_id = session_manager.create_new_session(sid)
    await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.system_prompt}, room=sid)
    
@sio.event
async def restore_session(sid, data):
    session_id = data.get('session_id')
    if session_manager.test_is_session_id(session_id):
        messages = session_manager.get_session_messages(session_id)
        session_manager.map_sid_to_session(sid, session_id)
        await sio.emit('session_restored', {'session_id': session_id, 'chat_history': messages}, room=sid)
    elif await db.test_if_chat_history_exists(session_id):
        messages = await db.get_chat_messages(session_id)
        session_manager.insert_session_from_db(session_id, messages)
        session_manager.map_sid_to_session(sid, session_id)
        messages = session_manager.get_session_messages(session_id)
        await sio.emit('session_restored', {'session_id': session_id, 'chat_history': messages}, room=sid)
    else:
        session_id = session_manager.create_new_session(sid)
        await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.system_prompt}, room=sid)

@app.get("/get_session_history/{session_id}")
async def get_history(session_id: str):
    return session_manager.get_session_messages(session_id)

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
