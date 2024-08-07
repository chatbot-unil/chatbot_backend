from fastapi import FastAPI
from contextlib import asynccontextmanager
import socketio
from pydantic import BaseModel
from chatbot.config import Config
from chatbot.agent import Agent
from chatbot.tools import Tools
from chatbot.retrieval import Retriever
from chatbot.database import Database
from chatbot.session import SessionManager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
            search_kwargs={"k": retrival['search_k']},
            search_type="similarity"
        )
        tools.add_retriever(retriver)
    tools.print_all_tools()
    is_created = await db.create_chat_history_table()
    print(f"Chat history table created: {is_created}")
    await db.print_chat_history_schema()
    session_manager = SessionManager(initial_message=Config.BOT_INIT_MESSAGE)
    agent = Agent(
        system_prompt=Config.SYSTEM_PROMPT, 
        init_message=Config.BOT_INIT_MESSAGE,
        tools=tools, stream=Config.USE_STREAM, 
        session_get_func=session_manager.get_session_history
    )
    app.mount("/graph", StaticFiles(directory="graph"), name="graph")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
	allow_origin_regex=Config.ALLOWED_ORIGINS,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins_regex=Config.ALLOWED_ORIGINS
)

app_asgi = socketio.ASGIApp(sio, app)
class Query(BaseModel):
    question: str
    session_id: str
    
class User(BaseModel):
    user_uuid: str

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
async def init(sid, data):
    user_uuid = data.get('user_uuid')
    if not user_uuid:
        await sio.emit('error', {'message': 'User UUID is required.'}, room=sid)
        return
    session_id = session_manager.create_new_session(sid)
    await db.add_session(user_uuid, session_id)
    await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.init_message}, room=sid)
    
@sio.event
async def restore_session(sid, data):
    print(data)
    session_id = data.get('session_id')
    if isinstance(session_id, list):
        session_id = session_id[0]
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
        await sio.emit('session_init', {'session_id': session_id, 'initial_message': agent.init_message}, room=sid)

@app.post("/api/v1/query")
async def query(query: Query):
    if not session_manager.test_is_session_id(query.session_id):
        return {'message': 'Session not found.'}
    return agent.query_invoke(query.question, query.session_id)

@app.get("/api/v1/get_user_sessions/{user_uuid}")
async def get_user_sessions(user_uuid: str):
    sessions = await db.get_sessions(user_uuid)
    if not sessions:
        return {"session_ids": []}
    return {"session_ids": sessions}

@app.get("/api/v1/get_session_history/{session_id}")
async def get_history(session_id: str):
    return session_manager.get_session_messages(session_id)

@app.get("/api/v1/check_user_exists/{user_uuid}")
async def get_user(user_uuid: str):
    user = await db.test_if_user_exists(user_uuid)
    if user:
        return {"user_exists": True}
    else:
        return {"user_exists": False}
    
@app.post("/api/v1/create_user")
async def create_user():
    user_uuid = await db.create_user()
    return {"user_uuid": user_uuid}

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
