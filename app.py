from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socketio
from pydantic import BaseModel
from config import Config
from chatbot.agent import Agent
from chatbot.tools import Tools
from chatbot.retrieval import Retriever
from chatbot.session import HumanMessage
from chatbot.session import AIMessage

@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever_students_automne = Retriever(
        chroma_host=Config.CHROMADB_HOST,
        chroma_port=Config.CHROMADB_PORT,
        collection_name="students_autumn",
        description= (
            "Ce retriever est basé sur une base de données de statistiques de l'Université de Lausanne."
            "Il permet de répondre à des questions concernant les inscriptions étudiants de l'Université de Lausanne. par faculté, par sexe, par nationalité de 2011 à 2021."
            "Il permet aussi de répondre à des qestions concernant les Étudiant·e·s inscrit·e·s au semestre d’automne par faculté selon le lieu de domicile avant les études de 2011 à 2021"
        )
    )
    demographics_retriever = Retriever(
        chroma_host=Config.CHROMADB_HOST,
        chroma_port=Config.CHROMADB_PORT,
        collection_name="demographics_and_population",
        description= (
            "Ce retriever est basé sur une base de données de statistiques de l'Université de Lausanne."
            "Il permet de répondre à des questions concernant des données démographiques et de population pour le canton de Vaud et la Suisse. Ces données on été fournies par l'Office fédéral de la statistique."
        )
    )
    acronyms_retriever = Retriever(
        chroma_host=Config.CHROMADB_HOST,
        chroma_port=Config.CHROMADB_PORT,
        collection_name="abbreviations_and_acronyms",
        description= (
            "Ce retriever est basé sur une base de données de statistiques de l'Université de Lausanne."
            "Il contient des abréviations et acronymes utilisés dans l'annuaire statistique de l'Université de Lausanne."
        )
    )
    tools = Tools()
    tools.add_retriever(retriever_students_automne)
    tools.add_retriever(demographics_retriever)
    tools.add_retriever(acronyms_retriever)
    global agent
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
    agent.session_manager.get_session_history(query['session_id']).messages.append(HumanMessage(content=query['question']))
    output = []
    if Config.USE_STREAM:
        await sio.emit('response_start', True, room=sid)
        await sio.emit('response_is_stream', True, room=sid)
        async for result in agent.query_stream(query['question'], query['session_id']):
            output.append(result)
            await sio.emit('response', result, room=sid)
        await sio.emit('response_end', True, room=sid)
    else:
        await sio.emit('response_start', True, room=sid)
        await sio.emit('response_is_stream', False, room=sid)
        result = agent.query_invoke(query['question'], query['session_id'])
        await sio.emit('response', result['output'], room=sid)
        output.append(result['output'])
        await sio.emit('response_end', True, room=sid)
    agent.session_manager.get_session_history(query['session_id']).messages.append(AIMessage(content=output))

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
