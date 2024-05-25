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
        "dans "
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

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/query")
def query_data(query: Query):
    result = conversational_rag_chain.invoke(
        {"input": query.question},
        config={
            "configurable": {"session_id": query.session_id},
        },
    )
    print(result['chat_history'])
    return {
        "response": result['answer'],
    }

@app.post("/init_session")
def init_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
