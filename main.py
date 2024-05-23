# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import json
# import uuid
# from dotenv import load_dotenv
# import chromadb
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.schema import Document
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import time

# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# chroma_host = str(os.getenv("CHROMADB_HOST"))
# chroma_port = int(os.getenv("CHROMADB_PORT"))

# embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
# collection = client.get_collection("students_autumn_2011_2021")

# db = Chroma(collection_name=collection.name, client=client, embedding_function=embedding_function)

# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":4})

# llm = ChatOpenAI(model_name="gpt-4o")

# prompt = ChatPromptTemplate.from_template("""Avec les données suivantes, répondez à la question posée.

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

# retrieval_chain = create_retrieval_chain(retriever, document_chain)
# query = "Combien d'étudiantes sont inscrits en 2020 à la Faculté des lettres ?"
# time_start = time.time()
# result = retrieval_chain.invoke({"input": query})
# time_end = time.time()
# print(result['answer'])
# print(f"Time taken: {round(time_end - time_start, 2)} seconds")

from contextlib import asynccontextmanager
import os
import json
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
import time

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    chroma_host = str(os.getenv("CHROMADB_HOST"))
    chroma_port = int(os.getenv("CHROMADB_PORT"))
    
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection("students_autumn_2011_2021")

    global db, retriever, llm, prompt, document_chain, retrieval_chain
    db = Chroma(collection_name=collection.name, client=client, embedding_function=embedding_function)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-4o")

    prompt = ChatPromptTemplate.from_template("""Avec les données suivantes, répondez à la question posée.

		<context>
		{context}
		</context>

		Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    yield

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    question: str

@app.post("/query")
def query_data(query: Query):
    time_start = time.time()
    result = retrieval_chain.invoke({"input": query.question})
    time_end = time.time()
    return {
        "response": result['answer'],
        "time_taken": round(time_end - time_start, 2)
    }

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API du chatbot de l'Université de Lausanne."}
