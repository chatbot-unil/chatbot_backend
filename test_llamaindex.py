import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
import openai
import chromadb
import json
import time

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

chroma_host = os.getenv("CHROMADB_HOST")
chroma_port = os.getenv("CHROMADB_PORT")

remote_db = chromadb.HttpClient(host=chroma_host, port=chroma_port)

chroma_collection = remote_db.get_or_create_collection("students_autumn_2011_2021")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

query = "Combien d'étudiants sont inscrits en 2014 à la Faculté des lettres ?"
time_start = time.time()
results = index.as_query_engine(kwargs={"k": 10}).query(query)
time_end = time.time()

print(f"Question: {query}")
print(f"Resultats: {results}")

print(f"Time taken: {round(time_end - time_start, 2)} seconds")