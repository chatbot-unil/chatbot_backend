import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
import openai
import chromadb
import json

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

chroma_host = os.getenv("CHROMADB_HOST")
chroma_port = os.getenv("CHROMADB_PORT")

remote_db = chromadb.HttpClient(host=chroma_host, port=chroma_port)

chroma_collection = remote_db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

files = [
    "data/students_autumn/FBM.json",
    "data/students_autumn/FCUE.json",
    "data/students_autumn/FDCA.json",
    "data/students_autumn/FGSE.json",
    "data/students_autumn/FTSR.json",
    "data/students_autumn/HEC.json",
    "data/students_autumn/Lettres.json",
    "data/students_autumn/SSP.json",
    "data/students_autumn/TOTAL.json",
]

documents = []

for file_path in files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        context = data.get('context', '')
        for year, content in data.get('data', {}).items():
            for faculty, stats in content.items():
                document = {
                    "year": year,
                    "faculty": faculty,
                    "stats": stats,
                    "context": context
                }
                documents.append(document)

from llama_index.core import Document
indexed_documents = [Document(text=json.dumps(doc)) for doc in documents]

index = VectorStoreIndex.from_documents(indexed_documents, storage_context=storage_context)

query = "How many female students were enrolled in FBM in 2020?"
results = index.as_query_engine().query(query)

print(f"Query: {query}")
print(f"Results: {results}")