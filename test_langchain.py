import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import uuid
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

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chroma_host = os.getenv("CHROMADB_HOST", "localhost")
chroma_port = int(os.getenv("CHROMADB_PORT", "3003"))

embedding_function = OpenAIEmbeddings()

client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
collection = client.get_or_create_collection("students_autumn_2011_2021")

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

def process_json_files(file_paths):
    documents = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            context = data.get('context', '')
            for year, content in data.get('data', {}).items():
                for faculty, stats in content.items():
                    document_content = json.dumps({
                        "year": year,
                        "faculty": faculty,
                        "stats": stats
                    }, ensure_ascii=False)
                    metadata = {
                        "context": context,
                        "year": year,
                        "faculty": faculty
                    }
                    documents.append(Document(page_content=document_content, metadata=metadata))

    for doc in documents:
        collection.add(
            ids=[str(uuid.uuid1())],
            metadatas=doc.metadata,
            documents=[doc.page_content]
        )
    
    return documents

documents = process_json_files(files)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

db = Chroma.from_documents(texts, embedding_function)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":4})

llm = ChatOpenAI(model_name="gpt-4o")

prompt = ChatPromptTemplate.from_template("""Avec les données suivantes, répondez à la question posée.

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)
query = "Combien d'étudiantes sont inscrits en 2020 à la Faculté des lettres ?"
time_start = time.time()
result = retrieval_chain.invoke({"input": query})
time_end = time.time()
print(result)
print(f"Time taken: {round(time_end - time_start, 2)} seconds")