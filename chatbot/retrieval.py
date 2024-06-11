import chromadb
from langchain_community.vectorstores import Chroma
from .embeddings import initialize_embeddings

class Retriever:
    def __init__(self, chroma_host=None, chroma_port=None, collection_name=None, description=None, search_type="similarity", search_kwargs={"k": 10}):
        self.embedding_function = initialize_embeddings()
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.collection_name = collection_name
        self.client = self._initialize_client()
        self.collection = self._get_collection()
        self.db = self._initialize_db()
        self.retriever = self._initialize_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self.description = description

    def _initialize_client(self):
        return chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)

    def _get_collection(self):
        return self.client.get_collection(self.collection_name)

    def _initialize_db(self):
        return Chroma(collection_name=self.collection.name, client=self.client, embedding_function=self.embedding_function)

    def _initialize_retriever(self, search_type, search_kwargs):
        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def retrieve(self, query):
        return self.retriever(query)
    
    def get_retriver(self):
        return self.retriever
    
    def add_document(self, document):
        # TODO: Add document to the collection
        return None

# Exemple d'utilisation:
# retriever = Retriever(chroma_host="host", chroma_port=8000, collection_name="my_collection")
# result = retriever.retrieve("your_query")
