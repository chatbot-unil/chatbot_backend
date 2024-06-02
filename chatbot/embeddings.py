from langchain_openai import OpenAIEmbeddings

def initialize_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")
