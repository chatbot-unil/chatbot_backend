from langchain.tools.retriever import create_retriever_tool
class Tools:
    def __init__(self):
        self.retrievers = []

    def get_retrievers(self):
        return self.retrievers

    def add_retriever(self, retriever):
        description = getattr(retriever, 'description', None) or "Aucune description"
        
        tool = create_retriever_tool(
            retriever.get_retriver(),
            name=retriever.collection_name,
            description=description
        )
        self.retrievers.append(tool)

    def get_retriever(self, name):
        for retriever in self.retrievers:
            if retriever.name == name:
                return retriever
        return None
