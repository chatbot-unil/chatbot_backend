from langchain.tools.retriever import create_retriever_tool
from typing import List, Union
from langchain.tools import BaseTool
from .function import Functions

class Tools:
    def __init__(self):
        self.retrievers: List[BaseTool] = []
        self.functions: List[BaseTool] = Functions().get_all_functions()

    def get_retrievers(self) -> List[BaseTool]:
        return self.retrievers
    
    def get_retriever(self, name: str) -> Union[BaseTool, None]:
        for retriever in self.retrievers:
            if retriever.name == name:
                return retriever
        return None

    def add_retriever(self, retriever):
        description = getattr(retriever, 'description', None) or "Aucune description"
        
        tool = create_retriever_tool(
            retriever.get_retriver(),
            name=retriever.collection_name,
            description=description
        )
        self.retrievers.append(tool)
    
    def print_retrievers(self):
        for retriever in self.retrievers:
            print(retriever)

    def print_all_tools(self):
        for tool in self.get_all_tools():
            print(tool)

    def get_all_tools(self) -> List[BaseTool]:
        return self.retrievers + self.functions