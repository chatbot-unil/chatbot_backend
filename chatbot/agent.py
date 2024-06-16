from .tools import Tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from typing import List
from .config import Config

class LineGraph(BaseModel):
    year: List[int] = Field(..., description="La liste des années pour créer le graphique.")
    data: List[List[int]] = Field(..., description="La liste des séries de données pour créer le graphique.")
    labels: List[str] = Field(..., description="Les étiquettes pour chaque série de données.")
    title: str = Field(..., description="Le titre du graphique. (en français) est obligatoire.")
    x_label: str = Field(..., description="L'étiquette de l'axe des x. (en français) est obligatoire.")
    y_label: str = Field(..., description="L'étiquette de l'axe des y. (en français) est obligatoire.")
    is_start_zero: bool = Field(False, description="Définir si l'axe des y doit commencer à zéro. Par défaut, False.")

@tool("create_line_graph", args_schema=LineGraph)
def create_line_graph(year: List[int], data: List[List[int]], labels: List[str], title: str, x_label: str, y_label: str, is_start_zero: bool):
    """Ce tool permet de créer un graphique en courbe à partir des données fournies."""

    plt.figure(figsize=(10, 5))

    if is_start_zero:
        plt.ylim(0, max([max(series) for series in data]) * 1.1)

    for i, series in enumerate(data):
        label = labels[i]
        plt.plot(year, series, marker='o', linestyle='-', label=label)
        for j, value in enumerate(series):
            plt.annotate(value, (year[j], value), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    timestamp = str(time.time())
    directory = 'graph'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f'{timestamp}.png')
    plt.savefig(filepath)
    plt.close()

    public_ip = Config.PUBLIC_IP

    return f"http://{public_ip}:3001/graph/{timestamp}.png"


class BarGraph(BaseModel):
    categories: List[str] = Field(..., description="Les catégories pour le graphique en barres.")
    values: List[List[float]] = Field(..., description="Les valeurs correspondantes pour chaque catégorie.")
    labels: List[str] = Field(..., description="Les étiquettes pour chaque série de données.")
    title: str = Field(..., description="Le titre du graphique. (en français)")
    x_label: str = Field(..., description="L'étiquette de l'axe des x. (en français)")
    y_label: str = Field(..., description="L'étiquette de l'axe des y. (en français)")
    is_start_zero: bool = Field(True, description="Définir si l'axe des y doit commencer à zéro. Par défaut, True.")

@tool("create_bar_graph", args_schema=BarGraph)
def create_bar_graph(categories: List[str], values: List[List[float]], labels: List[str], title: str, x_label: str, y_label: str, is_start_zero: bool):
    """Ce tool permet de créer un graphique en barres à partir des données fournies."""

    plt.figure(figsize=(10, 5))
    
    x = range(len(categories))
    
    if is_start_zero:
        plt.ylim(0, max([max(series) for series in values]) * 1.1)
    
    bar_width = 0.2 
    for i, series in enumerate(values):
        plt.bar([p + bar_width * i for p in x], series, width=bar_width, label=labels[i])
        for j, value in enumerate(series):
            plt.text(x[j] + bar_width * i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks([p + bar_width * (len(values) - 1) / 2 for p in x], categories)
    plt.legend()
    plt.grid(True)

    timestamp = str(time.time())
    directory = 'graph'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f'{timestamp}.png')
    plt.savefig(filepath)
    plt.close()
    
    public_ip = Config.PUBLIC_IP

    return f"http://{public_ip}:3001/graph/{timestamp}.png"

# TODO: Tool bug to fix
# class ProportionGraph(BaseModel):
#     labels: List[str] = Field(..., description="Les étiquettes pour chaque proportion.")
#     sizes: List[float] = Field(..., description="Les proportions correspondantes.")
#     title: str = Field(..., description="Le titre du graphique. (en français)")

# @tool("create_proportion_graph", args_schema=ProportionGraph)
# def create_proportion_graph(labels: List[str], sizes: List[float], title: str):
#     """Ce tool permet de créer un graphique de proportions à partir des données fournies."""
#     plt.figure(figsize=(8, 8))
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
#     plt.title(title)
#     plt.axis('equal')

#     timestamp = str(time.time())
#     directory = 'graph'
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filepath = os.path.join(directory, f'{timestamp}.png')
#     plt.savefig(filepath)
#     plt.close()

#     return f"http://localhost:3001/graph/{timestamp}.png"

class Agent:

    def __init__(self, system_prompt, init_message, session_get_func, tools=None, stream=True):
        self.init_message = init_message
        self.system_prompt = system_prompt
        self.use_stream = stream
        self.tools = tools or Tools()
        self.session_get_history = session_get_func
        self.llm = ChatOpenAI(model_name="gpt-4o", streaming=self.use_stream)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.toolkit = [create_line_graph, create_bar_graph]
        for retriever in self.tools.get_retrievers():
            self.toolkit.append(retriever)
        self.agent = create_tool_calling_agent(self.llm, tools=self.toolkit, prompt=self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)
        self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_get_history, history_messages_key="chat_history", input_messages_key="input")
    
    def update_agent_executor(self):
        self.agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools.get_retrievers(), self.prompt), tools=self.tools.get_retrievers(), verbose=True)
        self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_get_history, history_messages_key="chat_history", input_messages_key="input")
          
    def add_retriever(self, retriever):
        self.tools.add_retriever(retriever)
        self.update_agent_executor()

    def update_tools(self, tools):
        self.tools = tools
        self.update_agent_executor()

    def get_retrievers(self):
        return self.tools.get_retrievers()
    
    def query_invoke(self, input_message, session_id):
        return self.agent_with_chat_history.invoke(
            {
                "input": input_message
            }, 
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )
    
    async def query_stream(self, input_message, session_id):
        async for event in self.agent_with_chat_history.astream_events(
            {
                "input": input_message
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
            version="v2"
        ) :
            type_ = event["event"]
            if type_ == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content