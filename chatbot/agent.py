from .tools import Tools
from .session import SessionManager
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

class Agent:

    def __init__(self, system_prompt, init_message, tools=None, stream=True):
        self.system_prompt = system_prompt
        self.use_stream = stream
        self.tools = tools or Tools()
        self.session_manager = SessionManager(initial_message=init_message)
        self.llm = ChatOpenAI(model_name="gpt-4o", streaming=self.use_stream)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, self.tools.get_retrievers(), self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools.get_retrievers(), verbose=True)
        self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_manager.get_session_history, history_messages_key="chat_history", input_messages_key="input")
    
    def update_agent_executor(self):
        self.agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools.get_retrievers(), self.prompt), tools=self.tools.get_retrievers(), verbose=True)
        self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_manager.get_session_history, history_messages_key="chat_history", input_messages_key="input")
          
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
            {"input": input_message}, 
            config={"configurable": {"session_id": session_id}}
        )
    
    async def query_stream(self, input_message, session_id):
        async for event in self.agent_with_chat_history.astream_events(
            {"input": input_message},
            config={
                "configurable": {"session_id": session_id},
            },
            version="v1"
        ) :
            type_ = event["event"]
            if type_ == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
    
    def create_new_session(self):
        return self.session_manager.create_new_session()
    
    def get_session_messages(self, session_id):
        return self.session_manager.get_session_messages(session_id)
    
    def delete_session(self, session_id):
        self.session_manager.delete_session(session_id)
    
    def test_is_session_id(self, session_id):
        return self.session_manager.test_is_session_id(session_id)
