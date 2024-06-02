from .tools import Tools
from .session import SessionManager
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

class Agent:

	def __init__(self, system_prompt, init_message, tools=None):
		self.system_prompt = system_prompt
		self.tools = tools or Tools()
		self.session_manager = SessionManager(initial_message=init_message)
		self.llm = ChatOpenAI(model_name="gpt-4o")
		self.prompt = hub.pull("hwchase17/openai-functions-agent")
		self.agent = create_tool_calling_agent(self.llm, self.tools.get_retrievers(), self.prompt)
		self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools.get_retrievers(), verbose=True)
		self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_manager.get_session_history, input_messages_key="input", history_messages_key="chat_history")
	
	def update_agent_executor(self):
		self.agent_executor = AgentExecutor(agent=create_tool_calling_agent(self.llm, self.tools.get_retrievers(), self.prompt), tools=self.tools.get_retrievers(), verbose=True)
		self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_manager.get_session_history, input_messages_key="input", history_messages_key="chat_history")
	
	def add_retriever(self, retriever):
		self.tools.add_retriever(retriever)
		self.update_agent_executor()

	def update_tools(self, tools):
		self.tools = tools
		self.update_agent_executor()

	def get_retrievers(self):
		return self.tools.get_retrievers()

	def query(self, input_message, session_id):
		return self.agent_with_chat_history.invoke(
			{"input": input_message}, 
			config={"configurable": {"session_id": session_id}}
		)
	
	def create_new_session(self):
		return self.session_manager.create_new_session()
	
	def get_session_messages(self, session_id):
		return self.session_manager.get_session_messages(session_id)
	
	def delete_session(self, session_id):
		self.session_manager.delete_session(session_id)
	
	def test_is_session_id(self, session_id):
		return self.session_manager.test_is_session_id(session_id)
