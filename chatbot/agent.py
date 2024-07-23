from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from .tools import Tools
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
        self.agent = create_tool_calling_agent(self.llm, tools=self.tools.get_all_tools(), prompt=self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools.get_all_tools(), verbose=True)
        self.agent_with_chat_history = RunnableWithMessageHistory(self.agent_executor, self.session_get_history, history_messages_key="chat_history", input_messages_key="input")
    
    def query_invoke(self, input_message, session_id):
        return self.agent_with_chat_history.invoke(
            {
                "input": input_message
            }, 
            config={
                "configurable": {
                    "session_id": session_id
                },
                "metadata": {
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
                },
                "metadata": {
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