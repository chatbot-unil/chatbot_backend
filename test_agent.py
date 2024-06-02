from config import Config
from chatbot.session import SessionManager
from chatbot.retrieval import Retriever
from chatbot.tools import Tools
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if __name__ == "__main__":
    system_prompt = Config.SYSTEM_PROMPT
    retriever = Retriever(
        chroma_host=Config.CHROMADB_HOST,
        chroma_port=Config.CHROMADB_PORT,
        collection_name=Config.COLLECTION_NAME,
        description="Ce retriever est basé sur une base de données de statistiques de l'Université de Lausanne. Il permet de répondre à des questions concernant les inscriptions étudiants de l'Université de Lausanne. par faculté, par sexe, par nationalité de 2011 à 2021."
    )

    tools = Tools()
    tools.add_retriever(retriever)
    session_manager = SessionManager(initial_message=Config.BOT_INIT_MESSAGE)
    session_id = session_manager.create_new_session()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    for tool in tools.get_retrievers():
        print(f"Tool Name: {tool.name}, Description: {tool.description}")

    agent = create_tool_calling_agent(llm, tools.get_retrievers(), prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools.get_retrievers(), verbose=True)
    
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        session_manager.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    while True:
        input_message = input("User: ")
        if input_message == "exit":
            break

    agent_with_chat_history.invoke(
        {"input": input_message},
        config={
            "configurable": {"session_id": session_id},
        },
    )