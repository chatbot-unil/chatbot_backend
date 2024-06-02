from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from .session import SessionManager

def initialize_bot(system_prompt, retriever):
    llm = ChatOpenAI(model_name="gpt-4o")

    history_context = (
        "Tu possèdes un historique de conversation avec l'utilisateur. "
        "\n\n"
    )
    history_context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_context),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = create_history_aware_retriever(
        llm, retriever, history_context_prompt
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(history_retriever, document_chain)
    
    session_manager = SessionManager()

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        session_manager.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def query_bot(bot_chain, input_message, session_id, use_streaming=False):
    if use_streaming:
        return bot_chain.stream(
            {"input": input_message},
            config={
                "configurable": {"session_id": session_id},
            },
        )
    else:
        return bot_chain.invoke(
            {"input": input_message},
            config={
                "configurable": {"session_id": session_id},
            },
        )