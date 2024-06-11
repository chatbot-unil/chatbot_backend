from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from .session import SessionManager

class ChatBot:
    def __init__(self, system_prompt, retriever):
        self.llm = ChatOpenAI(model_name="gpt-4o")

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
        self.history_retriever = create_history_aware_retriever(
            self.llm, retriever, history_context_prompt
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.history_retriever, self.document_chain)
        
        self.session_manager = SessionManager()

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.retrieval_chain,
            self.session_manager.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def query(self, input_message, session_id, use_streaming=False):
        if use_streaming:
            return self.conversational_rag_chain.stream(
                {"input": input_message},
                config={
                    "configurable": {"session_id": session_id},
                },
            )
        else:
            return self.conversational_rag_chain.invoke(
                {"input": input_message},
                config={
                    "configurable": {"session_id": session_id},
                },
            )
