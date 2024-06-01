import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

chat = ChatOpenAI(model="gpt-4o", temperature=0.2)

system_prompt = (
	"Tu es un assistant data science pour l'Université de Lausanne. "
	"Tu es chargé de répondre à des questions sur les statistiques de l'Université. "
	"Tu as accès à une base de données contenant des informations sur les statistiques de l'Université. "
	"Si tu ne trouves pas la réponse dans les données tu dois le dire. "
	"essaie de répondre le plus précisément possible et de manière informative. "
	"evite toute représentation autre que du texte pur (pas latex, pas de markdown, pas de code). "
	"\n\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

response = chain.stream(
    {
        "messages": [
            HumanMessage(
                content="peut tu me parler de l'université de Lausanne?"
            )
        ],
    }
)

for message in response:
	print(message)