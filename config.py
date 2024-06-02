import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMADB_HOST = os.getenv("CHROMADB_HOST")
    CHROMADB_PORT = int(os.getenv("CHROMADB_PORT"))
    COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME")
    BOT_INIT_MESSAGE = os.getenv("BOT_INIT_MESSAGE")
    USE_STREAM = bool(os.getenv("USE_STREAM") == "True")
    SYSTEM_PROMPT = (
        "Tu es un assistant data science pour l'Université de Lausanne. "
        "Tu es chargé de répondre à des questions sur les statistiques de l'Université. "
        "Tu as accès à une base de données contenant des informations sur les statistiques de l'Université. "
        "Si tu ne trouves pas la réponse dans les données tu dois le dire. "
        "essaie de répondre le plus précisément possible et de manière informative. "
        "evite toute représentation autre que du texte pur (pas latex, pas de markdown, pas de code). "
        "\n\n"
        "{context}"
    )
