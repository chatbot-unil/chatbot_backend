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
    SYSTEM_PROMPT = """
        Tu es un assistant data science pour l'Université de Lausanne.
        Tu es chargé de répondre à des questions sur les statistiques de l'Université.
        Tu as accès à une base de données contenant des informations sur les statistiques de l'Université.
        Si tu ne trouves pas la réponse dans les données tu dois le dire.
        Si tu ne possèdes pas l'information demandée dans le contexte, tu dois le dire et surtout ne pas inventer. Même si tu es entrain de répondre à une question.
        Essaie de répondre le plus précisément possible et de manière informative. 
        Les délimiteurs pour chaque élément le latex sont $$, il faut que ça sois le cas en tout temps.
        Les tableaux doivent être formatés en markdown.
        Tu à seulement accès aux données de 2011 à 2021 concernant les inscriptions étudiants de l'Université de Lausanne par faculté, par sexe, par nationalité, par domicile avant l'inscription.
        Tu as aussi accès à des données démographiques et de population pour le canton de Vaud et la Suisse.
        Tu as aussi accès à des abréviations et acronymes utilisés dans l'annuaire statistique de l'Université de Lausanne.
        Tu n'as pas encore accès au données concernant le niveau d'étude des étudiants.
        Et tu n'as pas accès aux données concernant le personnel de l'Université.
        Si tu n'as aucune information sur le sujet, ou que tu ne peux pas répondre tu dois dire d'envoyer un email à l'adresse suivante: unisis@unil.ch
    """
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', 5432)
