# Backend du projet chatbot

## Introduction

Ce git contient le code du backend du projet chatbot. Il fait partie de mon projet de bachelor. Le but de ce projet est de créer un chatbot qui permet de répondre à des questions sur des données statistiques. Le chatbot sera capable de répondre à des questions sur des données statistiques provenant de l'annuaire statistique de l'Université de Lausanne ainsi que de faire des graphiques.

## Technologies utilisées

Le backend est écrit en Python et utilise le framework FastAPI. FastAPI est un framework web moderne pour construire des APIs avec Python 3.6+ basé sur des annotations de type. Il est très rapide (hautes performances) et facile à apprendre. Concernant l'interaction avec les LLM (Large Language Models), j'utilise la librairie langchain qui permet de faire des requêtes à des modèles de langage. Ainsi que la librairie llamaIndex qui permet de faire des requêtes à une base de données vectorisée. Pour la génération de graphiques, j'utilise la librairie matplotlib.

## Installation

Le backend est utilisé via un environnement virtuel. Pour le créer, il suffit de lancer les commandes suivantes :

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
