FROM python:latest

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY .env .env
COPY app.py app.py
COPY config.py config.py
COPY chatbot chatbot

EXPOSE 8000

CMD ["uvicorn", "app:app_asgi", "--host", "0.0.0.0", "--port", "8000"]
