FROM python:3.12.0

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py
COPY chatbot chatbot

EXPOSE 8000

CMD ["uvicorn", "app:app_asgi", "--host", "0.0.0.0", "--port", "8000"]
