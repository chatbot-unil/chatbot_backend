FROM python:latest

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY .env .env
COPY main.py main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
