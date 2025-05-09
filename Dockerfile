FROM python:3.10.12

WORKDIR /server

RUN apt-get update && apt-get install -y openssh-client

RUN pip install --upgrade pip==25.1.1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
