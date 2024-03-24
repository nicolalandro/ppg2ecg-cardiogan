FROM python:3.10

WORKDIR /code
COPY packages.txt .
RUN apt-get update
RUN cat packages.txt | xargs -P1 apt-get install -y 

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && pip install -r requirements.txt
