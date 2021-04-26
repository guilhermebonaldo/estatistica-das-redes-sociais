FROM python:3.6-slim-buster AS base_image

ARG APP_DIR=/usr/app/

USER root

RUN mkdir ${APP_DIR}

WORKDIR ${APP_DIR}

RUN apt-get update

RUN apt-get install -y build-essential

#RUN apt-get install python3-psycopg2 -y

COPY requirements.txt ${APP_DIR}

RUN apt-get -y install graphviz
RUN pip3 install -r requirements.txt

RUN chmod -R 755 /usr/app