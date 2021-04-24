FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m aniket

RUN chown -R aniket:aniket /home/aniket/

COPY --chown=aniket . /home/aniket/app/

USER aniket

RUN cd /home/aniket/app/ && pip3 install -r requirements.txt

WORKDIR /home/aniket/app 
