# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.13.0rc1-gpu-jupyter

WORKDIR /tf

COPY requirements.txt requirements.txt
RUN apt-get install -y libcairo2-dev libgirepository1.0-dev
RUN pip3 install -r requirements.txt
