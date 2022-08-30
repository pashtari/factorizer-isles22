
FROM python:3.9-slim

RUN mkdir -p /model /input /output

COPY requirements.txt /model/requirements.txt
COPY predict.py /model/predict.py
COPY /logs /model/logs 

WORKDIR /model

RUN apt-get -y update
RUN apt-get -y install git

RUN python -m pip install --user -U pip

RUN python -m pip install --user -r requirements.txt

ENTRYPOINT [ "python", "./predict.py"]
