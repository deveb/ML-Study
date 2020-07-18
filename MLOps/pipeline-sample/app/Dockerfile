FROM python:3

MAINTAINER eb <hwang@eunb.it>

ENV PYTHONUNBUFFERED 0

RUN apt-get update && apt-get -y install libpq-dev
# libpq-dev for postgres

COPY    ./    /pipeline-sample/app
WORKDIR /pipeline-sample/app
RUN    pip install -r requirements.txt

CMD    ["python", "-m", "flask", "run", "--host=0.0.0.0"]
