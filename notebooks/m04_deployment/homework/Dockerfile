FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock",  "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "starter.py" ]

RUN mkdir -p output/fhv

RUN python starter.py fhv 2021 04