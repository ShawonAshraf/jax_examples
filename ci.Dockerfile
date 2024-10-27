FROM python:3.12.2-bullseye

RUN pip install poetry
WORKDIR /ci
COPY . /ci
RUN poetry config virtualenvs.create false
RUN poetry install 
RUN poetry run sanity_check

