FROM python:3.12.2-bullseye

RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install 
RUN JAX_PLATFORMS=cpu poetry run sanity_check

