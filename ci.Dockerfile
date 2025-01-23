FROM continuumio/miniconda3:latest

COPY env.yml tmp/env.yml
WORKDIR /workspaces

RUN conda install -c conda-forge cxx-compiler


RUN conda env create -f /tmp/env.yml
COPY ./utils/sanity_check.py .

RUN conda run -n jax-examples python sanity_check.py
