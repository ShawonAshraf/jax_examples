FROM continuumio/miniconda3:latest

COPY env.yml tmp/env.yml
WORKDIR /workspaces


RUN conda env create -f /tmp/env.yml 
RUN conda activate jax-examples
RUN python utils/sanity_check.py
