FROM continuumio/miniconda3:latest

WORKDIR /workspaces
RUN mkdir -p /workspaces/.cache/huggingface
RUN chmod -R 777 /workspaces/.cache/huggingface
ENV HF_HOME=/workspaces/.cache/huggingface
ENV HF_TOKEN=$HF_TOKEN
ENV WANDB_API_KEY=$WANDB_API_KEY


COPY env.yml tmp/env.yml
WORKDIR /workspaces


RUN conda env create -f /tmp/env.yml 
