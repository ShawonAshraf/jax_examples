FROM python:3.12.2-bullseye

WORKDIR /workspaces
RUN mkdir -p /workspaces/.cache/huggingface
RUN chmod -R 777 /workspaces/.cache/huggingface
ENV HF_HOME=/workspaces/.cache/huggingface
ENV HF_TOKEN=$HF_TOKEN
ENV WANDB_API_KEY=$WANDB_API_KEY

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
--extra-index-url https://download.pytorch.org/whl/cpu
