FROM python:3.12.2-bullseye

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
--extra-index-url https://download.pytorch.org/whl/cpu
