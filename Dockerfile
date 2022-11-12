FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# install python
RUN apt-get update
RUN apt-get install python -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-venv -y

# create a user and switch to it
RUN useradd -ms /bin/bash jaxuser
USER jaxuser

# copy
WORKDIR /home/jaxuser/jax_examples
COPY . .

# add user local bin to path
ENV PATH="/usr/bin/:/usr/local/bin:/home/jaxuser/.local/bin:$PATH"

# install packages
RUN pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# stops jaxlib from pre allocating all gpu memory
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# run server and open port
CMD [ "jupyter-lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"]
EXPOSE 8888
