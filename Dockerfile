FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ENV PATH="/root/.local/bin:${PATH}"
RUN pip install --user pipenv


RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
COPY . .
RUN pipenv install --deploy --system
