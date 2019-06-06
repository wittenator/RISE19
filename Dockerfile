FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN pip3 install --user keras

COPY . .
