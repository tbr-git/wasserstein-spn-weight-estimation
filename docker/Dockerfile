FROM jupyter/tensorflow-notebook
#FROM cschranz/gpu-jupyter:v1.4_cuda-11.2_ubuntu-20.04
USER root
RUN apt update && apt install -y graphviz
USER jovyan
COPY ./environment.yml .
RUN conda env update --file ./environment.yml
COPY ./example-nb/example-nb.ipynb ./work
