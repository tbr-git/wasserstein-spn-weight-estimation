FROM quay.io/jupyter/tensorflow-notebook
#FROM cschranz/gpu-jupyter:v1.4_cuda-11.2_ubuntu-20.04
USER root
RUN apt update && apt install -y graphviz
USER jovyan
COPY --chown=${NB_UID}:${NB_GID} ./environment.yml /tmp/
RUN mamba env update --file /tmp/environment.yml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
COPY ./example-nb/example-nb.ipynb ./work
