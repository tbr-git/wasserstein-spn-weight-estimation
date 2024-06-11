# Wasserstein SPN Weight Estimation
A method that uses a Wasserstein loss to optimize the weights of a stochastic Petri net. This method almost directly optimizes the nets Earth Mover's Stochastic Conformance score.

## Getting Started
This repository contains a `docker` folder that helps you running the application in a Docker container.
Currently, we use the `jupyter/tensorflow-notebook` as base image. Therefore, running `docker compose up` will start a notebook in the background, which is a bit of overkill.
However, this facilitates easy experimentation with the method (calling it repetitively or using `PM4Py` to show and analyze the resulting SPN).

To run the container:

1. Change into the `docker` folder.
2. Change the NB_UID and NB_GID to your own ids (otherwise you may run into problems with file permissions). Potentially, you will also have to change the port for the notebook in the `docker-compose.yml` file (currently this is set to 8889).
3. Run `docker compose build`
4. Run `docker compose up`
5. Attach to the notebook. You will find an example notebook how to call the method. **Or:** Attach to the running container. In the `ot_backprop_pnwo` folder, there is a `scripts` folder that illustrates how you can run the method.

## Non-docker Setup
You can create a `conda` enviroment as follows:

    conda create --name <env> --file requirements.txt

Note that the requirement file is a bit bloated (i.e., it contains a jupyter notebook and a lot of data science packages that are not strictly required).
