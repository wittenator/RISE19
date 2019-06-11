# RISE19
This is the repository for the prospective project on which I am going to work during my researh internship at the Nottingham Trent university.

### Prerequisites

You will need to have [Docker](https://docs.docker.com/install/) installed in order to run the notebook. If you want to run the code with GPU support you need [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) as well.


### Installing

A step by step series of examples that tell you how to get a development env running.

First you need to clone the repository.
In order to do that navigate to the folder where you want to save the project and execute:

```
git clone https://github.com/wittenator/RISE19.git
```

Then proceed by building and running the container:
```
cd RISE19/
./start.sh
```
If you want to enable the GPU support, add ```--gpu``` to the bash command.
