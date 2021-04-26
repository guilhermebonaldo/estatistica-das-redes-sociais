# Social network statistics

This project is based on **MAE5908 - Estatística de Redes Sociais (2020)** course, part of Universidade de São Paulo Statistics department (IME-USP) master's program.

Based on work of professor [Jefferson Antonio Galves](http://lattes.cnpq.br/5430021088108855) and [Kadmo de Souza Laxa](http://lattes.cnpq.br/2387830098447117)

This template is an adpatation of [Felipe Penha](https://github.com/felipepenha) [py-greenhouse project](https://github.com/felipepenha/py-greenhouse)


-------------------------

## Project Organization :pencil2:

      ├── artifacts    <- Trained and serialized models, pipelines, etc. 
      │ 
      ├── data         <- saved data
      │
      ├── notebooks    <- Jupyter notebooks. 
      |
      ├── src
      │  ├── models.py <- Script of available models
      │  ├── plots.py  <- Script to plot runs results
      │  └── utils.py  <- Script for helper functions.
      │
      ├── references   <- Data dictionaries, manuals, and all other explanatory materials.
      │
      ├── reports      <- Generated analysis as HTML, PDF, LaTeX, etc.
      │
      ├── docker-compose.yml  <- instructions to build and run containers/services.
      ├── Dockerfile          <- Dockerfile for base-image.
      ├── Makefile            <- Wrapper for the project cli instructions.
      ├── requirements.txt    <- The requirements file with installed libraries.
      └── README.md           <- The top-level README for developers using this project. 
      

---------------------

## Quick Start :clock10:

This is a template repository. [Follow this link for instructions to create a repository from a template](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template).

First, make sure `make`, `git`, `docker` and `docker-compose` are installed in your system.

These are requirements for your local machine, ideally a Debian Linux OS (the following can be achieved in Widowns by downloading WSL, that starts a Linux VM inside Windows :confused:):

#### [docker](https://docs.docker.com/engine/install/)

Follow the [instructions in the docker docs](https://docs.docker.com/engine/install/linux-postinstall/) to ensure that $USER has root access to docker.

#### [docker-compose](https://docs.docker.com/compose/install/)

Install docker compose to declarativelly manage creation of docker images and containers. The docker compose statements are wrapped in `Makefile`


#### VS Code

In your local machine:

1. [install VS Code](https://code.visualstudio.com/docs/setup/linux),

2. use Crtl+Shift+P to access commands

3. In commands, search `Extensions: install extensions`and install:

   - Docker:`ms-azuretools.vscode-docker`

   - Remote SSH - to connect via SSH to Tenbu's Data Science Server

   - Python

   - [`ms-vscode-remote.remote-containers`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

   - [extra] Dracula is a good theme extension :wink:

   - [extra] Edit csv (good for handling .csv files)



#### [git](https://git-scm.com/download/linux)

```
sudo apt-get git
```

#### make

```
sudo apt-get update
sudo apt-get install build-essential
```

The development work is performed via `make` commands.

To see the most up to date list of available commands run

```bash
$ make help
```


---------------


