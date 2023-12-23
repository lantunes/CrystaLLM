CrystaLLM
==============

CrystaLLM is a Transformer-based Large Language Model of the CIF (Crystallographic Information File) format. The model 
can be used to generate crystal structures, and is based on the [GPT-2 model](https://github.com/openai/gpt-2). This 
repository contains code that can be used to reproduce the experiments in the paper
_[Crystal Structure Generation with Autoregressive Large Language Modeling](https://arxiv.org/abs/2307.04340)_. The 
model definition, training, and inference code in this repository is derived from the code in the 
[nanoGPT](https://github.com/karpathy/nanoGPT) repository.

<img src="resources/crystallm-github.png" width="100%"/>

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Creating a Local Environment](#creating-a-local-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Obtaining the Training Data](#obtaining-the-training-data)
- [Training the Model](#training-the-model)
- [Generating Crystal Structures](#generating-crystal-structures)
  - [Using the Pre-trained Model](#using-the-pre-trained-model)
  - [Monte Carlo Tree Search Decoding](#monte-carlo-tree-search-decoding)
- [The Challenge Set](#the-challenge-set)
- [Tests](#tests)
- [Need Help?](#need-help)
- [Citation](#citation)

## Getting Started

### Prerequisites

This project requires Python 3.9 or greater.

### Creating a Local Environment

To work with the models, clone this repository to your local machine. Then perform the following steps to create and 
activate a local environment:

1. Create a Python virtual environment:

```shell
$ python -m venv crystallm_venv
```

2. Activate the virtual environment:

```shell
$ source crystallm_venv/bin/activate
```

### Installing Dependencies

This project uses Poetry for dependency management. Install the required packages by running:

```shell
$ pip install poetry
$ poetry install
```

This command reads the `pyproject.toml` file, and installs all the dependencies in the virtual environment.

## Obtaining the Training Data

TODO

## Training the Model

TODO

## Generating Crystal Structures

TODO

### Using the Pre-trained Model

TODO

### Monte Carlo Tree Search Decoding

TODO

## The Challenge Set

TODO 

## Tests

TODO

## Need Help?

If you encounter any issues, or have any questions, please feel free to open an issue in this repository.

## Citation

Please use the following bibtex entry:
```
@article{antunes2023crystal,
  title={Crystal Structure Generation with Autoregressive Large Language Modeling},
  author={Antunes, Luis M and Butler, Keith T and Grau-Crespo, Ricardo},
  journal={arXiv preprint arXiv:2307.04340},
  year={2023}
}
```