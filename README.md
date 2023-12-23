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
  - [Downloading the Original CIF Files](#downloading-the-original-cif-files)
  - [Deduplicating the Original CIF Files](#deduplicating-the-original-cif-files)
  - [Pre-processing the Original CIF Files](#pre-processing-the-original-cif-files)
  - [Splitting the Dataset into Train, Validation and Test Sets](#splitting-the-dataset-into-train-validation-and-test-sets)
  - [Tokenizing the Dataset](#tokenizing-the-dataset)
  - [Identifying CIF Start Indices](#identifying-cif-start-indices)
  - [Using Your Own CIF Files](#using-your-own-cif-files)
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

- This project requires Python 3.9 or greater. 

- This project uses Poetry for dependency management. Install Poetry if it's 
not installed on your system, by following the instructions [here](https://python-poetry.org/docs/#installation).

### Creating a Local Environment

Perform the following steps to create and activate a local environment:

1. Create a Python virtual environment:

```shell
$ python -m venv crystallm_venv
```

2. Activate the virtual environment:

```shell
$ source crystallm_venv/bin/activate
```

### Installing Dependencies

Clone this repository to your local machine. Then, from the root of the cloned project, install the required packages 
by running:

```shell
$ poetry install
```

This command reads the `pyproject.toml` file, and installs all the dependencies in the virtual environment.

## Obtaining the Training Data

### Downloading the Original CIF Files

The pre-assembled collection of CIF files which have been downloaded from the Materials Project (MP), the OQMD, and 
NOMAD are contained in the `cifs_v1_orig.pkl.gz` file. To download this file, execute the following command from the 
root of the cloned project:

```shell
$ python bin/fetch_data.py cifs_v1_orig.pkl.gz
```
This file contains a serialized Python list of 3,551,492 CIF strings, each as a 2-tuple, `(ID, CIF string)`, where 
every ID is unique.

_NOTE: This file is over 600 MB in size._

### Deduplicating the Original CIF Files

TODO

### Pre-processing the Original CIF Files

TODO

### Splitting the Dataset into Train, Validation and Test Sets

TODO

### Tokenizing the Dataset

TODO

### Identifying CIF Start Indices

TODO

### Using Your Own CIF Files

TODO - from a user-provided directory of CIFs, script to prepare a .pkl.gz file like the original downloaded CIFs

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