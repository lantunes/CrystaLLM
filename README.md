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
  - [Pre-processing the CIF Files](#pre-processing-the-cif-files)
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

The pre-assembled collection of CIF files which have been downloaded from the 
[Materials Project (MP)](materialsproject.org), the [OQMD](https://oqmd.org/), and 
[NOMAD](https://nomad-lab.eu/nomad-lab/) are contained in the `cifs_v1_orig.tar.gz` file. To download this file, 
execute the following command from the root of the cloned project:

```shell
$ python bin/download.py cifs_v1_orig.tar.gz
```

This archive contains 3,551,492 CIF files, each containing a unique filename assigned by us which indicates the 
origin of the file, which we refer to as its ID. For subsequent steps, we require that inputs be provided as a 
serialized Python list, in pickle format, because it is the most efficient format we found for working with over 
3 million CIF files in this context. Therefore, we provide a utility for converting the .tar.gz file to a .pkl.gz file:

```shell
$ python bin/tar_to_pickle.py cifs_v1_orig.tar.gz cifs_v1_orig.pkl.gz
```

The resulting .pkl.gz file contains a serialized Python list of 3,551,492 `(ID, CIF string)` 2-tuples. 
Alternatively, the `cifs_v1_orig.pkl.gz` can be downloaded directly:

```shell
python bin/download.py cifs_v1_orig.pkl.gz
```

However, please be aware that this .pkl.gz file is reliant on Python's serialization mechanism, and may not be 
compatible with future versions of Python.

_NOTE: These files are close to 700 MB in size._

_This dataset includes data from the [Materials Project](https://materialsproject.org/)._ 
> A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. 
Persson (*=equal contributions). The Materials Project: A materials genome approach to accelerating materials 
innovation. APL Materials, 2013, 1(1), 011002.

_This dataset includes data from the [OQMD database](http://oqmd.org/)._
>  J. E. Saal, S. Kirklin, M. Aykol, B. Meredig, and C. Wolverton. Materials Design and Discovery with 
High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD). JOM 65, 1501-1509 (2013).

_This dataset includes data from [NOMAD](https://nomad-lab.eu/nomad-lab/)._
> M. Scheidgen, L. Himanen, A. Ladines, D. Sikter, M. Nakhaee, Á. Fekete, T. Chang, A. Golparvar, J. Márquez, 
S. Brockhauser, S. Brückner, L. Ghiringhelli, F. Dietrich, D. Lehmberg, T. Denell, A. Albino 1, H. Näsström, S. Shabih, 
F. Dobener, M. Kühbach, R. Mozumder, J. Rudzinski, N. Daelman, J. Pizarro, M. Kuban, C. Salazar, P. Ondračka, 
H.-J. Bungartz, C. Draxl. NOMAD: A distributed web-based platform for managing materials science research data. Journal 
of Open Source Software, 8(90), 5388.

_This dataset is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)._

### Deduplicating the Original CIF Files

The original CIF dataset contains duplicates when combinations of cell composition and space group are considered. To 
disambiguate CIF files that have the same combination of cell composition and space group, we choose the CIF file with
the lowest volume per formula unit. To deduplicate the original CIF file dataset, execute the following command from 
the root of the cloned project:

```shell
$ python bin/deduplicate.py cifs_v1_orig.pkl.gz --out cifs_v1_dedup.pkl.gz
```

This will produce the `cifs_v1_dedup.pkl.gz` file, which contains a serialized Python list of 2,285,914 CIF strings, 
each as a 2-tuple, `(ID, CIF string)`, where every ID is unique.

Alternatively, the `cifs_v1_dedup.pkl.gz` file can be downloaded directly:

```shell
$ python bin/download.py cifs_v1_dedup.pkl.gz
```

The `cifs_v1_dedup.tar.gz` file can also be downloaded and converted locally to the `cifs_v1_dedup.pkl.gz` file using 
the `tar_to_pickle.py` script.

### Pre-processing the CIF Files

Before the CIF dataset can be used, it must be standardized and augmented. We refer to this step as _pre-processing_.
To pre-process the CIF dataset, execute the following command from the root of the cloned project:

```shell
$ python bin/preprocess.py cifs_v1_dedup.pkl.gz --out cifs_v1_preproc.pkl.gz --workers 4
```

This will produce the `cifs_v1_preproc.pkl.gz` file, which contains a serialized Python list of 2,285,719 augmented 
CIF strings. The number of processes can be specified with the `workers` argument, to speed up processing.

Alternatively, the `cifs_v1_preproc.pkl.gz` file can be downloaded directly:

```shell
$ python bin/download.py cifs_v1_preproc.pkl.gz
```

### Splitting the Dataset into Train, Validation and Test Sets

To split the CIF dataset into train, validation and test sets, execute the following command from the root of the 
cloned project:

```shell
$ python bin/split.py cifs_v1_preproc.pkl.gz \
--train_out cifs_v1_train.pkl.gz \
--val_out cifs_v1_val.pkl.gz \
--test_out cifs_v1_test.pkl.gz
```

This will produce the `cifs_v1_train.pkl.gz`, `cifs_v1_val.pkl.gz`, and `cifs_v1_test.pkl.gz` files. The 
`random_state`, `validation_size`, and `test_size` arguments can also be specified, but have default values of 
`20230610`, `0.10`, and `0.0045`, respectively.

The `cifs_v1_train.pkl.gz` file contains a serialized Python list of 2,047,889 CIF strings. The `cifs_v1_val.pkl.gz`
file contains a serialized Python list of 227,544 CIF strings. The `cifs_v1_test.pkl.gz` file contains a serialized 
Python list of 10,286 CIF strings.

Alternatively, the `cifs_v1_train.pkl.gz`, `cifs_v1_val.pkl.gz`, and `cifs_v1_test.pkl.gz` files can be downloaded 
directly, using, for example:

```shell
$ python bin/download.py cifs_v1_train.pkl.gz
```

### Tokenizing the Dataset

TODO

### Identifying CIF Start Indices

TODO

### Using Your Own CIF Files

TODO - from a user-provided directory of CIFs, prepare a .pkl.gz file like the original downloaded CIFs
- simply tar and gzip the directory of CIFs, and use `bin/tar_to_pickle.py` 

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

To run the unit tests:

```shell
$ python -m unittest discover tests
```

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
