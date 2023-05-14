CrystaLLM
==============

To activate the venv:
```
$ source ../venvs/crystal_gpt_env/bin/activate
```

The `nanoGPT` module contains code lifted from `https://github.com/karpathy/nanoGPT`. 

To train a model on a macbook, from the root of this project, do:
```shell
$ python nanoGPT/train.py config/train_cif.py --device=cpu --compile=False
```
alternatively, on a GPU:
```shell
$ python nanoGPT/train.py config/train_cif.py --device=cuda
```
NOTE: Add `--dtype=float16` if an error occurs because of using `bfloat16`.

To sample from a trained model, from the root of this project, do:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="data_NaCl"
```
alternatively:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="FILE:out/prompt.txt"
```
