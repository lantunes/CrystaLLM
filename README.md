CrystalGPT-dev
==============

To activate the venv:
```
$ source ../venvs/crystal_gpt_env/bin/activate
```

The `nanoGPT` module contains code lifted from `https://github.com/karpathy/nanoGPT`. 

To train a model, from the root of this project, do:
```shell
$ python nanoGPT/train.py config/train_cif.py --device=cpu --compile=False
```

To sample from a trained model, from the root of this project, do:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="data_NaCl"
```
alternatively:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="FILE:out/prompt.txt"
```
