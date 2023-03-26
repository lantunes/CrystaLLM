CrystalGPT-dev
==============

To activate the venv:
```
$ source ../venvs/crystal_gpt_env/bin/activate
```

The `nanoGPT` module contains code lifted from `https://github.com/karpathy/nanoGPT`. 

To train a model, from the root of this project, do:
```shell
$ python nanoGPT/train.py config/train_cif.py --device=cpu --compile=False --eval_iters=20 \
--log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
--max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

To sample from a trained model, from the root of this project, do:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="data_NaCl"
```
alternatively:
```shell
$ python nanoGPT/sample.py --device=cpu --out_dir=out/cif_model --start="FILE:out/prompt.txt"
```
