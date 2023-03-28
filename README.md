CrystalGPT-dev
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

Open questions:
- Are there permutation and/or rotation invariances that need to be taken into account
with these CIF files? For example, can we augment the dataset with more CIF files that contain
rotated versions of the original configuration?

- Can adding more data help? The `all_cif_structures.csv.gz` file contains only 47,737 structures 
from the Materials Project, that were used to construct the Ricci database. But there are over 80,000
structures in total in the Materials Project. Can we also use the structures in the OQMD (if they don't
exist in the Materials Project)? If the OQMD doesn't provide CIF files, perhaps we can convert from VASP
to CIF using `https://github.com/egplar/vasp2cif`. Can we get access to the ICSD?

- Does a block size of 3072 perform better than a block size of 2048? How do we evaluate one model vs another?
The loss achieved on the validation set? One way would be to withold a test set from pre-training, and then
feed the model a starting string, like "data_NaCl", and see what properties it predicts in generative mode.

- Can we get better performance by using more parameters (i.e. a bigger model)? Smaller batch size? Both?
- should the learning rate be `learning_rate = 6e-4`, `min_lr = 6e-5`? 
- what importance does regularization, via dropout, have? 

- Do we need to include all the whitespaces? Does the CIF format require multiple whitespaces in places, or
is a single whitespace enough? If we replace multiple consecutive whitespaces with a single whitespace, then 
the block sizes can get much smaller, and the number of tokens in the dataset is reduced. 

- Should we consider the space groups to be their own tokens? Right now, the space groups are comprised of tokens used
primarily for atomic symbols
- The Materials Project .cif files in `all_cif_structures.csv.gz` are apparently not symmetrized, and all space groups
are the same value of 'P 1'; but if we download a symmetrized .cif from the site, it will contain differnt space groups
- there are no oxidation states on the atoms in `all_cif_structures.csv.gz`
