import sys
sys.path.append(".")
import numpy as np
from lib import get_cif_tokenizer
from tqdm import tqdm
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle


"""
We construct the evaluation set from the CIFs in the validation set,
since the model didn't directly see the CIFs of the validation set in 
training. 
"""
if __name__ == '__main__':
    validation_set_fname = "../out/mp_oqmd_cifs_semisymm_Z_props/val.bin"
    out_file = "../out/mp_oqmd_cifs_semisymm_Z_props_evalcifs.pkl.gz"
    symmetrized = True
    includes_props = True

    validation_set_ints = np.fromfile(validation_set_fname, dtype=np.uint16)
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)

    id_to_token = tokenizer.id_to_token

    cifs = []

    curr_cif_tokens = None

    for i in tqdm(validation_set_ints):
        token = id_to_token[i]

        if token == "data_":
            if curr_cif_tokens is not None and len(curr_cif_tokens) > 0:
                cifs.append(''.join(curr_cif_tokens))

            curr_cif_tokens = []

        if curr_cif_tokens is not None:
            curr_cif_tokens.append(token)
    # we may have a CIF that's in progress, but we won't add it to cifs, since we
    #  don't know if it's complete (i.e. all the necessary tokens have been read)

    with gzip.open(out_file, "wb") as f:
        pickle.dump(cifs, f)
