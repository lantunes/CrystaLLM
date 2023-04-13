import sys
sys.path.append(".")
import numpy as np
from lib import get_cif_tokenizer, CIFData, populate_cif_data
from tqdm import tqdm
import gzip
from sklearn.model_selection import train_test_split
try:
    import cPickle as pickle
except ImportError:
    import pickle


"""
We construct a training, validation, and test set for the VPFU (volume per formula unit)
fine-tuning task. We use the structures of the pre-training validation set, since the
model didn't see any of these structures during pre-training.
"""
if __name__ == '__main__':
    validation_set_fname = "../out/mp_oqmd_cifs_semisymm_Z_props/val.bin"
    out_file = "../out/mp_oqmd_cifs_semisymm_Z_props__vpfu.pkl.gz"
    symmetrized = True
    includes_props = True
    test_size = 0.125
    val_size = 0.10
    random_state = 20230413

    validation_set_ints = np.fromfile(validation_set_fname, dtype=np.uint16)
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    newline_id = tokenizer.token_to_id["\n"]

    id_to_token = tokenizer.id_to_token

    cif_datas = []

    curr_cif_data = None
    curr_cif_tokens = []

    for i in tqdm(validation_set_ints):
        token = id_to_token[i]

        if token == "data_":
            # populate the existing CIFData if it exists
            if curr_cif_data is not None:
                populate_cif_data(curr_cif_data, ''.join(curr_cif_tokens))
                cif_datas.append(curr_cif_data)

            # start a new CIFData
            curr_cif_data = CIFData()
            curr_cif_tokens = []

        curr_cif_tokens.append(token)
    # we may have a CIFData that's in progress, but we won't add it to cif_datas, since we
    #  don't know if it's complete (i.e. all the necessary tokens have been read)

    X = []
    y = []
    formula_to_idx = {}
    max_len = 0

    for cif_data in tqdm(cif_datas):
        if cif_data.formula_units > 0:
            comp = cif_data.composition
            vpfu = cif_data.cell_volume / cif_data.formula_units
            if comp in formula_to_idx:
                # we've seen this composition before; use the smallest VPFU
                idx = formula_to_idx[comp]
                if vpfu < y[idx]:
                    y[idx] = vpfu
            else:
                X_ids = tokenizer.encode(tokenizer.tokenize_cif(f"data_{comp}"))
                if len(X_ids) > max_len:
                    max_len = len(X_ids)
                X.append(X_ids)
                y.append(vpfu)
                formula_to_idx[comp] = len(X) - 1

    # pad the sequences with newlines
    for i in range(len(X)):
        X_ids = X[i]
        padding_length = max_len - len(X_ids)
        padding = [newline_id] * padding_length
        X[i] = X_ids + padding

    X_A, X_test, y_A, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(X_A, y_A, test_size=val_size, random_state=random_state)

    print(f"train size: {len(X_train)}")
    print(f"validation size: {len(X_val)}")
    print(f"test size: {len(X_test)}")

    with gzip.open(out_file, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
