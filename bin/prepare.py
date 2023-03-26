import os
import csv
import gzip
import numpy as np
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

from lib import tokenize_cif, encode, TOKEN_TO_ID, ID_TO_TOKEN


if __name__ == '__main__':
    fname = "../data/all_cif_structures.csv.gz"

    cifs = []

    with gzip.open(fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in tqdm(reader):
            mpid = row[0]
            cif = row[1].replace('"', '')

            # filter out some lines in the CIF
            lines = cif.split('\\n')
            cif_lines = []
            for line in lines:
                line = line.strip()
                if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                    cif_lines.append(line)
            cif_lines.append("\n")
            cifs.append("\n".join(cif_lines))

    tokenized_cifs = []
    for cif in tqdm(cifs):
        tokenized_cifs.append(tokenize_cif(cif))

    lens = [len(t) for t in tokenized_cifs]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs]

    # for i, t in enumerate(tokenized_cifs):
    #     if t.count("<unk>") > 0:
    #         print(cifs[i])

    print(f"mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"total unk counts: {np.sum(unk_counts)}")

    # create a single stream of tokens that will be the dataset
    data = []
    for t in tqdm(tokenized_cifs):
        data.extend(t)

    # create the train and test splits (90-10)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), '../out/train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), '../out/val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': len(TOKEN_TO_ID),
        'itos': ID_TO_TOKEN,
        'stoi': TOKEN_TO_ID,
    }
    with open(os.path.join(os.path.dirname(__file__), '../out/meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)