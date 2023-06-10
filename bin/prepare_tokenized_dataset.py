import sys
sys.path.append(".")
import os
import numpy as np
import random
import gzip
import multiprocessing as mp
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

from lib import get_cif_tokenizer, array_split


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def tokenize(chunk_of_cifs, symmetrized, includes_props, queue=None):
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    tokenized = []
    for cif in tqdm(chunk_of_cifs, disable=queue is not None):
        if queue:
            queue.put(1)
        tokenized.append(tokenizer.tokenize_cif(cif))
    return tokenized


def preprocess(cifs_raw):
    cifs = []
    for cif in tqdm(cifs_raw):
        # filter out some lines in the CIF
        lines = cif.split('\n')
        cif_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                cif_lines.append(line)
        cif_lines.append("\n")
        cifs.append("\n".join(cif_lines))
    return cifs


if __name__ == '__main__':
    train_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.train.pkl.gz"
    val_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.val.pkl.gz"
    out_dir = "../out/mp_oqmd_nomad_cifs_semisymm_Z_props"
    symmetrized = True
    includes_props = True
    workers = 4

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with gzip.open(train_fname, "rb") as f:
        cifs_raw_train = pickle.load(f)

    with gzip.open(val_fname, "rb") as f:
        cifs_raw_val = pickle.load(f)

    # shuffle the order of the train CIFs
    random.shuffle(cifs_raw_train)

    cifs_train = preprocess(cifs_raw_train)
    cifs_val = preprocess(cifs_raw_val)

    # tokenize the train CIFs in parallel
    chunks = array_split(cifs_train, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers)
    watcher = pool.apply_async(progress_listener, (queue, len(cifs_train),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(tokenize, (chunk, symmetrized, includes_props, queue))
        jobs.append(job)

    tokenized_cifs_train = []
    for job in jobs:
        tokenized_cifs_train.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    lens = [len(t) for t in tokenized_cifs_train]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs_train]
    print(f"train min tokenized length: {np.min(lens):,}")
    print(f"train max tokenized length: {np.max(lens):,}")
    print(f"train mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"train total unk counts: {np.sum(unk_counts)}")

    # tokenize the validation CIFs
    tokenized_cifs_val = tokenize(cifs_val, symmetrized, includes_props)

    lens = [len(t) for t in tokenized_cifs_val]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs_val]
    print(f"val min tokenized length: {np.min(lens):,}")
    print(f"val max tokenized length: {np.max(lens):,}")
    print(f"val mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"val total unk counts: {np.sum(unk_counts)}")

    # create a single stream of tokens that will be the dataset
    train_data = []
    for t in tqdm(tokenized_cifs_train):
        train_data.extend(t)

    val_data = []
    for t in tqdm(tokenized_cifs_val):
        val_data.extend(t)

    # encode both to integers
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    train_ids = tokenizer.encode(train_data)
    val_ids = tokenizer.encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"vocab size: {len(tokenizer.token_to_id)}")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': len(tokenizer.token_to_id),
        'itos': tokenizer.id_to_token,
        'stoi': tokenizer.token_to_id,
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
