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


def tokenize(chunk_of_cifs, symmetrized, includes_props, queue):
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    tokenized = []
    for cif in chunk_of_cifs:
        queue.put(1)
        tokenized.append(tokenizer.tokenize_cif(cif))
    return tokenized


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm_props.pkl.gz"
    out_dir = "../out/mp_oqmd_cifs_semisymm_props"
    symmetrized = True
    includes_props = True
    workers = 4

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with gzip.open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    # shuffle the order of the CIFS
    random.shuffle(cifs_raw)

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

    chunks = array_split(cifs, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers)
    watcher = pool.apply_async(progress_listener, (queue, len(cifs),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(tokenize, (chunk, symmetrized, includes_props, queue))
        jobs.append(job)

    tokenized_cifs = []
    for job in jobs:
        tokenized_cifs.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    lens = [len(t) for t in tokenized_cifs]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs]

    print(f"min tokenized length: {np.min(lens):,}")
    print(f"max tokenized length: {np.max(lens):,}")
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
